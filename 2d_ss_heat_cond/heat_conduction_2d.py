#!/usr/bin/env python
"""2D steady-state heat conduction on the unit square.

    -lap(T) + c(x,y) T = f(x,y)  in (0,1)^2,   T = 0 on the boundary

All parameters are read from a YAML config file (default: heat_cond-config.yaml next to this
script). n, seed, outdir and device can be overridden on the command line.

    python heat_conduction_2d.py                       # uses heat_cond-config.yaml
    python heat_conduction_2d.py --config my.yaml      # another config file
    python heat_conduction_2d.py --n 128 --seed 3      # override single values

The pipeline runs three processes in sequence:

    1. assemble      -> A.npz, b.npy
    2. solve         -> x.npy, run_record.json
    3. postprocess   -> temperature_map.png

The solver uses CuPy's CG (cupyx.scipy.sparse.linalg.cg) on an NVIDIA GPU when available,
and otherwise falls back to SciPy's CG (scipy.sparse.linalg.cg) on the CPU.
The device used is recorded.
"""

import argparse
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "heat_cond-config.yaml"
DEVICES = ("auto", "gpu", "cpu")
RANGE_KEYS = [("dissipation", "amplitude_range"), ("dissipation", "frequency_range"),
              ("dissipation", "phase_range_pi"), ("sources", "strength_range"),
              ("sources", "position_range")]
# Numeric config entries and their type. PyYAML reads e.g. `1e-8` as a string, so these
# are converted explicitly when the config is loaded.
INT_KEYS = [("run", "n"), ("run", "seed"), ("sources", "count"),
            ("solver", "max_iterations"), ("plot", "dpi"), ("dissipation", "frequency_range")]
FLOAT_KEYS = [("dissipation", "base"), ("dissipation", "amplitude_range"),
              ("dissipation", "phase_range_pi"), ("sources", "sigma"),
              ("sources", "strength_range"), ("sources", "position_range"),
              ("solver", "tolerance")]


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr)


def is_under_resolved(n, sigma):
    """The grid is under-resolved if a source (width sigma) spans fewer than 3 grid steps h."""
    return sigma < 3.0 / (n + 1)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def load_config(path, overrides):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["run"].update({k: v for k, v in overrides.items() if v is not None})
    cfg["run"]["outdir"] = Path(cfg["run"]["outdir"])
    convert_numbers(cfg)
    validate_config(cfg)
    if cfg["run"]["seed"] is None:
        # Draw a concrete random seed, so the run can still be reproduced from the record
        cfg["run"]["seed"] = np.random.SeedSequence().entropy
    return cfg


def to_number(value, kind, name):
    """Convert a config value (number or numeric string such as '1e-8') to int or float."""
    if kind is int and isinstance(value, int) and not isinstance(value, bool):
        return value  # keep exact (large seeds would lose precision through float)
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"Invalid config: {name} must be a number, got {value!r}")
    if kind is int:
        if not x.is_integer():
            raise SystemExit(f"Invalid config: {name} must be an integer, got {value!r}")
        return int(x)
    return x


def convert_numbers(cfg):
    for kind, keys in ((int, INT_KEYS), (float, FLOAT_KEYS)):
        for sec, key in keys:
            value, name = cfg[sec][key], f"{sec}.{key}"
            if key == "seed" and value is None:
                continue  # seed: null -> a random seed is drawn in load_config
            if isinstance(value, list):
                cfg[sec][key] = [to_number(v, kind, name) for v in value]
            else:
                cfg[sec][key] = to_number(value, kind, name)


def validate_config(cfg):
    def check(cond, msg):
        if not cond:
            raise SystemExit(f"Invalid config: {msg}")

    run, dis, src, sol = cfg["run"], cfg["dissipation"], cfg["sources"], cfg["solver"]
    check(run["n"] >= 1, "run.n must be >= 1")
    check(run["seed"] is None or run["seed"] >= 0, "run.seed must be >= 0 (or null)")
    check(run["device"] in DEVICES, "run.device must be auto, gpu or cpu")
    for sec, key in RANGE_KEYS:
        r = cfg[sec][key]
        check(len(r) == 2 and r[0] <= r[1], f"{sec}.{key} must be [lo, hi] with lo <= hi")
    check(dis["base"] - max(abs(a) for a in dis["amplitude_range"]) > 0,
          "dissipation.base - max|amplitude| must be > 0 (c must stay positive for CG)")
    check(src["count"] >= 0, "sources.count must be >= 0")
    check(src["sigma"] > 0, "sources.sigma must be > 0")
    check(sol["tolerance"] > 0, "solver.tolerance must be > 0")
    check(sol["max_iterations"] >= 1, "solver.max_iterations must be >= 1")
    # Plot settings are checked here, so a typo fails before the solve
    import matplotlib
    check(cfg["plot"]["colormap"] in matplotlib.colormaps,
          f"plot.colormap '{cfg['plot']['colormap']}' is not a matplotlib colormap")
    check(cfg["plot"]["dpi"] > 0, "plot.dpi must be > 0")


# --------------------------------------------------------------------------- #
# Process 1: System assembly (CPU)
# --------------------------------------------------------------------------- #
def draw_parameters(seed, dis, src):
    """Draw all random parameters with scalar calls in order."""
    rng = np.random.default_rng(seed)
    k_lo, k_hi = (int(k) for k in dis["frequency_range"])
    ph_lo, ph_hi = (p * np.pi for p in dis["phase_range_pi"])

    p = dict(A_c=rng.uniform(*dis["amplitude_range"]))
    p["k_x"] = int(rng.integers(k_lo, k_hi + 1))
    p["k_y"] = int(rng.integers(k_lo, k_hi + 1))
    p["phi_x"] = rng.uniform(ph_lo, ph_hi)
    p["phi_y"] = rng.uniform(ph_lo, ph_hi)
    p["sources"] = [(rng.uniform(*src["strength_range"]),   # q_m
                     rng.uniform(*src["position_range"]),   # x_m
                     rng.uniform(*src["position_range"]))   # y_m
                    for _ in range(int(src["count"]))]
    return p


def assemble(cfg):
    run, dis, src = cfg["run"], cfg["dissipation"], cfg["sources"]
    n, seed, outdir = int(run["n"]), run["seed"], run["outdir"]
    sigma = float(src["sigma"])
    h = 1.0 / (n + 1)

    if is_under_resolved(n, sigma):
        warn(f"n={n}: grid is under-resolved (sigma={sigma} < 3h = {3 * h:.4f}).")

    p = draw_parameters(seed, dis, src)

    # 2D Laplacian as Kronecker sum of the 1D second-derivative matrix (zero Dirichlet BCs)
    D1 = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)) / h**2
    I = sp.identity(n)
    A_base = sp.kron(I, D1) + sp.kron(D1, I)

    # Interior grid; indexing="ij" gives X[i-1, j-1] = x_i, so ravel() (C order)
    # gives k = (i-1) n + (j-1)
    pts = np.arange(1, n + 1) * h
    X, Y = np.meshgrid(pts, pts, indexing="ij")

    c = dis["base"] + p["A_c"] * np.sin(np.pi * p["k_x"] * X + p["phi_x"]) \
                               * np.sin(np.pi * p["k_y"] * Y + p["phi_y"])
    f = np.zeros_like(X)
    for q, xm, ym in p["sources"]:
        f += q * np.exp(-((X - xm) ** 2 + (Y - ym) ** 2) / (2 * sigma**2))

    A = (A_base + sp.diags(c.ravel())).tocsr()
    b = f.ravel()

    outdir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(outdir / "A.npz", A)
    np.save(outdir / "b.npy", b)
    print(f"[assemble] n={n} N={n * n} nnz={A.nnz} seed={seed} -> {outdir}/A.npz, b.npy")


# --------------------------------------------------------------------------- #
# Process 2: Iterative solver (GPU, with CPU fallback)
# --------------------------------------------------------------------------- #
def get_backend(device):
    """Return (array module, CSR constructor, CG solver, device->host copy, device name)."""
    if device != "cpu":
        try:
            import cupy as cp
            from cupyx.scipy.sparse import csr_matrix
            from cupyx.scipy.sparse.linalg import cg
            if cp.cuda.runtime.getDeviceCount() > 0:
                name = cp.cuda.runtime.getDeviceProperties(0)["name"]
                name = name.decode() if isinstance(name, bytes) else name
                return cp, csr_matrix, cg, cp.asnumpy, f"gpu:{name}"
        except Exception as e:  # ImportError, or CUDA driver/runtime problems
            if device == "gpu":
                raise SystemExit(f"device 'gpu' requested but CuPy/CUDA unavailable: {e}")
        warn("no CUDA GPU/CuPy found, running CG on the CPU.")
    from scipy.sparse.linalg import cg
    return np, sp.csr_matrix, cg, np.asarray, "cpu"


def conjugate_gradient(cg, A, b, xp, tol, maxiter):
    """Solve Ax = b from x0 = 0 with the library CG (SciPy on CPU, CuPy on GPU).

    Returns (x, iterations, true relative residual ||b - Ax|| / ||b||).
    """
    b_norm = float(xp.linalg.norm(b))
    if b_norm == 0.0:  # no sources: x = 0 is exact (CuPy's CG would divide 0/0 here)
        return xp.zeros_like(b), 0, 0.0

    iterations = 0

    def count(_xk):  # called by the library once per CG iteration
        nonlocal iterations
        iterations += 1

    # Newer SciPy names the relative tolerance `rtol`, older SciPy and CuPy call it `tol`.
    tol_kw = "rtol" if "rtol" in inspect.signature(cg).parameters else "tol"
    # atol=0: stop only on the relative criterion ||r|| <= tol * ||b||
    x, _info = cg(A, b, x0=xp.zeros_like(b), maxiter=maxiter, atol=0.0,
                  callback=count, **{tol_kw: tol})

    rel_res = float(xp.linalg.norm(b - A @ x)) / b_norm  # true residual, not the recursive one
    return x, iterations, rel_res


def solve(cfg):
    outdir = cfg["run"]["outdir"]
    tol = float(cfg["solver"]["tolerance"])
    maxiter = int(cfg["solver"]["max_iterations"])

    A_cpu = sp.load_npz(outdir / "A.npz").tocsr().astype(np.float64)
    b_cpu = np.load(outdir / "b.npy").astype(np.float64)

    xp, csr_matrix, cg, to_host, dev_name = get_backend(cfg["run"]["device"])
    A, b = csr_matrix(A_cpu), xp.asarray(b_cpu)  # host -> device (no-op on CPU)

    t0 = time.perf_counter()
    x, iters, rel_res = conjugate_gradient(cg, A, b, xp, tol, maxiter)
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0

    converged = rel_res < tol
    n = math.isqrt(b_cpu.size)
    record = {
        "n": n,
        "seed": cfg["run"]["seed"],
        "under_resolved": is_under_resolved(n, cfg["sources"]["sigma"]),
        "converged": converged,
        "iterations": iters,
        "final_relative_residual": rel_res,
        "tolerance": tol,
        "max_iterations": maxiter,
        "device": dev_name,
        "N": b_cpu.size,
        "solve_time_s": elapsed,
    }
    with open(outdir / "run_record.json", "w") as fh:
        json.dump(record, fh, indent=2)

    x_path = outdir / "x.npy"
    if converged:
        np.save(x_path, to_host(x))  # device -> host
        print(f"[solve] converged on {dev_name} in {iters} iterations ({elapsed:.3f} s), "
              f"rel. residual {rel_res:.3e} -> {x_path}")
    else:
        # Don't leave a solution or plot from an earlier run next to this run's record
        x_path.unlink(missing_ok=True)
        (outdir / "temperature_map.png").unlink(missing_ok=True)
        print(f"[solve] NOT converged on {dev_name} after {iters} iterations ({elapsed:.3f} s), "
              f"rel. residual {rel_res:.3e}; solution not saved.", file=sys.stderr)
    return record


# --------------------------------------------------------------------------- #
# Process 3: Post-processing and visualization (CPU)
# --------------------------------------------------------------------------- #
def postprocess(cfg):
    import matplotlib
    matplotlib.use("Agg")  # headless, works on all OSes / clusters
    import matplotlib.pyplot as plt

    n, outdir = int(cfg["run"]["n"]), cfg["run"]["outdir"]
    x = np.load(outdir / "x.npy")
    if x.size != n * n:
        raise SystemExit(f"x.npy has {x.size} entries, expected n^2 = {n * n}")
    T_full = np.pad(x.reshape(n, n), 1)  # add the T = 0 boundary

    vmax = float(np.abs(T_full).max()) or 1.0  # symmetric around T = 0 (sources vs sinks)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(T_full.T, origin="lower", extent=[0, 1, 0, 1],
                   cmap=cfg["plot"]["colormap"], vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label="T")
    ax.set(xlabel="x", ylabel="y", title=f"Steady-state temperature (n={n})")
    fig.tight_layout()
    out = outdir / "temperature_map.png"
    fig.savefig(out, dpi=cfg["plot"]["dpi"])
    plt.close(fig)
    print(f"[postprocess] -> {out}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help=f"YAML config file (default: {DEFAULT_CONFIG.name} next to the script)")
    ap.add_argument("--n", type=int, help="override run.n")
    ap.add_argument("--seed", type=int, help="override run.seed")
    ap.add_argument("--outdir", type=Path, help="override run.outdir")
    ap.add_argument("--device", choices=DEVICES, help="override run.device")
    args = ap.parse_args()

    overrides = {k: getattr(args, k) for k in ("n", "seed", "outdir", "device")}
    cfg = load_config(args.config, overrides)

    assemble(cfg)
    if not solve(cfg)["converged"]:
        sys.exit(1)
    postprocess(cfg)


if __name__ == "__main__":
    main()
