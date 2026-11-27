# 2D Steady-State Heat Conduction

Solves 2D steady-state heat conduction on the unit square with zero-temperature (Dirichlet) boundaries:

$$
-\nabla^2 T(x,y) + c(x,y)\,T(x,y) = f(x,y) \quad \text{in } (0,1)^2, \qquad T = 0 \text{ on } \partial\Omega
$$

- $c(x,y) = \text{base} + A_c \sin(\pi k_x x + \phi_x)\sin(\pi k_y y + \phi_y)$ is a spatially varying heat-loss coefficient.
- $f(x,y)$ is a sum of $M$ Gaussian heat sources ($q_m > 0$) and sinks ($q_m < 0$).

All random parameters are drawn from a seeded generator, so the same seed always gives the same problem.

## Model details

### Default parameters

With the default config, the coefficient and the sources are

$$
c(x,y) = 2.0 + A_c \sin(\pi k_x x + \phi_x)\sin(\pi k_y y + \phi_y), \qquad
f(x,y) = \sum_{m=1}^{5} q_m \exp\left(-\frac{(x - x_m)^2 + (y - y_m)^2}{2\sigma^2}\right), \quad \sigma = 0.1
$$

with the random parameters drawn independently as

$$
A_c \sim \mathcal{U}(0.5, 1.5), \quad k_x, k_y \sim \mathcal{U}\{1, 2, 3\}, \quad \phi_x, \phi_y \sim \mathcal{U}(0, 2\pi), \quad q_m \sim \mathcal{U}(-10, 10), \quad x_m, y_m \sim \mathcal{U}(0.15, 0.85)
$$

Since $|A_c| \le 1.5$, $c(x,y) \ge 0.5 > 0$ everywhere, which keeps $A$ symmetric positive definite for CG.

### Order of the random draws

The generator is `rng = numpy.random.default_rng(seed)`. Each parameter is drawn with a separate scalar call, in this order (shown with the default ranges; the code takes them from the config):

1. `A_c = rng.uniform(0.5, 1.5)`
2. `k_x = rng.integers(1, 4)`, then `k_y = rng.integers(1, 4)`
3. `phi_x = rng.uniform(0, 2*np.pi)`, then `phi_y = rng.uniform(0, 2*np.pi)`
4. For each source $m = 1, \dots, M$: `q_m = rng.uniform(-10, 10)`, `x_m = rng.uniform(0.15, 0.85)`, `y_m = rng.uniform(0.15, 0.85)`

Any reimplementation has to keep this order to reproduce the same problem for a given seed.

### Grid and vector ordering

The interior grid points are $(x_i, y_j) = (ih, jh)$ for $i, j = 1, \dots, n$, with $h = 1/(n+1)$. The boundary points ($i$ or $j$ equal to $0$ or $n+1$) are fixed at $T = 0$ and are not unknowns.

All vectors (`b.npy`, `x.npy`, the diagonal $c$) use the 0-based index map

$$
k = (i-1)\,n + (j-1)
$$

so $\mathbf{x} = [T_{1,1}, T_{1,2}, \dots, T_{1,n}, T_{2,1}, \dots, T_{n,n}]$. `x.npy` is the temperature at the interior points: `x.reshape(n, n)` gives the field with the first index along $x$, and padding it with a ring of zeros adds the boundary.

All arrays are stored and computed in `float64`, so that CG can reach the $10^{-8}$ residual tolerance.

### Resolution check

A run is flagged as under-resolved (a warning, and `"under_resolved": true` in `run_record.json`) when a source spans fewer than three grid steps, $\sigma < 3h$. With $\sigma = 0.1$ this flags $n \le 28$. The original specification uses the cutoff $n < 30$; the two differ only at $n = 29$, where $\sigma = 3h$ exactly.

## Files

| File | Purpose |
|---|---|
| `heat_conduction_2d.py` | The pipeline (assembly, solver, plot) |
| `heat_cond-config.yaml` | All adjustable parameters |
| `heat_cond-env.yaml` | micromamba/conda environment |

## Setup

Requires [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). Works on macOS, Linux and Windows.

```bash
micromamba create -f heat_cond-env.yaml
micromamba activate 2d_ss_heat_cond
```

The dependency versions are deliberately not pinned, so the environment resolves on as many machines as possible. For reproducible measurement runs, versions can be pinned.

### GPU (optional)

The solver runs on an NVIDIA GPU through CuPy when one is available. On a Linux or Windows machine with an NVIDIA GPU, uncomment `cupy` and `cuda-version` in `heat_cond-env.yaml` before creating the environment. Set `cuda-version` to match your machine: `nvidia-smi` shows the highest CUDA version the installed driver supports.

With `device: auto` (the default), the solve falls back to the CPU with a warning when CuPy or a GPU is missing (for example on macOS). With `device: gpu`, the script stops with an error instead. `device: cpu` always uses the CPU. The `device` field in `run_record.json` shows which device was actually used.

## Usage

Run the full pipeline with the parameters from `heat_cond-config.yaml`:

```bash
python heat_conduction_2d.py
```

`n`, `seed`, `outdir` and `device` can be overridden for a single run, and another config file can be used:

```bash
python heat_conduction_2d.py --n 128 --seed 3 --device cpu --outdir out_128
python heat_conduction_2d.py --config my-config.yaml
```

The pipeline runs three steps in sequence:

| Step | What it does | Output |
|---|---|---|
| `assemble` | Builds the sparse matrix $A$ and right-hand side $b$ (CPU) | `A.npz`, `b.npy` |
| `solve` | Solves $Ax = b$ with Conjugate Gradient (GPU or CPU) | `x.npy`, `run_record.json` |
| `postprocess` | Plots the temperature field (CPU) | `temperature_map.png` |

All outputs are written to `outdir` (default `out/`).

## Configuration

`heat_cond-config.yaml` contains every adjustable parameter, with comments:

- `run`: grid size `n` (interior points per dimension, $N = n^2$ unknowns), `seed`, `outdir`, `device` (`auto`, `gpu` or `cpu`)
- `dissipation`: the parameters of $c(x,y)$ and the ranges its random values are drawn from
- `sources`: number of sources $M$, Gaussian width $\sigma$, and the ranges for source strength and position
- `solver`: relative residual tolerance and maximum number of iterations
- `plot`: colormap and resolution

Invalid values (for example a range with `lo > hi`, or parameters that would make $c(x,y)$ negative) are rejected with a message before anything runs.

## Solver and run record

The system is solved with unpreconditioned Conjugate Gradient from $x_0 = 0$, using `cupyx.scipy.sparse.linalg.cg` on the GPU and `scipy.sparse.linalg.cg` on the CPU. The solve stops when the relative residual $\|Ax - b\|_2 / \|b\|_2$ drops below `solver.tolerance` (default $10^{-8}$) or after `solver.max_iterations` iterations.

Every solve writes `run_record.json`:

```json
{
  "n": 64,
  "seed": 0,
  "under_resolved": false,
  "converged": true,
  "iterations": 213,
  "final_relative_residual": 9.87e-09,
  "tolerance": 1e-08,
  "max_iterations": 10000,
  "device": "cpu",
  "N": 4096,
  "solve_time_s": 0.004
}
```

- `seed` is the seed actually used. With `seed: null`, a random seed is drawn and recorded here, so the run can be reproduced.
- `under_resolved` is `true` when the source width is smaller than three grid steps ($\sigma < 3h$).
- `final_relative_residual` is the true residual, recomputed after the solve.
- `device` is `cpu` or `gpu:<GPU name>`.
- `solve_time_s` covers the CG solve and the final residual check, not file loading, transfers to or from the GPU, or plotting.

If the solve does not converge, no `x.npy` or plot is saved (old ones in `outdir` are removed), and the script exits with code 1.
