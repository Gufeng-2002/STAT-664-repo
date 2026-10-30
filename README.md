# Quantile breakpoint detection — bounded heteroskedastic simulation

A reproducible Monte Carlo study of testing for slope changes in a conditional quantile regression. The response is bounded in `(0,1)` using an `expit(logit(m(X)) + sigma(X) * Z)` data-generating process with shifted Student-t noise and heteroskedastic scale.

## Contents

- [`Quantile_Breakpoint_Simulation.ipynb`](Quantile_Breakpoint_Simulation.ipynb) — fully executed notebook including code, tables and plots.
- [`images/`](images/) — nine original 300-dpi PNG figures exported from the notebook.

## Method

- True target quantile: `tau* = 0.8`.
- True conditional quantile: `m(x) = beta0 + beta1*x + sum_j delta_j * (x - kappa_j)_+`.
- Default breakpoints: `kappa = (0.40, 0.70)`; strength `delta = (-0.90, 0.90)`.
- Null: no slope changes. Alternative: one or more slope changes.
- Detection statistic: `T_loss = S0 - min_{0 <= k <= K_test} S_k` using pinball loss.
- Calibrate `c_alpha` by independent Monte Carlo draws under the null, then reject if `T_loss > c_alpha`.
- Analyze power against slope-change strength, sample size, and probability `p` of sampling `X` within `w` of a kink.

The general model can have K breakpoints; the supplied notebook implements `K = K_test = 2`. See the opening notation table and Sections 0–8 of the notebook.

## Reproduce

Python packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `joblib`, `ipython`, `jupyter`.

```bash
python -m pip install -r requirements.txt
jupyter notebook Quantile_Breakpoint_Simulation.ipynb
```

Run cells in order. The notebook uses a fixed random-seed strategy and saves PNG figures to `images/`. Full Monte Carlo runs can be compute-intensive; set `RUN_MONTE_CARLO=False` for quick checks of data generation and fitting.

**Note:** These are synthetic simulation findings. The calibrated Type I error guarantee depends on reproducing the same null design and full detector configuration in calibration and evaluation.
