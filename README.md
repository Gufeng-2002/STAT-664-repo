# Quantile breakpoint detection — bounded heteroskedastic simulation

A reproducible Monte Carlo study of testing for slope changes in a conditional quantile regression. The response is bounded in `(0,1)` using an `expit(logit(m(X)) + sigma(X) * Z)` data-generating process with shifted Student-t noise and heteroskedastic scale.

## Contents

- [`Synthetic_Data_Demo.ipynb`](Synthetic_Data_Demo.ipynb) — quick examples of data generation, configuration and plotting, with complete DataFrames available for every example.
- [`agg_rank_tests/ART_Quantile_Breakpoints.ipynb`](agg_rank_tests/ART_Quantile_Breakpoints.ipynb) — exploratory global/local ART rank-CUSUM tests using fitted quantile residuals and indicators, with permutation calibration and null/power checks.
- [`synthetic_data/generator.py`](synthetic_data/generator.py) — reusable `QuantileDataGenerator` class (about 240 lines, including documentation and plotting).
- [`synthetic_data/Quantile_Breakpoint_Simulation.ipynb`](synthetic_data/Quantile_Breakpoint_Simulation.ipynb) — original simulation study, now using the shared generator. Its saved Monte Carlo outputs are from the original run; the full study has not been rerun after extraction.
- [`tests/test_generator.py`](tests/test_generator.py) — reproducibility, model, sampling and plotting checks.

## Generate a dataset

From the repository root:

```python
from synthetic_data import QuantileDataGenerator

model = QuantileDataGenerator(
    n=150, tau=0.8,
    breakpoints=(0.4, 0.7), slope_changes=(-0.9, 0.9),
    beta0=0.65, beta1=-0.2,
    sigma=(0.4, 0.6), df=8,
    near_share=0.05, half_width=0.05, seed=2026,
)
data = model.generate()           # Complete pandas DataFrame, no figure.
data_with_plot = model.generate(plot=True)  # DataFrame plus displayed figure.
ax = model.plot(data)             # Plot an existing dataset without resampling.
# ax.figure.savefig("example.png", dpi=300, bbox_inches="tight")
```

| Setting | Meaning |
|---|---|
| `n` | Sample size; override with `generate(n=500)` |
| `tau` | Quantile level whose true curve is `m(x)` |
| `breakpoints`, `slope_changes` | Any K ordered knots, with one slope change per knot; use `(), ()` for K=0 |
| `beta0`, `beta1` | Intercept and initial slope; segment slopes accumulate the changes |
| `sigma` | Positive noise scales at the left and right endpoints, linear **on the logit scale** |
| `noise`, `df` | `"t"` (default) or `"normal"`; Student-t degrees of freedom |
| `x_range`, `x_beta` | Predictor support and scaled Beta design; `(1,1)` is uniform |
| `half_width` | Scalar window half-width or a tuple with one width per breakpoint |
| `near_share` | Probability of falling in the **union** of the windows; `None` keeps the original Beta design |
| `seed` | Initializes the object's stream; `generate(seed=123)` reproduces a particular dataset |

`near_share` is a probability, not an exact count or a probability per knot. Windows are merged if they overlap and clipped to the support. Inside and outside their union, sampling preserves the conditional Beta shape using inverse-CDF sampling, avoiding rejection loops for narrow windows. Impossible sampling configurations raise `ValueError`.

Every DataFrame contains `X`, `Y`, `m_true`, `m_null`, `sigma`, `Z` and `near_breakpoint`; configuration is in `data.attrs`. `Z` is the shifted residual on the logit scale. The response remains bounded through the logit/expit construction; extreme residuals can round to 0 or 1 in floating-point arithmetic. No responses are clipped or redrawn.

For specified predictor values, use `model.generate(x=[0.1, 0.4, 0.7], seed=123)`. Evaluate the model directly with `model.location(x)`, `model.scale(x)` and `model.true_quantile(x, tau=0.9)`. The illustration's solid target curve and dashed null line are known model curves. Other conditional quantiles generally are not piecewise linear. Add them with `model.plot(data, quantiles=(0.1, 0.9))`.

To import from another Python environment, install the local repository:

```bash
python -m pip install "/path/to/STAT-664-repo[plot]"
```

Alternatively, copy the `synthetic_data` folder into your project. Importing it does not execute the original notebook. Core generation requires only NumPy, pandas and SciPy; plotting additionally requires Matplotlib.

## Method

- True target quantile: `tau* = 0.8`.
- True conditional quantile: `m(x) = beta0 + beta1*x + sum_j delta_j * (x - kappa_j)_+`.
- Default breakpoints: `kappa = (0.40, 0.70)`; strength `delta = (-0.90, 0.90)`.
- Null: no slope changes. Alternative: one or more slope changes.
- Detection statistic: `T_loss = S0 - min_{0 <= k <= K_test} S_k` using pinball loss.
- Calibrate `c_alpha` by independent Monte Carlo draws under the null, then reject if `T_loss > c_alpha`.
- Analyze power against slope-change strength, sample size, and probability `p` of sampling `X` within `w` of a kink.

The generator supports K=0, 1, 2 or more breakpoints. The original study's default is `K = K_test = 2`, and its detector experiments retain their original configuration and scope. See its opening notation table and Sections 0–8.

## Reproduce

Python packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `joblib`, `ipython`, `jupyter`.

```bash
python -m pip install -r requirements.txt
jupyter notebook Synthetic_Data_Demo.ipynb
```

Run cells in order. For the original study, open `synthetic_data/Quantile_Breakpoint_Simulation.ipynb`. It saves PNG figures to `images/` relative to its working directory. Full Monte Carlo runs can be compute-intensive; set `RUN_MONTE_CARLO=False` for quick checks of data generation and fitting.

Run the generator checks from the repository root with `python -m unittest discover -s tests -v`.

**Note:** These are synthetic simulation findings. The calibrated Type I error guarantee depends on reproducing the same null design and full detector configuration in calibration and evaluation.
