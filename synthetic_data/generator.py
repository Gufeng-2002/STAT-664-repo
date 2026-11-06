"""Location–scale–shape simulation extracted from Quantile_Breakpoint_Simulation.

Y = expit(logit(m(X)) + sigma(X) * Z), where Q_Z(tau) = 0.
Only numpy, pandas and scipy are needed; plotting imports matplotlib on demand.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit


@dataclass(frozen=True)
class QuantileDataGenerator:
    """Configure once, then call generate() to return a DataFrame.

    m(x) = beta0 + beta1*x + sum(delta_j * max(x-kappa_j, 0)).
    Use empty breakpoints and slope_changes for K=0. sigma gives the
    logit-scale noise at the two ends of x_range. noise is 't' or 'normal'.
    near_share is P(X in the UNION of breakpoint windows), not a fixed count
    or a probability per breakpoint; None keeps the original Beta design.
    half_width is a scalar or one width per breakpoint. Overlapping windows
    are merged and clipped to x_range. Within each region the Beta shape is
    preserved. seed initializes a stream; successive calls draw new datasets.
    """

    n: int = 150
    tau: float = 0.8
    breakpoints: tuple = (0.4, 0.7)
    slope_changes: tuple = (-0.9, 0.9)
    beta0: float = 0.65
    beta1: float = -0.2
    sigma: tuple = (0.4, 0.6)
    df: float = 8.0
    noise: str = "t"
    x_range: tuple = (0.0, 1.0)
    x_beta: tuple = (1.0, 1.0)
    near_share: float | None = None
    half_width: float | tuple = 0.05
    seed: int | None = None
    _rng: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self._check_n(self.n)
        for name in ("breakpoints", "slope_changes", "sigma", "x_range", "x_beta"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError(f"{name} must be a finite one-dimensional sequence.")
            object.__setattr__(self, name, tuple(values.tolist()))
        if any(len(getattr(self, name)) != 2 for name in ("sigma", "x_range", "x_beta")):
            raise ValueError("sigma, x_range and x_beta each need two values.")
        lo, hi = self.x_range
        k = np.asarray(self.breakpoints)
        if not lo < hi or not 0 < self.tau < 1:
            raise ValueError("Require x_range[0] < x_range[1] and 0 < tau < 1.")
        if len(k) != len(self.slope_changes) or np.any(np.diff(k) <= 0):
            raise ValueError("Provide ordered, distinct breakpoints and one slope change per breakpoint.")
        if np.any((k <= lo) | (k >= hi)):
            raise ValueError("Breakpoints must be strictly inside x_range.")
        if not np.isfinite([self.beta0, self.beta1, self.df]).all() or self.df <= 0:
            raise ValueError("beta0 and beta1 must be finite; df must be finite and positive.")
        if min(self.sigma) <= 0 or min(self.x_beta) <= 0:
            raise ValueError("Noise scales and Beta shapes must be positive.")
        if self.noise not in ("t", "normal"):
            raise ValueError("noise must be 't' or 'normal'.")
        widths = np.asarray(self.half_width, dtype=float)
        if widths.ndim > 1 or (widths.ndim == 1 and len(widths) != len(k)):
            raise ValueError("half_width must be a scalar or one value per breakpoint.")
        if not np.isfinite(widths).all() or np.any(widths < 0):
            raise ValueError("Window half-widths must be finite and nonnegative.")
        object.__setattr__(self, "half_width", float(widths) if widths.ndim == 0 else tuple(widths.tolist()))
        if self.near_share is not None:
            if not np.isfinite(self.near_share) or not 0 <= self.near_share <= 1:
                raise ValueError("near_share must be None or a probability in [0, 1].")
            inside = self.windows()
            covered = sum(b - a for a, b in inside)
            if (self.near_share > 0 and not inside) or (self.near_share < 1 and covered >= hi - lo):
                raise ValueError("The requested sampling probability assigns mass to an empty region.")
        # A piecewise-linear curve attains its extrema at endpoints or knots.
        m = self.location(np.r_[lo, k, hi])
        if not np.isfinite(m).all() or np.any((m <= 0) | (m >= 1)):
            raise ValueError("m(x) must stay strictly inside (0, 1) over the entire x_range.")
        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))

    @staticmethod
    def _check_n(n):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a nonnegative integer.")

    @property
    def distribution(self):
        """Unshifted, unit-scale residual distribution."""
        return stats.t(self.df) if self.noise == "t" else stats.norm()

    def location(self, x):
        """True conditional tau-quantile m(x); accepts scalars or arrays."""
        x = np.asarray(x, dtype=float)
        return self.null_curve(x) + sum(d * np.maximum(x - k, 0) for k, d in zip(self.breakpoints, self.slope_changes))

    def null_curve(self, x):
        """Reference straight line with all slope changes set to zero."""
        return self.beta0 + self.beta1 * np.asarray(x, dtype=float)

    def scale(self, x):
        """Linear heteroskedastic scale on the logit scale."""
        lo, hi = self.x_range
        return self.sigma[0] + (self.sigma[1] - self.sigma[0]) * (np.asarray(x) - lo) / (hi - lo)

    def true_quantile(self, x, tau=None):
        """Exact conditional quantile at any probability; defaults to self.tau."""
        tau = np.asarray(self.tau if tau is None else tau, dtype=float)
        if np.any(~np.isfinite(tau) | (tau <= 0) | (tau >= 1)):
            raise ValueError("Quantile probabilities must be in (0, 1).")
        shift = self.distribution.ppf(tau) - self.distribution.ppf(self.tau)
        return expit(logit(self.location(x)) + self.scale(x) * shift)

    def windows(self):
        """Merged, clipped breakpoint windows as (left, right) pairs."""
        lo, hi = self.x_range
        widths = np.broadcast_to(self.half_width, (len(self.breakpoints),))
        intervals = sorted((max(lo, k - w), min(hi, k + w)) for k, w in zip(self.breakpoints, widths) if w > 0)
        merged = []
        for a, b in intervals:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        return merged

    def near_breakpoints(self, x):
        """Membership in the union of the sampling windows."""
        x = np.asarray(x, dtype=float)
        near = np.zeros(x.shape, dtype=bool)
        for a, b in self.windows():
            near |= (x >= a) & (x <= b)
        return near

    def _draw_intervals(self, count, intervals, rng):
        """Sample a Beta design conditional on an interval union, without rejection."""
        lo, hi = self.x_range
        edges = (np.asarray(intervals) - lo) / (hi - lo)
        base = stats.beta(*self.x_beta)
        cdf, sf = base.cdf(edges), base.sf(edges)
        tail = cdf[:, 0] > 0.5
        masses = np.where(tail, sf[:, 0] - sf[:, 1], cdf[:, 1] - cdf[:, 0])
        if masses.sum() <= 0:
            raise ValueError("The requested region has numerically zero Beta probability; adjust x_beta or windows.")
        index = rng.choice(len(intervals), size=count, p=masses / masses.sum())
        u = rng.random(count) * masses[index]
        upper = tail[index]
        values = np.empty(count)
        values[~upper] = base.ppf(cdf[index[~upper], 0] + u[~upper])
        values[upper] = base.isf(sf[index[upper], 0] - u[upper])
        return lo + (hi - lo) * values

    def draw_x(self, n, *, rng=None):
        """Draw predictors; near_share controls an iid probability, not exact counts."""
        self._check_n(n)
        rng = self._rng if rng is None else rng
        lo, hi = self.x_range
        if self.near_share is None:
            return lo + (hi - lo) * rng.beta(*self.x_beta, size=n)
        inside = self.windows()
        outside, cursor = [], lo
        for a, b in inside:
            if cursor < a:
                outside.append((cursor, a))
            cursor = b
        if cursor < hi:
            outside.append((cursor, hi))
        near = rng.random(n) < self.near_share
        x = np.empty(n)
        for mask, intervals in ((near, inside), (~near, outside)):
            if mask.any():
                x[mask] = self._draw_intervals(mask.sum(), intervals, rng)
        return x

    def generate(self, n=None, *, x=None, seed=None, plot=False):
        """Return columns X, Y, m_true, m_null, sigma, Z, near_breakpoint.

        n overrides the configured sample size. Alternatively x supplies fixed
        one-dimensional predictors, bypassing the random X design. An explicit
        seed reproduces one dataset without advancing the object's stream.
        plot=True also displays the illustration; the return is still a DataFrame.
        The theoretical Y support is (0,1); extreme noise can round to endpoints
        in floating-point arithmetic. Responses are never clipped or redrawn.
        """
        rng = self._rng if seed is None else np.random.default_rng(seed)
        fixed_x = x is not None
        if x is None:
            x = self.draw_x(self.n if n is None else n, rng=rng)
        else:
            x = np.asarray(x, dtype=float)
            if n is not None or x.ndim != 1 or not np.isfinite(x).all():
                raise ValueError("Supply either n or a finite one-dimensional x array.")
            if np.any((x < self.x_range[0]) | (x > self.x_range[1])):
                raise ValueError("All supplied predictors must lie in x_range.")
        z = self.distribution.rvs(size=len(x), random_state=rng) - self.distribution.ppf(self.tau)
        m, sigma = self.location(x), self.scale(x)
        data = pd.DataFrame(dict(X=x, Y=expit(logit(m) + sigma * z), m_true=m,
                                 m_null=self.null_curve(x), sigma=sigma, Z=z,
                                 near_breakpoint=self.near_breakpoints(x)))
        data.attrs = {name: getattr(self, name) for name, setting in self.__dataclass_fields__.items() if setting.init}
        data.attrs.update(n=len(x), fixed_x=fixed_x)
        if plot:
            import matplotlib.pyplot as plt
            self.plot(data)
            plt.show()
        return data

    def plot(self, data, *, ax=None, quantiles=(), title=None):
        """Plot an existing dataset without resampling; return a Matplotlib Axes.

        Shows the true target quantile, dashed null line, sampling windows and
        predictor rug. quantiles=(0.1, 0.9) adds exact conditional quantile curves.
        Save with ax.figure.savefig(path, dpi=300, bbox_inches='tight').
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
        for j, (a, b) in enumerate(self.windows()):
            ax.axvspan(a, b, color="#C13D43", alpha=0.10, lw=0, label="Breakpoint windows" if j == 0 else None)
        for k in self.breakpoints:
            ax.axvline(k, color="#C13D43", ls=":", lw=1.2)
        near = self.near_breakpoints(data.X.to_numpy())
        for mask, color, label in ((~near, "#157A8C", "Outside windows"), (near, "#C13D43", "Inside windows")):
            ax.scatter(data.X[mask], data.Y[mask], s=28, alpha=0.7, color=color, label=f"{label} ({mask.sum()} points)")
            ax.plot(data.X[mask], np.full(mask.sum(), 0.015), "|", ms=9, color=color, alpha=0.6)
        grid = np.unique(np.r_[np.linspace(*self.x_range, 600), self.breakpoints])
        ax.plot(grid, self.null_curve(grid), "--", color="#657580", lw=1.8, label=r"Null line $m_0(x)$")
        ax.plot(grid, self.location(grid), color="#243746", lw=2.4, label=rf"True $\tau^*={self.tau:g}$ quantile $m(x)$")
        for q in quantiles:
            ax.plot(grid, self.true_quantile(grid, q), "--", lw=1, label=f"True {q:g} quantile")
        design = "Base Beta design" if self.near_share is None else f"p = {self.near_share:g}"
        ax.set(xlabel="X", ylabel="Y", ylim=(0, 1), title=title or f"{design} | w = {self.half_width}, n = {len(data)}")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper right")
        return ax
