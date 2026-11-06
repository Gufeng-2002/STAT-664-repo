"""Run from the repository root: python -m unittest discover -s tests -v."""

import unittest

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from synthetic_data import QuantileDataGenerator


class GeneratorTests(unittest.TestCase):
    def test_original_notebook_base_design_is_reproduced(self):
        rng = np.random.default_rng(2026)
        x = rng.beta(1, 1, size=150)
        m = 0.65 - 0.2*x - 0.9*np.maximum(x-0.4, 0) + 0.9*np.maximum(x-0.7, 0)
        z = rng.standard_t(8, size=150) - stats.t.ppf(0.8, 8)
        y = expit(logit(m) + (0.4 + 0.2*x)*z)
        data = QuantileDataGenerator(seed=2026).generate()
        np.testing.assert_array_equal(data.X, x)
        np.testing.assert_allclose(data.Y, y, atol=1e-15)

    def test_zero_one_and_many_breakpoints(self):
        for knots, changes in [((), ()), ((0.5,), (-0.3,)),
                               ((0.2, 0.4, 0.6, 0.8), (-0.3, 0.2, -0.2, 0.3))]:
            model = QuantileDataGenerator(breakpoints=knots, slope_changes=changes)
            x = np.linspace(0, 1, 50)
            expected = 0.65 - 0.2*x
            for k, d in zip(knots, changes):
                expected += d*np.maximum(x-k, 0)
            np.testing.assert_allclose(model.location(x), expected)
            np.testing.assert_allclose(model.true_quantile(x), expected)
            self.assertEqual(len(model.generate(20)), 20)

    def test_true_quantile_for_both_noise_shapes(self):
        x = np.linspace(0, 1, 100)
        for noise in ("t", "normal"):
            model = QuantileDataGenerator(noise=noise)
            q = model.true_quantile(x, 0.35)
            z = (logit(q) - logit(model.location(x))) / model.scale(x)
            np.testing.assert_allclose(model.distribution.cdf(z + model.distribution.ppf(model.tau)), 0.35)

    def test_conditional_coverage(self):
        # Conditional coverage in five disjoint predictor regions, not only overall.
        data = QuantileDataGenerator(seed=310).generate(100_000)
        for indices in np.array_split(data.sort_values("X").index.to_numpy(), 5):
            group = data.loc[indices]
            self.assertLess(abs((group.Y <= group.m_true).mean() - 0.8), 0.015)

    def test_sampling_probability_and_conditional_beta_shape(self):
        model = QuantileDataGenerator(x_beta=(2, 3), near_share=0.08, seed=17)
        data = model.generate(100_000)
        near = data.near_breakpoint
        self.assertLess(abs(near.mean()-0.08), 0.005)
        base = stats.beta(2, 3)
        intervals = model.windows()
        masses = np.array([base.cdf(b)-base.cdf(a) for a, b in intervals])
        # Reweighting preserves the relative mass and the conditional CDF in each window.
        first = data.X[near & (data.X <= intervals[0][1])]
        self.assertLess(abs(len(first)/near.sum() - masses[0]/masses.sum()), 0.02)
        a, b = intervals[0]
        transformed = (base.cdf(first)-base.cdf(a)) / (base.cdf(b)-base.cdf(a))
        self.assertLess(stats.kstest(transformed, "uniform").statistic, 0.03)

    def test_overlapping_clipped_and_individual_windows(self):
        model = QuantileDataGenerator(half_width=(0.2, 0.5), near_share=0.4, seed=19)
        self.assertEqual(len(model.windows()), 1)
        np.testing.assert_allclose(model.windows(), [(0.2, 1.0)])
        data = model.generate(20_000)
        self.assertTrue(data.X.between(0, 1).all())
        self.assertLess(abs(data.near_breakpoint.mean()-0.4), 0.02)

    def test_extreme_probabilities_and_empty_windows(self):
        for p in (0, 1):
            data = QuantileDataGenerator(near_share=p, seed=4).generate(500)
            self.assertTrue((data.near_breakpoint == bool(p)).all())
        model = QuantileDataGenerator(breakpoints=(), slope_changes=(), near_share=0)
        self.assertFalse(model.generate().near_breakpoint.any())
        full = QuantileDataGenerator(half_width=1, near_share=1)
        self.assertTrue(full.generate().near_breakpoint.all())

    def test_scaled_predictor_range(self):
        model = QuantileDataGenerator(x_range=(2, 4), breakpoints=(2.8, 3.4),
                                      slope_changes=(-0.2, 0.2), beta0=0.7, beta1=-0.05,
                                      near_share=0.2, half_width=0.1)
        np.testing.assert_allclose(model.scale([2, 4]), [0.4, 0.6])
        self.assertTrue(model.generate().X.between(2, 4).all())

    def test_seeded_stream_and_overrides(self):
        a, b = QuantileDataGenerator(seed=23), QuantileDataGenerator(seed=23)
        pd.testing.assert_frame_equal(a.generate(), b.generate())
        self.assertFalse(a.generate().equals(b.generate(seed=23)))
        expected = b.generate()
        b.generate(seed=5)  # An override must not advance the object's stream.
        pd.testing.assert_frame_equal(a.generate(), b.generate())
        self.assertFalse(expected.empty)
        pd.testing.assert_frame_equal(a.generate(seed=8), a.generate(seed=8))

    def test_fixed_predictors_and_empty_dataframe(self):
        model = QuantileDataGenerator()
        data = model.generate(x=[0, 0.4, 0.7, 1], seed=99)
        np.testing.assert_array_equal(data.X, [0, 0.4, 0.7, 1])
        self.assertTrue(data.attrs["fixed_x"])
        self.assertEqual(list(model.generate(0).columns),
                         ["X", "Y", "m_true", "m_null", "sigma", "Z", "near_breakpoint"])

    def test_invalid_settings_fail_before_sampling(self):
        invalid = [dict(tau=1), dict(sigma=(0, 1)), dict(df=0), dict(x_beta=(-1, 1)),
                   dict(breakpoints=(0.7, 0.4)), dict(slope_changes=(0.1,)),
                   dict(breakpoints=(0, 0.7)), dict(near_share=1.1), dict(noise="unknown"),
                   dict(half_width=-0.1), dict(half_width=(0.1,)), dict(n=2.5),
                   dict(beta0=np.nan), dict(near_share=0.5, half_width=0),
                   dict(near_share=0.5, half_width=1),
                   dict(breakpoints=(), slope_changes=(), near_share=0.1),
                   # Both endpoints are valid but the internal vertex is not.
                   dict(beta0=0.5, beta1=2, breakpoints=(0.5,), slope_changes=(-4,))]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                QuantileDataGenerator(**settings)
        model = QuantileDataGenerator()
        for kwargs in (dict(n=-1), dict(n=True), dict(x=[1.1]), dict(x=[[0.3]]),
                       dict(x=[np.nan]), dict(n=2, x=[0.2, 0.3])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model.generate(**kwargs)

    def test_plot_does_not_resample_or_change_the_return_type(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from unittest.mock import patch
        model = QuantileDataGenerator(near_share=0.05)
        data = model.generate(seed=45)
        before = data.copy(deep=True)
        ax = model.plot(data, quantiles=(0.1, 0.9))
        self.assertEqual(len(ax.collections), 2)
        self.assertTrue(any("Null line" in line.get_label() for line in ax.lines))
        pd.testing.assert_frame_equal(data, before)
        with patch.object(plt, "show"):
            plotted = model.generate(seed=45, plot=True)
        pd.testing.assert_frame_equal(plotted, data)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
