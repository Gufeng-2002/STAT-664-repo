# ART experiments for quantile breakpoints

Open [ART_Quantile_Breakpoints.ipynb](ART_Quantile_Breakpoints.ipynb), select **Course Python**, and run all cells. It imports `QuantileDataGenerator` from the adjacent `synthetic_data` package and keeps the ART experiment code in the notebook.

The experiment compares signed residuals and below-line indicators from one fitted straight-line quantile regression. It shows observed and permuted ranks, global rank CUSUM, local reranking over prespecified windows, and permutation distributions for global and joint local-maximum statistics. A small simulation compares null p-value calibration and alternative rejection rates, with known-null-line scores as diagnostic benchmarks.

Change `N`, `TAU`, `B`, `REPLICATIONS`, and the generator settings in the first code cell. The default uses 300 observations, 1,999 rank permutations, and 300 generated datasets under each hypothesis. Each permutation reuses the scores and applies the full set of scans; it does not refit quantile regression. Binary ties are broken randomly once per dataset and retained across local windows.

This is a trial adaptation of ART to an X-ordered conditional-quantile problem. Neither raw residuals nor fitted indicators automatically satisfy exchangeability under a straight conditional-quantile null. Read the null calibration results before interpreting alternative rejection as power. Local CUSUM argmax locations are exploratory splits, not guaranteed estimates of slope breakpoints.

Sources: [shared discussion](https://chatgpt.com/s/t_6aed779cf77c8191b73d79e52e5de90f), [Cui et al., ART](https://doi.org/10.1093/jrsssb/qkag037), and the supplied article at `../../articles/qkag037.pdf`.
