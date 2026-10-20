# MATLAB notebook data audit

Standalone MATLAB rewrite of `notebook_lite`. Copy this folder anywhere; it does
not import Python, `data_audit`, or the GUI. Results are structs containing ordinary
MATLAB tables with explicit `sample_id` columns.

## Start

Use **MATLAB R2022b or newer** with **Statistics and Machine Learning Toolbox**
(`pdist2`, `tiedrank`). Open this folder in MATLAB, then open **`analysis.mlx`** in
the Live Editor and choose **Run All**. Change the CSV path, ID, label, optional
group, and explicit feature columns in section 1. `demo.csv` is synthetic.
No package installation or network connection is needed during analysis.

`analysis.mlx` is a Live Script with narrative sections and executable MATLAB
code. `analysis.m` is its matching text companion for code review and batch use.
After changing the companion, run `build_live_script` in MATLAB to regenerate
`analysis.mlx` (uses MATLAB's internal Live Editor conversion utility).
Save the executed Live Script to retain outputs. The committed notebook has
no cached results; it must be run on your machine.

```matlab
cd notebook_lite_matlab
checks                    % numerical and leakage checks
open analysis.mlx         % interactive notebook
% run('analysis.m')        % equivalent text-script batch run
```

## Included

- `neighbors.m`: batched exact distances, deterministic ties, self/group
  exclusion, automatic/fixed/no radius, uniform/inverse-distance weighting,
  effective-neighbor support, coordinates and individual neighbor evidence.
- `oof.m`: repeated stratified or stratified group folds, training-fold-local
  representations and models, exact split IDs, class-aligned probabilities,
  repeat coverage, dispersion and alternative agreement. No full-data final fit.
- `score_evidence.m`: original ranking formulas, first-seen class tie ordering,
  per-class evidence and original-label/leading-pair overlap evidence.
- `fit_representation.m`: identity, population-standardization, PCA by count or
  variance fraction, whitening, pipelines, optional native autoencoder, and
  training-scoped custom fitter handles. `center_only.m` is a small example.
- Exploration: thresholds, ID-aware native interactive scatter/data tips,
  histograms, sample inspection, metadata summaries, pair heatmaps, percentile
  method comparisons, separate review notes, and full/filtered/selected exports.

IDs and labels are normalized to strings. IDs are read as strings before import
so leading zeroes survive. APIs take `X` as a numeric-feature table and `y`, `ids`,
and optional `Groups` in exactly the same row order. Row alignment is explicit;
MATLAB vectors do not carry pandas indices. Missing/nonfinite features, missing
labels/groups and duplicate IDs fail explicitly. Feature names/order must remain
consistent in your calls. Metadata never enters the feature matrix implicitly.

A custom representation is `@(train) transform`, where `transform(new)` returns a
finite numeric matrix with unchanged rows and consistent width. A custom metric
is `@(a,b) nonnegativeScalar`. A custom OOF estimator is
`@(trainX,trainCodes,numClasses,seed) predictor`; `predictor(testX)` must return
normalized probabilities with columns **1:numClasses** in first-seen label order.
Any custom training, tuning, or calibration must remain inside the training fold.

## Interpretation and backend differences

`label_issue_score = max(0, q(alternative)-q(original))` and
`pair_overlap = 2*min(q(a),q(b))`. Scores are not error probabilities and thresholds
need calibration. Weak support keeps scores but causes flags to abstain.
Flags are numeric **1 / 0 / NaN** (review / below threshold / abstain).
One repeat or identical repeat partitions gives missing stability estimates.
Dispersion between repeats is not a confidence interval. No labels change.

Neighbor radius uses NumPy-compatible linear quantiles. Six golden fixtures in
`reference/` were generated from the source Python implementation on `demo.csv`
using identity features, k=20, all three radius modes and both weightings.
MATLAB checks compare scores, support and distance diagnostics within `1e-10`.

OOF uses MATLAB-native stratification/group allocation and a self-contained BFGS
multinomial logistic fit with L2 penalty C=1 and unpenalized intercept. Its random
fold assignments and optimizer differ from scikit-learn; OOF scores need not match
Python row-for-row. Group allocation is greedy, with explicit failure when a fold
cannot meet training-class support. The caller's random state is restored.

The optional autoencoder requires **Deep Learning Toolbox** and uses
`trainAutoencoder`/`encode`, CPU only. Its architecture/training differ from the
Python PyTorch implementation (including early stopping); it is not a numerical
replica. It is not used by the demo or default checks. Its normalization and fit
remain confined to the training fold.

Exports go to ignored `exports/`: CSV tables, split JSON, native interactive FIG,
and PNG. CSV requires explicit types when reloaded. Native MATLAB figures replace
Plotly/HTML; no browser lasso synchronization, GUI jobs or legacy audit bundles.
Filtering an existing result does not reanalyze the reference population.

## Validation status

**Passed on MATLAB R2025b in [GitHub Actions](https://github.com/dotanankri24d/auto_data_audit/actions/runs/39380758410):**
all numerical checks and an end-to-end execution of the actual `analysis.mlx`.
MATLAB is not installed locally; interactive Live Editor UI was not inspected.
`checks.m` covers IDs,
duplicate-feature neighbors, radius abstention, invalid inputs, group isolation,
fold-local fitting, RNG restoration, class evidence normalization, single-repeat
stability, direct formulas, and Python numerical fixtures.
`.github/workflows/matlab.yml` runs these checks and the actual Live Script with
MATLAB R2025b on GitHub Actions. Optional autoencoder is excluded.
