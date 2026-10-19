# Data Audit

A standalone Python library and local Streamlit app for investigating tabular label quality and class overlap. Findings are evidence relative to the chosen features, representation, method and scale—not probabilities that a label is wrong.

## Start the app

Install Python 3.11–3.13, then run `python run_ui.py` (or `./run_ui.sh` / `run_ui.cmd`). The launcher creates a dedicated `.audit-runtime` environment, installs tested UI dependencies, binds only to localhost, waits for readiness, and opens a browser. It works from other working directories and paths containing spaces or Unicode. Stop with Ctrl+C.

```sh
python run_ui.py --prepare-only
python run_ui.py --offline --no-browser --port 8501
```

After preparation, offline startup performs no installation or network calls outside localhost. For first-time offline preparation set `DATA_AUDIT_WHEELHOUSE` to a directory containing the verified dependency wheels plus `setuptools` and `wheel`. `DATA_AUDIT_RUNTIME` can relocate the product environment. Use an exclusive runtime directory for each checkout. No administrator privileges are needed.

The UI offers a marked synthetic demo or CSV/Excel upload, column mapping, configurable representations and methods, background runs with cooperative cancellation, linked graph/table selection, evidence inspection, policy thresholds, review notes, explicit CSV export scopes, and complete `.audit.zip` runs. No database service or external credentials are required.

## Use the library

```sh
python -m pip install .           # core, including CSV/.xlsx/.xls readers
python -m pip install '.[ui]'      # optional Streamlit + Plotly
python -m pip install '.[ae]'      # optional PyTorch autoencoder
```

```python
from data_audit import (
    load_csv, audit, PCAEmbedding, NeighborAudit, OOFAudit, ReviewPolicy,
)

frame = load_csv('samples.csv', id_column='sample_id')
result = audit(
    frame[['feature_0', 'feature_1', 'feature_2']], frame['label'],
    embedding=PCAEmbedding(n_components=2, standardize=True, whiten=True),
    methods=[NeighborAudit(), OOFAudit(n_splits=5, n_repeats=3)],
)
decision = result.apply_policy('neighbors', ReviewPolicy())
result.export_csv('findings.csv', include_decisions=True)
result.save('run.audit.zip')
```

`python examples/basic_audit.py --output /tmp/my-audit` runs a complete synthetic example. To view a result calculated in code, call `from data_audit.ui import show; show(result)` after installing the UI extra. The installed command `data-audit ui` also accepts `--offline`, `--no-browser`, and `--port`.

## Data contracts

- Explicit numeric features only. Inputs must be finite, labels present, feature names unique, and sample IDs unique/non-missing. pandas indices must agree exactly, including order; no hidden alignment or joins occur. NumPy input accepts `sample_ids`/`feature_names` and otherwise receives deterministic names.
- `load_csv`, `load_excel`, `list_excel_sheets`, and `map_columns` are public library functions. Readers accept local paths or seekable binary files. Source imports are CSV, `.xlsx`, and `.xls` only. Multi-sheet workbooks require explicit selection. Pass `dtype` overrides for selected columns. Textual IDs, including leading zeros, are retained; Excel IDs must be stored as text when zeros matter. Formula cells use cached values without evaluation.
- The audit owns a snapshot. Groups isolate OOF train/test partitions; excluding same-group neighbors is a separate setting.
- Neighbor defaults require enough samples for `k=30` automatic radius calibration. For a small dataset choose a smaller `n_neighbors` or explicitly choose `radius_mode='none'`. Available neighbors are reported without lowering support requirements. Inverse-distance weights use `distance_epsilon` and effective-neighbor support.
- OOF standardization and embedding fitting happen inside each training fold. Each planned training fold is checked for category counts; missing categories fail explicitly. Model classes are aligned by identity, not column position. Fold/repeat IDs, seeds, and exact split IDs are saved. Models must implement `predict_proba`; adapters can expose that interface explicitly.
- `label_issue_score = max(0, q[alternative] - q[original])`; pair overlap is `2 * min(q[a], q[b])`. All original-label pairs and the leading pair are recorded, including zeros. Ties use first-seen category order. Scores from different methods are not automatically comparable.
- Support is method-specific. A missing policy flag means abstention, distinct from `False`. A single repeat cannot estimate between-repeat dispersion. Threshold changes only recompute flags; representation, radius, or fold changes require a new audit.

## Extend with ordinary Python

```python
from data_audit import bind_embedding
import numpy as np

def fit(X, epsilon=1e-8):
    mean, scale = X.mean(0), np.maximum(X.std(0), epsilon)
    return lambda new: (new - mean) / scale

embedding = bind_embedding(fit, epsilon=1e-6)
```

See `examples/extensions`. For UI discovery put files under `embeddings/`, `distances/`, `methods/`, or `policies/` and define exactly `fit`, `distance`, `analyze`, or `decide`. Explicitly load the trusted local directory in the sidebar (or use `DATA_AUDIT_EXTENSIONS`). Discovery executes Python; uploaded source data is never treated as code. A refresh re-reads source and records a new hash for the next run.

The library derives fields and validates arguments from the same signature. Supported editable types: bool, int, float, str, scalar `Literal`, and optional versions. Annotate required parameters and `None` defaults. Unsupported/ambiguous fields are diagnosed. Arbitrary estimator/callable objects remain valid in the code API; UI objects must be supplied or registered components. Methods return a mapping with a `samples` DataFrame and may include evidence, class evidence, diagnostics, overlaps and representations. Additional sample fields get a method namespace; unavailable standard scores stay missing.

## Persistence and export

`load_run(path)` opens results without fitting or importing extension code. Bundles contain a versioned manifest and typed JSON tables, never pickle or executable objects. IDs, labels, nullable flags, categories, index and column order are retained. Unknown schemas and oversized bundles are rejected. Omit the source snapshot with `result.save(path, include_data=False)` when appropriate; original-feature details and new projections then become unavailable. Re-executing a custom component requires its code and dependencies. Restored bundles are viewable results, not executable recipes.

`export_csv` supports explicit sample/method scope and optional flags, features and metadata. `export_tables` produces a ZIP of CSV tables. CSV uses UTF-8 BOM by default and cannot preserve every pandas type; bundles are the full-fidelity format. Review notes are a separate result layer and do not mutate source labels.

## Develop and validate

```sh
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -c requirements-ui.lock -e '.[ui,dev]'
python -m pytest -q
python -m ruff check src tests examples run_ui.py
python -m build
```

Use a feature branch, focused commits, and a pull request into `main`. CI runs on Linux and Windows with Python 3.11–3.13 and has a separate optional autoencoder job. `requirements-ui.lock` constrains the tested UI environment; core installation keeps compatible dependency ranges. See [validation evidence](docs/VALIDATION.md) and the [product requirements](docs/PRD.md).

Runs report queued/running/completed/partial/failed/cancelled states and progress events. `continue_on_error=True` keeps explicitly failed rows and technical tracebacks. `CancellationToken` stops between batches, folds and methods; an individual external fit completes before the next boundary. Numerical reproducibility assumes the same environment and seeds (CPU built-ins tolerance `1e-8`); GPU/external components may be less deterministic.

For the browser suite install `.[ui,dev,browser]` and Chromium (`python -m playwright install chromium`, or set `DATA_AUDIT_CHROMIUM` to an existing executable). For the CPU autoencoder validation used here: `python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`. Full offline bootstrap integration is enabled by setting `DATA_AUDIT_TEST_WHEELHOUSE` to a populated wheelhouse. Optional integration checks explicitly skip when their prerequisites are absent.
