# Validation evidence

Validation was performed on Linux with Python 3.11.17, 3.12.14, and 3.13.16. The repository started with only a README. Tests exercise real computations and file formats, not placeholder responses. The attached PRD is retained as `docs/PRD.md`.

## Final results

| Environment | Result |
| --- | --- |
| Python 3.12.14, all optional integrations enabled | **51 passed, zero skipped or failed** |
| Python 3.11.17, standard development dependencies | **45 passed**, three explicit optional skips |
| Python 3.13.16, standard development dependencies | **45 passed**, three explicit optional skips |
| Python 3.12.14, clean core-only wheel | **39 passed**, no UI dependencies installed |
| Ruff lint and format checks | Passed |
| Wheel and source distribution builds | Passed |
| Core/dev and isolated product dependency checks | Passed |

The Python 3.11/3.13 skips are the browser module (Playwright absent), optional autoencoder (PyTorch absent), and full offline bootstrap (wheelhouse not supplied). All are executed successfully in the Python 3.12 run. The final full suite took 131.67 seconds. Parent-process coverage reported 69%; browser/server child processes and Streamlit's executed app script are outside that line-coverage measurement, so it is not a measure of end-to-end UI coverage.

## Executed workflows

- Core package import and a separately built wheel in a clean environment without Streamlit, Plotly, or PyTorch.
- Strict pandas/NumPy input contracts, stable IDs, explicit role mapping, immutable source snapshots, and malformed component outputs.
- Exact neighbor analysis, self/group exclusion, duplicate feature vectors, radius calibration/filtering, effective support, empty neighborhoods and small reference populations.
- Fold-local OOF preprocessing with a tracking transformer; group isolation; deliberately reversed estimator class order; rare-class errors; repeat coverage and dispersion. Identical partitions do not supply extra stability evidence.
- The PRD's four numerical score examples, deterministic ties, all required overlap pairs, abstention, unsupported fields and policy-only threshold changes.
- External extension discovery, signature-driven forms, source-hash refresh, custom embeddings/methods/policies/distances and diagnostic errors.
- CSV, `.xlsx`, `.xls`, BOMs, literal textual IDs (including `001` and `NA`), multiple sheets, duplicate headers, bad numeric cells and missing cached formula values.
- Full typed bundle round trips with categorical, string, integer and boolean labels, nullable metadata, timestamps, unnamed Series and column/index names. No source-code or pickle loading. Reopening without source data or extension code remains possible.
- Streamlit AppTest exercises configuration, external component fields, PCA variance selection, worker completion, policy changes, search, review notes and projection. `show(result)` is verified to restore the same run without fitting again.
- Real Chromium tests block external requests and exercise the complete demo flow, box selection, stable-ID propagation, selected/filtered CSV row counts, a complete saved-run download and restoration, and separate CSV/`.xlsx`/`.xls` uploads with leading-zero IDs.
- Installed CLI readiness, functional HTTP response, port collision diagnostics and child cleanup. Full source bootstrap uses a fresh runtime under a Hebrew path with spaces, a local wheelhouse and a network guard that rejects external connections. It then restarts offline from a different working directory through the POSIX wrapper.
- Wheel and source distribution builds, the executable example, dependency consistency and Ruff checks.

The UI dependencies include a tested Starlette pin. Browser testing found that the newest Starlette release's gzip response changes were incompatible with the pinned Streamlit middleware and interrupted ZIP downloads. The pin fixes that failure without modifying dependency source or bypassing verification.

## Acceptance trace

| PRD criterion | Coverage |
| --- | --- |
| AC01 | `test_core_import_no_ui`, custom-method tests, clean core-only wheel installation and export |
| AC02–AC04 | `test_external_extensions_schema_refresh_and_output`, `test_pca_variance_and_extension_forms`, generic bindings/forms |
| AC05 | Extension range/type errors and matching UI component calls; `test_signature_diagnostics` |
| AC06 | CSV/Excel reader tests and real browser upload/export tests preserve textual IDs |
| AC07 | `test_input_validation_and_snapshot` rejects mismatch, duplicate IDs, missing/nonfinite data |
| AC08 | Neighbor self/duplicate/group/radius and effective-support boundary tests |
| AC09–AC10 | OOF tracking-transformer and group-disjoint split assertions |
| AC11 | Four parameterized numerical examples in `test_score_definitions` |
| AC12 | Missing/insufficient support and policy abstention tests |
| AC13 | Single-repeat and identical-partition dispersion tests |
| AC14–AC15 | Unchanged score tables/run IDs after policy changes; source/recipe snapshots and UI rerun warning |
| AC16 | Real browser box selection, persistent sample IDs, selected-row marks and scoped export; UI selection callbacks |
| AC17 | Display-only projection scope, separate persistence, and no OOF projection reuse |
| AC18 | Real downloaded selected/filtered CSV row counts and core export-scope assertions |
| AC19–AC20 | Typed bundle round trips and opening saved results after deleting extension code |
| AC21–AC23 | Source bootstrap, Unicode/spaces, prepared offline restart and blocked-external-network checks |
| AC24 | Failed-method rows, partial states, errors and cooperative cancellation at progress boundaries |
| AC25 | Reversed model classes produce the same evidence as normal class ordering |
| AC26 | Explicit schema errors for ambiguous/open/unsupported signatures |
| AC27–AC29 | Clean core readers, CSV/Excel parity, real UI uploads, multi-sheet and cell-error tests |
| AC30 | Public readers, UI inputs and package dependencies contain CSV/Excel source routes only; no SQL connectors |

The PRD's timed human extension-writing trial and independently reviewed real-world label-quality study require participants and datasets. Automated external-extension tests validate the technical integration path; they do not substitute for those studies.

## Performance

`python benchmarks/benchmark.py` uses seed 42, 10,000 samples, 50 features and five categories. This cloud machine exposed five logical CPUs and approximately 33 GiB host memory; it is not the PRD's proposed eight-core/16 GB reference machine. Native thread pools were capped at eight.

| Operation | Recorded duration |
| --- | ---: |
| Neighbors, default k/radius/support | 8.44 s |
| OOF, 5 folds × 3 repeats | 18.40 s |
| Filter a 100,000-row table | 0.014 s |
| Construct a 20,000-point Plotly figure | 0.204 s |
| Serialize that figure | 0.006 s |
| Prepared source UI startup, offline, Unicode path | 5.02 s |

The graph timings measure server-side construction/serialization, not browser rendering or interaction latency. No claim is made that the browser's 20,000-point selection target has been benchmarked. Full analysis/export scope is retained when the view samples 20,000 points. Downloads are generated on demand to avoid re-serializing entire run bundles during filtering.

## Platform limits

Windows and Python-version CI jobs are supplied, along with a separate Chromium job and a CPU autoencoder job. This machine can execute Linux tests only; Windows process cleanup and the `.cmd` wrapper await CI on an actual Windows runner. GPU determinism is not asserted. Performance is dependent on the selected model, features, representation and available resources.

Git operations use the existing authenticated HTTPS proxy. The GitHub REST endpoint was denied by the running egress policy. `api.github.com` was saved as a draft network addition; saving a draft does not activate it or establish API credentials. Pull-request creation and hosted CI status must be reported separately from local validation.
