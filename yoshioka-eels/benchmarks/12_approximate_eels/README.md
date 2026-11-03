# Affordable approximate EELS

Approximate spectra.

[All benchmark groups](../README.md)

| Script | Purpose |
|---|---|
| [approximate_eels.py](approximate_eels.py) | Resumable affordable bulk-q Si EELS, separate from strict benchmark gates. |
| [approximate_eels_analysis.py](approximate_eels_analysis.py) | Plots and declared approximate gates; no modification of strict benchmarks. |
| [approximate_eels_compare.py](approximate_eels_compare.py) | Conditional broad-detector comparison; no fitted axes or absolute matching. |
| [approximate_eels_cost.py](approximate_eels_cost.py) | Measure source phases on saved production sectors without changing results. |
| [approximate_eels_ehm.py](approximate_eels_ehm.py) | Diagnostic bulk-q spectra from independently refitted, corrected MD data. |
| [approximate_eels_finalize.py](approximate_eels_finalize.py) | Requirement-level completion audit and streamed preservation/provenance. |
| [approximate_eels_gamma_patch.py](approximate_eels_gamma_patch.py) | Integrate the thermal acoustic Gamma cell instead of omitting its interior. |
| [approximate_eels_repeated.py](approximate_eels_repeated.py) | Validated finite-T two-insertion controls and a reduced Si optical pilot. |
| [approximate_eels_report.py](approximate_eels_report.py) | Assemble measured approximate-EELS results, figures, provenance and limits. |
| [approximate_eels_resume.py](approximate_eels_resume.py) | Bounded invocations with committed-q progress checks, never blind restarts. |
| [approximate_eels_validation.py](approximate_eels_validation.py) | Independent numerical-source and approximation diagnostics for bulk-q EELS. |
| [prepare_approximate_eels_cold.py](prepare_approximate_eels_cold.py) | Prepare new-directory reproduction configurations without changing history. |

## Configurations

- [approximate_eels_20261102_v1.json](configs/approximate_eels_20261102_v1.json)
- [approximate_eels_confirmation_20261102_v1.json](configs/approximate_eels_confirmation_20261102_v1.json)
- [approximate_eels_covariance_20261102_v1.json](configs/approximate_eels_covariance_20261102_v1.json)
- [approximate_eels_ehm_20261102_v1.json](configs/approximate_eels_ehm_20261102_v1.json)
- [approximate_eels_production_20261102_v1.json](configs/approximate_eels_production_20261102_v1.json)
- [approximate_eels_refinement_20261102_v1.json](configs/approximate_eels_refinement_20261102_v1.json)

Use the existing `python -m benchmarks.<module>` commands from the project root. The grouped file path is for browsing and editing.

[Corresponding results](../../benchmark_results/12_approximate_eels/README.md)
