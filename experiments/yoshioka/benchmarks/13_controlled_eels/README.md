# Controlled model and thickness comparisons

Four model comparisons.

[All benchmark groups](../README.md)

| Script | Purpose |
|---|---|
| [controlled_eels.py](controlled_eels.py) | Bounded controlled four-model EELS calculations using the validated source. |
| [controlled_eels_analysis.py](controlled_eels_analysis.py) | Paired four-model spectra, raw weights, and separately labelled sensitivities. |
| [controlled_eels_finalize.py](controlled_eels_finalize.py) | Completion, regression, and streamed integrity audits of controlled EELS. |
| [controlled_eels_report.py](controlled_eels_report.py) | Figures and measured interpretation of the controlled EELS comparison. |
| [controlled_eels_run.py](controlled_eels_run.py) | Resume bounded controlled-comparison stages with explicit progress checks. |
| [controlled_eels_sampling.py](controlled_eels_sampling.py) | Explicit compatible nested-Sobol prefix reuse for a justified source refinement. |
| [controlled_eels_validate.py](controlled_eels_validate.py) | Independent complex source/atomic-sum controls for the actual fitted modes. |

Use the existing `python -m benchmarks.<module>` commands from the project root. The grouped file path is for browsing and editing.

[Corresponding results](../../benchmark_results/13_controlled_eels/README.md)
