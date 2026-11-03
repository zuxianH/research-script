# Cross benchmark reports and audits

Project wide evidence.

[All benchmark groups](../README.md)

| Script | Purpose |
|---|---|
| [evidence_diagnostics.py](evidence_diagnostics.py) | Postprocess immutable B7/B8/B11 data; never normalize EELS spectra to fit. |
| [final_report.py](final_report.py) | Assemble actual statuses, immutable-input audit, costs and reproduction commands. |
| [repair_analysis.py](repair_analysis.py) | Read-only aggregation of historical and newly completed repair evidence. |
| [repair_manifest.py](repair_manifest.py) | Streamed content manifest for new repair evidence, with optional verification. |
| [repair_report.py](repair_report.py) | Consolidate measured repair results without changing historical evidence. |

Use the existing `python -m benchmarks.<module>` commands from the project root. The grouped file path is for browsing and editing.

[Corresponding results](../../benchmark_results/99_reports_and_audits/README.md)
