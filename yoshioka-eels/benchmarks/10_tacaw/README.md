# TACAW reference comparisons

Benchmark 10.

[All benchmark groups](../README.md)

| Script | Purpose |
|---|---|
| [repair_metadata.py](repair_metadata.py) | Scoped TACAW metadata discovery and independently enabled comparisons. |
| [tacaw_comparison.py](tacaw_comparison.py) | Benchmark 10: immutable TACAW audit and metadata-gated comparison tools. |
| [tacaw_material_comparison.py](tacaw_material_comparison.py) | Conditional qualitative TACAW/material comparison, without fitted axes/scales. |
| [tacaw_qualitative.py](tacaw_qualitative.py) | User-authorized qualitative TACAW audit; calibration assumptions stay explicit. |

## Configurations

- [tacaw_20261102_v1.json](configs/tacaw_20261102_v1.json)
- [tacaw_user_metadata_20261102.json](configs/tacaw_user_metadata_20261102.json)

Use the existing `python -m benchmarks.<module>` commands from the project root. The grouped file path is for browsing and editing.

[Corresponding results](../../benchmark_results/10_tacaw/README.md)
