# Elastic silicon diffraction

Benchmark 2.

[All benchmark groups](../README.md)

| Script | Purpose |
|---|---|
| [abtem_reference.py](abtem_reference.py) | ASE structure verification and the independent abTEM 1.0.11 adapter. |
| [elastic_observables.py](elastic_observables.py) | Probability-preserving diffraction and fixed-pixel elastic observables. |
| [elastic_outputs.py](elastic_outputs.py) | Measured figures and report for the elastic benchmark; no simulation code. |
| [elastic_refinement.py](elastic_refinement.py) | Versioned Benchmark 2: identical constants and targeted static refinements. |
| [elastic_si110.py](elastic_si110.py) | Benchmark 2: shared Lobato slices, SI elastic propagation versus abTEM. |
| [elastic_streaming.py](elastic_streaming.py) | Memory-bounded elastic refinements; exact slice operators streamed from disk. |

## Configurations

- [elastic_repair_20261102_v1.json](configs/elastic_repair_20261102_v1.json)
- [elastic_repair_20261102_v2.json](configs/elastic_repair_20261102_v2.json)

Use the existing `python -m benchmarks.<module>` commands from the project root. The grouped file path is for browsing and editing.

[Corresponding results](../../benchmark_results/02_elastic/README.md)
