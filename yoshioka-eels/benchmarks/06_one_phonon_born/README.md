# One phonon Born scattering

Benchmark 6.

[All benchmark groups](../README.md)

| Script | Purpose |
|---|---|
| [born_outputs.py](born_outputs.py) | Absolute spectra, momentum maps, convergence plots, and Benchmark 6 report. |
| [born_quadrature.py](born_quadrature.py) | Accelerated full-space Born BZ cubature, independently checked against sources. |
| [born_reference.py](born_reference.py) | Independent finite-window Born quadrature from analytical Lobato plane kernels. |
| [one_phonon_born.py](one_phonon_born.py) | Benchmark 6: Si one-phonon transition potentials and longitudinal Born sources. |

## Configurations

- [born.json](configs/born.json)

Use the existing `python -m benchmarks.<module>` commands from the project root. The grouped file path is for browsing and editing.

[Corresponding results](../../benchmark_results/06_one_phonon_born/README.md)
