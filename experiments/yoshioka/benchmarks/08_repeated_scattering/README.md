# Repeated scattering and auxiliary controls

Benchmarks 3B and 8.

[All benchmark groups](../README.md)

| Script | Purpose |
|---|---|
| [auxiliary_wave_controls.py](auxiliary_wave_controls.py) | Controlled auxiliary-wave, channel-phase and classical/Kubo comparisons. |
| [channel_reference.py](channel_reference.py) | Independent dense reference and distorted-wave first-order integral for B8. |
| [repeated_scattering.py](repeated_scattering.py) | Benchmarks 3B and 8: solvable propagation and two-mode spatial scattering. |
| [repeated_scattering_spectra.py](repeated_scattering_spectra.py) | Finish B8 matrix-element, absolute-spectrum and first-order comparisons. |

## Configurations

- [repeated_20261102_v1.json](configs/repeated_20261102_v1.json)

Use the existing `python -m benchmarks.<module>` commands from the project root. The grouped file path is for browsing and editing.

[Corresponding results](../../benchmark_results/08_repeated_scattering/README.md)
