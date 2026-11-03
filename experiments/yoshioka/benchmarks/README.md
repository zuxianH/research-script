# Benchmark scripts

The numbered folders group related work by benchmark. The same group names are used in [scripts](../benchmarks/README.md) and [results](../benchmark_results/README.md).

| Folder | Contents |
|---|---|
| [01_free_electron](01_free_electron/README.md) | Benchmark 1: free electron propagation |
| [02_elastic](02_elastic/README.md) | Benchmark 2: elastic silicon diffraction |
| [03_oscillator_zero_temperature](03_oscillator_zero_temperature/README.md) | Benchmark 3A: zero temperature oscillator response |
| [04_oscillator_finite_temperature](04_oscillator_finite_temperature/README.md) | Benchmark 4: finite temperature oscillator response |
| [05_lattice_dynamics](05_lattice_dynamics/README.md) | Benchmark 5: silicon lattice dynamics |
| [06_one_phonon_born](06_one_phonon_born/README.md) | Benchmark 6: One phonon Born scattering |
| [07_eels_convergence](07_eels_convergence/README.md) | Benchmark 7: EELS integration and convergence |
| [08_repeated_scattering](08_repeated_scattering/README.md) | Benchmarks 3B and 8: repeated scattering and auxiliary controls |
| [09_material_eels](09_material_eels/README.md) | Benchmarks 9A and 9B: Dynamical and nonlinear material EELS |
| [10_tacaw](10_tacaw/README.md) | Benchmark 10: TACAW reference comparisons |
| [11_thermal_models](11_thermal_models/README.md) | Benchmark 11: thermal models and linewidth validation |
| [12_approximate_eels](12_approximate_eels/README.md) | Approximate spectra: Affordable approximate EELS |
| [13_controlled_eels](13_controlled_eels/README.md) | Four model comparisons: controlled model and thickness comparisons |
| [14_two_phonon_eels](14_two_phonon_eels/README.md) | Two insertion approximation: Leading two phonon EELS |
| [90_workflow](90_workflow/README.md) | Workflow infrastructure: shared workflow and resume tools |
| [99_reports_and_audits](99_reports_and_audits/README.md) | Project wide evidence: cross benchmark reports and audits |

## Running benchmarks

Run from the project root with the existing module names and configuration paths. Group folders organize the files; Python imports remain under `benchmarks`.

```bash
cd /home/zuxian/Documents/research-script/yoshioka-eels
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.repair_resume status
.venv/bin/python -m benchmarks.free_electron --help
```

Configurations are stored inside each group under `configs`. Existing `benchmarks/configs/*.json` paths continue to work. Shared workflow and resume tools are in `90_workflow`.

## Compatibility and maintenance

Old flat paths remain available through compatibility links. `.hidden` keeps these entries out of normal views in file browsers that support it; terminals and some editors may still show them. Keep the links: existing commands, imports, report references, and cache provenance use those names.

Original code, configuration, and result contents are preserved. A few files whose location is part of their behavior stay physically at their original paths and have links in the groups. The two runs containing location-sensitive preservation helpers also stay in place and are linked into their groups. Use the [repair resume entry point](../benchmark_results/99_reports_and_audits/resume_repair.sh) to invoke the historical repair script through the organized layout.

Add new work to the appropriate group and preserve the established import and command paths. The move map, verification record, and layout restore utility are in [the organization record](../.organization/20261102_folder_cleanup/).

This package copy is for new development. Original organization records, frozen
sources and completed cases remain in `/home/zuxian/Documents/PhD/Yoshioka`.
Local data links are configured by `scripts/link_research_data.py`; see the
[package README](../README.md). Original source-hashed resume commands must run
against the original checkout. Use fresh output directories and manifests for
new package calculations. Some historical comparisons require external project
paths; copying a runner does not make its external data portable.
