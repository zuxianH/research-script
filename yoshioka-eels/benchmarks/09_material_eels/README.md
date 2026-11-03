# Dynamical and nonlinear material EELS

Benchmarks 9A and 9B.

[All benchmark groups](../README.md)

| Script | Purpose |
|---|---|
| [dynamical_si_eels.py](one_source/dynamical_si_eels.py) | Benchmark 9A dynamical one-source Si EELS; 9B feasibility kept separate. |
| [material_evidence.py](one_source/material_evidence.py) | B9 absolute observables, separate refinement gates and representation costs. |
| [phonon_bath_audit.py](one_source/phonon_bath_audit.py) | Diagnose longitudinal coherent sampling using completed 9A amplitudes. |
| [repair_9a.py](one_source/repair_9a.py) | Resumable independent-channel Si refinement with one live source batch. |
| [repair_9a_batched.py](one_source/repair_9a_batched.py) | Depth-major revision of the independent-channel Si refinement. |
| [repair_9a_bloch.py](one_source/repair_9a_bloch.py) | Symmetry-preserving B9A on the same bath and periodic physical specimen. |
| [repair_9a_bloch_direct.py](one_source/repair_9a_bloch_direct.py) | Symmetry-preserving B9A on the same bath and periodic physical specimen. |
| [repair_9a_bloch_mesh.py](one_source/repair_9a_bloch_mesh.py) | Symmetry-preserving B9A on the same bath and periodic physical specimen. |
| [repair_9a_bloch_windowed.py](one_source/repair_9a_bloch_windowed.py) | Symmetry-preserving B9A on the same bath and periodic physical specimen. |
| [repair_9a_depth.py](one_source/repair_9a_depth.py) | Depth-major revision of the independent-channel Si refinement. |
| [repair_9a_fourier_depth.py](one_source/repair_9a_fourier_depth.py) | Symmetry-preserving B9A on the same bath and periodic physical specimen. |
| [repair_9a_gemm.py](one_source/repair_9a_gemm.py) | Depth-major revision of the independent-channel Si refinement. |
| [repair_bloch_analysis.py](one_source/repair_bloch_analysis.py) | Distinct convergence gates for the symmetry-preserving source calculations. |
| [repair_bloch_validation.py](one_source/repair_bloch_validation.py) | Independent bath-basis, Born-normalization and full-cell propagation checks. |
| [repair_coordinate_order_audit.py](nonlinear/repair_coordinate_order_audit.py) | Audit coordinate quadrature at the largest computed reduced Si basis. |
| [repair_fourier_memory_pilot.py](one_source/repair_fourier_memory_pilot.py) | Cold largest-background and optional bounded source-sector memory pilots. |
| [repair_material_checkpoint.py](nonlinear/repair_material_checkpoint.py) | Fine reduced-material electron representation with slice-level resume. |
| [repair_material_checkpoint_init.py](nonlinear/repair_material_checkpoint_init.py) | Initialize an immutable zero-slice checkpoint before bounded invocations. |
| [repair_material_coordinate.py](nonlinear/repair_material_coordinate.py) | Converge the same reduced Si Hamiltonian using a checked coordinate action. |
| [repair_material_orders.py](nonlinear/repair_material_orders.py) | Coherent leading repeated amplitudes, with an independent dense control. |
| [repair_material_repeated.py](nonlinear/repair_material_repeated.py) | Finite-z nonlinear repeated Si scattering in a declared reduced Gamma bath. |
| [repair_matrix_free.py](nonlinear/repair_matrix_free.py) | Identical-physics dense/structured B3B and B8 validation and cost pilot. |
| [repair_source_local_audit.py](one_source/repair_source_local_audit.py) | Independent quadrature check of the already-exact frozen-slice source step. |

## Configurations

- [repair_material_20261102_v2.json](nonlinear/configs/repair_material_20261102_v2.json)
- [repair_material_checkpoint_20261102_v2.json](nonlinear/configs/repair_material_checkpoint_20261102_v2.json)
- [repair_material_coordinate_20261102_v2.json](nonlinear/configs/repair_material_coordinate_20261102_v2.json)
- [repair_material_coordinate_cutoff_slice_quadrature_20261102_v2.json](nonlinear/configs/repair_material_coordinate_cutoff_slice_quadrature_20261102_v2.json)
- [repair_material_coordinate_electron49_20261102_v2.json](nonlinear/configs/repair_material_coordinate_electron49_20261102_v2.json)
- [repair_material_coordinate_electron_20261102_v2.json](nonlinear/configs/repair_material_coordinate_electron_20261102_v2.json)
- [repair_material_gamma6_20261102_v2.json](nonlinear/configs/repair_material_gamma6_20261102_v2.json)
- [dynamical_si_20261102_v1.json](one_source/configs/dynamical_si_20261102_v1.json)
- [dynamical_si_bath4_20261102_v1.json](one_source/configs/dynamical_si_bath4_20261102_v1.json)
- [dynamical_si_padding_20261102_v1.json](one_source/configs/dynamical_si_padding_20261102_v1.json)
- [dynamical_si_resume_20261102_v1.json](one_source/configs/dynamical_si_resume_20261102_v1.json)
- [dynamical_si_thermal_resume_20261102_v1.json](one_source/configs/dynamical_si_thermal_resume_20261102_v1.json)
- [repair_9a_20261102_v2.json](one_source/configs/repair_9a_20261102_v2.json)
- [repair_9a_batched_20261102_v2.json](one_source/configs/repair_9a_batched_20261102_v2.json)
- [repair_9a_bath8_20261102_v2.json](one_source/configs/repair_9a_bath8_20261102_v2.json)
- [repair_9a_bath_20261102_v2.json](one_source/configs/repair_9a_bath_20261102_v2.json)
- [repair_9a_bloch_20261102_v2.json](one_source/configs/repair_9a_bloch_20261102_v2.json)
- [repair_9a_bloch_bath_20261102_v2.json](one_source/configs/repair_9a_bloch_bath_20261102_v2.json)
- [repair_9a_bloch_electron_20261102_v2.json](one_source/configs/repair_9a_bloch_electron_20261102_v2.json)
- [repair_9a_bloch_extended_20261102_v2.json](one_source/configs/repair_9a_bloch_extended_20261102_v2.json)
- [repair_9a_bloch_fullmesh_20261102_v2.json](one_source/configs/repair_9a_bloch_fullmesh_20261102_v2.json)
- [repair_9a_bloch_transverse_20261102_v2.json](one_source/configs/repair_9a_bloch_transverse_20261102_v2.json)
- [repair_9a_bloch_windowed_grid_20261102_v2.json](one_source/configs/repair_9a_bloch_windowed_grid_20261102_v2.json)
- [repair_9a_bloch_windowed_joint_20261102_v2.json](one_source/configs/repair_9a_bloch_windowed_joint_20261102_v2.json)
- [repair_9a_depth_20261102_v2.json](one_source/configs/repair_9a_depth_20261102_v2.json)
- [repair_9a_fft_grid_20261102_v2.json](one_source/configs/repair_9a_fft_grid_20261102_v2.json)
- [repair_9a_fft_joint_20261102_v2.json](one_source/configs/repair_9a_fft_joint_20261102_v2.json)
- [repair_9a_fourier_depth_grid_20261102_v2.json](one_source/configs/repair_9a_fourier_depth_grid_20261102_v2.json)
- [repair_9a_fourier_depth_joint_20261102_v2.json](one_source/configs/repair_9a_fourier_depth_joint_20261102_v2.json)
- [repair_9a_gemm_20261102_v2.json](one_source/configs/repair_9a_gemm_20261102_v2.json)
- [repair_9a_thick_20261102_v2.json](one_source/configs/repair_9a_thick_20261102_v2.json)

Use the existing `python -m benchmarks.<module>` commands from the project root. The grouped file path is for browsing and editing.

[Corresponding results](../../benchmark_results/09_material_eels/README.md)
