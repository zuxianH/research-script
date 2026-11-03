# EELS integration and convergence

Benchmark 7.

[All benchmark groups](../README.md)

| Script | Purpose |
|---|---|
| [bulk_eels.py](bulk_eels.py) | Fixed-specimen bulk BZ quadrature with explicit Bloch channel boundaries. |
| [convergence_outputs.py](convergence_outputs.py) | Actual convergence/error/cost reporting; incomplete categories stay explicit. |
| [convergence_provenance.py](convergence_provenance.py) | Keep numerical cache identities stable across reporting-only corrections. |
| [detector_cubature.py](detector_cubature.py) | Benchmark 7 finite-volume detector integration with unchanged physical BZ. |
| [eels_observables.py](eels_observables.py) | Predeclared absolute one-phonon observables shared by Benchmarks 6 and 7. |
| [repair_b7.py](repair_b7.py) | Extend, without mutating, the accepted B7 angular-cell calculation. |
| [repair_b7_joint.py](repair_b7_joint.py) | Independent full-q numerical-source joint gate, streamed and resumable. |
| [repair_b7_joint_finalize.py](repair_b7_joint_finalize.py) | Evaluate the complete independent joint gate; never accept a partial prefix. |
| [repair_b7_offset.py](repair_b7_offset.py) | Same bulk measure, shifted cubature nodes; independent offset diagnostic. |
| [repair_convergence.py](repair_convergence.py) | Versioned B7 measure audit and bounded accelerated phonon refinements. |
| [si_eels_convergence.py](si_eels_convergence.py) | Benchmark 7: resumable, fixed-model numerical convergence of phonon EELS. |

## Configurations

- [convergence.json](configs/convergence.json)
- [detector_cells_20261102_v1.json](configs/detector_cells_20261102_v1.json)
- [repair_20261102_resume_v1.json](configs/repair_20261102_resume_v1.json)
- [repair_20261102_v1.json](configs/repair_20261102_v1.json)
- [repair_b7_20261102_v2.json](configs/repair_b7_20261102_v2.json)
- [repair_b7_joint_20261102_v2.json](configs/repair_b7_joint_20261102_v2.json)

Use the existing `python -m benchmarks.<module>` commands from the project root. The grouped file path is for browsing and editing.

[Corresponding results](../../benchmark_results/07_eels_convergence/README.md)
