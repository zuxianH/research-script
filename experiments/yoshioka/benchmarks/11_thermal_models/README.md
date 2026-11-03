# Thermal models and linewidth validation

Benchmark 11.

[All benchmark groups](../README.md)

| Script | Purpose |
|---|---|
| [anharmonic_followup.py](anharmonic_followup.py) | B11 fit-uncertainty audit and explicitly unaccepted effective-harmonic EELS. |
| [anharmonic_modes.py](anharmonic_modes.py) | Benchmark 11A/11B: independent MOD trajectories, effective harmonic modes. |
| [anharmonic_validation.py](anharmonic_validation.py) | Independent mode correlations, fit convergence and linewidth audit for B11. |
| [repair_ehm_refit.py](repair_ehm_refit.py) | Independent EHM learning curves using repaired, energy-checked MD data. |
| [repair_learning_audit.py](repair_learning_audit.py) | Learning curves against a declared modal-energy independence proxy. |
| [repair_md_checkpoint_audit.py](repair_md_checkpoint_audit.py) | Actual-MOD checkpoint pilot and fresh-force checks at known failure times. |
| [repair_thermal.py](repair_thermal.py) | Longer independent NVE references; original effective-model failure retained. |
| [repair_thermal_audit.py](repair_thermal_audit.py) | Implementation, force-parity and sampling audits independent of EELS peaks. |
| [repair_thermal_checked.py](repair_thermal_checked.py) | Longer independent NVE references; original effective-model failure retained. |
| [repair_thermal_final_audit.py](repair_thermal_final_audit.py) | Separate checkpoint corruption, clean MD reference, and EHM inadequacy. |
| [validated_mode_eels.py](validated_mode_eels.py) | Reduced EHM EELS using only independently accepted Gamma mode subspaces. |

## Configurations

- [anharmonic_20261102_v1.json](configs/anharmonic_20261102_v1.json)
- [repair_thermal_20261102_v2.json](configs/repair_thermal_20261102_v2.json)
- [repair_thermal_checked_20261102_v2.json](configs/repair_thermal_checked_20261102_v2.json)

Use the existing `python -m benchmarks.<module>` commands from the project root. The grouped file path is for browsing and editing.

[Corresponding results](../../benchmark_results/11_thermal_models/README.md)
