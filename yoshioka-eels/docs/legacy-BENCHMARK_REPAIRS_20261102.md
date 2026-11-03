# Current benchmark repair entry points

The current measured results are in
`benchmark_results/repair_20261102_v2/report.md`. Historical reports and their
source implementations remain unchanged.

```bash
cd /home/zuxian/Documents/PhD/Yoshioka
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
bash benchmark_results/repair_20261102_v2/reproduce_and_resume.sh status
bash benchmark_results/repair_20261102_v2/reproduce_and_resume.sh help
```

Use `benchmarks.repair_resume` for the audited, bounded repair calculations.
Use the script's `prepare-cold` selection for fresh outputs on this machine.
Its inputs include immutable historical evidence and explicitly linked clean
implementation controls. The cold workspace is not a portable archive.

For new classical MOD trajectories, import `checkpointed_md` from
`yoshioka.molecular_dynamics_checked`. It reinitializes forces and neighbours
after `write_restart` before continuing `run pre no`. The original
`molecular_dynamics` module is preserved as historical source: its checkpoint
handling produced stale-neighbour force errors in some long 1000 K runs.
The current thermal runner is `benchmarks.repair_thermal_checked`, followed by
`benchmarks.repair_ehm_refit` and `benchmarks.repair_thermal_final_audit`.

The 9A runner streams independent coherent final channels in exact unit-cell
electron Bloch sectors. The fine electron grid, slices and electron joint
gate pass at fixed bath; full phonon convergence remains incomplete.
The nonlinear matrix-free 9B routines pass their small-system and Benchmark 8
controls, but the material result is explicitly a reduced Γ-mode calculation.
Neither result establishes full macroscopic coupled electron–phonon acceptance.

Passing references 1–6,3B,8 are reused. The report retains the incomplete B7
independent joint check, unresolved material phonon/channel convergence,
TACAW provenance blockers, effective-harmonic failure and unresolved widths.
