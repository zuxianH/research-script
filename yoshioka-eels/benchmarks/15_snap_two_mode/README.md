# SNAP Si [110] 10 K finite optical pilot

This runner defines a new finite model, not a continuation of the historical
MOD complete-pair benchmark. It uses fresh Si SNAP forces, a periodic 1×1×4
nuclear bath and two specimen-normalized real Gamma optical modes with fixed
laboratory x/y polarizations. The electron Hamiltonian retains the full nonlinear
Lobato displacement phase projected into total occupation n_a+n_b ≤ p. The
six p=2 target states do not limit the number of scattering events.

The frozen initial pilot configuration and criteria are in `config.json`.
Numerical data, figures, frozen sources and logs belong under:

`/home/zuxian/Documents/PhD/Yoshioka/benchmark_results/15_snap_two_mode/si110_snap_T10_p2_v1`

Open that directory's `REPORT.html` or `REPORT.md` for measured results,
convergence failures and the conditional raw SNAP/TRPMD comparison. Existing
completed cases are immutable. The supplied SNAP files and reference array are
read-only inputs; do not substitute MOD arrays or change fitted SNAP settings.

From the package directory, using the existing material-dependency interpreter
without installing anything into it:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
out=/home/zuxian/Documents/PhD/Yoshioka/benchmark_results/15_snap_two_mode/si110_snap_T10_p2_v1
py=/home/zuxian/Documents/PhD/Yoshioka/.venv/bin/python
flock -n "$out/.scientific-worker.lock" timeout 600s "$py" benchmarks/15_snap_two_mode/run.py summarize
```

The explicit stages are `equilibrium`, `fc`, `bath`, `checks`, `propagate`, and
`summarize`. For a *fresh* experiment first create a new output directory with
`inputs/`, `logs/`, `phonons/`, `checks/`, `cases/`, `reference/`, `figures/`, and
`provenance/`; copy the supplied small inputs and `config.json` there, preserving
hashes. Pass its path with `--output`. Do not call equilibrium/bath over a
completed preparation.

```bash
flock -n "$out/.scientific-worker.lock" timeout 600s "$py" benchmarks/15_snap_two_mode/run.py equilibrium
flock -n "$out/.scientific-worker.lock" timeout 600s "$py" benchmarks/15_snap_two_mode/run.py fc
flock -n "$out/.scientific-worker.lock" timeout 600s "$py" benchmarks/15_snap_two_mode/run.py bath
flock -n "$out/.scientific-worker.lock" timeout 600s "$py" benchmarks/15_snap_two_mode/run.py checks
flock -n "$out/.scientific-worker.lock" timeout 600s "$py" benchmarks/15_snap_two_mode/run.py propagate --p 2 --n 33 --fields 769 --steps 769 --dimension 32
```

For independent refinements change one of `--p`, `--n`, `--fields`, `--steps`, or
`--dimension`. `--case-tag` distinguishes an explicit same-parameter control.
Do not start an additional case until the previous invocation exits; the shared
file lock enforces one scientific worker. The runner checks 4 GiB RSS, a
conservative construction-time memory estimate and a 570 s checkpoint deadline,
with a 590 s alarm and external 600 s timeout. Large field arrays are tiled to
disk; mapped pages are dropped during construction and propagation. Estimate
both disk and runtime before increasing N. The cutoff is always a full N×N
reciprocal rectangle with M=2N−1 projection at each Hamiltonian action.

For an incomplete case, use the exact same command, configuration, bath and
numerical source. Resume checks the source/config/bath/field fingerprints,
initial-wave identity and committed wave checksum. At most an uncommitted
16-step block is discarded after a resource stop. Never bypass a digest mismatch.
A changed numerical implementation requires a separately named control/new case.

The reference overlay is conditional on the inspected notebook axes. A 20-pixel
ROI is not calibrated to 20 mrad. It compares different thicknesses and different
phonon baths, preserves raw array units separately, and does not establish
material accuracy, absolute intensity agreement or a runtime advantage.

The retained complete-pair bulk benchmark remains NOT_ACCEPTED with corrected
complete-net production count zero. No scheduler or remote publication is used.
