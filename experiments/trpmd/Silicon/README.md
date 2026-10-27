# Si(110) SNAP — i-PI/LAMMPS TRPMD

Recipe ID: `silicon-110-snap`; aliases: silicon 110, Si(110), SNAP silicon.
Work from `TRPMD/Silicon/` after copying the [portable template](../README.md).

```text
inputs/        native i-PI/LAMMPS templates and four-atom Si(110) cell
models/        user-supplied SNAP coefficients and parameters
scripts/       launch, monitor, file organization and Slurm smoke test
environment/   runtime selection and pinned dependencies
tests/         output checker
validation/    recorded checks
results/       separate run folders
```

Prepare a five-bead case without running engines:

```bash
source environment/env.sh
DRY_RUN=1 NUM_BEADS=5 NUM_CLIENT=5 RUN_NAME=prepare_b5 bash scripts/run.sh
```

On an allocated compute node, run one short case:

```bash
source environment/env.sh
NUM_BEADS=5 NUM_CLIENT=5 bash scripts/run.sh
```

Defaults: 2×2×2 (32 atoms), 300 K, 100 steps at 0.5 fs (50 fs). XML uses
PILE_G, lambda=0.5, weak centroid coupling tau=1 ns and seed 31415.
Bead positions start from the same crystal; this is an execution test.

For the Arrhenius CPU partition, after configuring your account:

```bash
mkdir -p results
sbatch --account=YOUR_CPU_ACCOUNT scripts/validate.slurm
```

That script tests one and five beads/clients with six CPUs, 2 GiB and five
minutes. Adapt partition/account/runtime for another site before submission.
Check fresh smoke output with:

```bash
../.venv/bin/python tests/check_smoke.py results/YOUR_RUN
```

Completed/prepared runs are organized automatically into inputs/models/outputs/
logs/restart/provenance/scripts with a README. Run-local `scripts/continue.sh`
adds steps in a separate segment, retaining parent outputs and checkpoints.
For longer simulations set `REPLICATE`, `RUNSTEPS`, `NUM_BEADS`, `NUM_CLIENT`,
`TEMPERATURE` and `IPI_TIMESTEP_FS` explicitly, with suitable resources.
The checker is specifically for fresh 32-atom/100-step smoke runs.

See [recorded validation](validation/RESULTS.md) and `template.json` for scope
and file identities. The original potential's publication/fit provenance was
not supplied; filenames identify these particular user-provided files.
