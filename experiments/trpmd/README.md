# Portable TRPMD template

Canonical recipe: **silicon-110-snap** in [Silicon/](Silicon/README.md).
Includes the user's four-atom Si(110) cell, SNAP coefficients/parameters, i-PI
XML, LAMMPS input, working launch/monitor/organization scripts and smoke checker.
This is the accepted 2026-10-26 layout; small inputs/models retain their hashes.

To recreate on another machine, from the `research-script` checkout:

```bash
destination=/absolute/path/to/new/TRPMD
mkdir "$destination"
cp -a experiments/trpmd/. "$destination/"
cd "$destination"
python3.11 -m venv .venv
.venv/bin/python -m pip install -r Silicon/environment/requirements-runtime.lock
```

The destination must be new; create its parent beforehand if needed. Nothing
from the old `.venv` or run results is required. These pins were tested on
Arrhenius x86-64 Linux; inspect compatibility before using other architectures.

For **SNAP silicon 110**, use [Silicon/README.md](Silicon/README.md). Defaults
are a 2×2×2 cell (32 atoms), 300 K, 100 steps at 0.5 fs; the Slurm smoke test
checks one and five beads. Production settings must be chosen separately.

Materials use `inputs/`, `models/`, `scripts/`, `environment/`, `tests/`,
`validation/` and `results/`. Completed runs use `inputs/`, `models/`, `outputs/`,
`logs/`, `restart/`, `provenance/`, `scripts/` and a short README. Provenance keeps
original scripts/settings/input snapshots and the file-location/hash map.

`Silicon/template.json` identifies the recipe and every bundled source file.
`Silicon/validation/RESULTS.md` summarizes actual Arrhenius checks; it does not
establish validation on another machine. Append new evidence at its actual site.

Verify the copied template without starting engines:

```bash
.venv/bin/python -m unittest discover -s Silicon/tests -p 'test_*.py' -v
```
