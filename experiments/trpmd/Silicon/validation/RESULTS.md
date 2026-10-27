# Recorded Si(110) SNAP validation — 2026-10-26

Arrhenius CPU, Python 3.11.17, i-PI 3.3.1, LAMMPS 2025.7.22.4.0, MPICH 5.0.2,
NumPy 2.4.6. Potential/structure identities are in `../template.json`.

| Job | Check | Outcome |
| --- | --- | --- |
| 4467264 | 32 atoms; one/five beads and clients; 100×0.5 fs | COMPLETED 0:0, 29 s |
| 4467889 | Same cases after material-folder reorganization | COMPLETED 0:0, 28 s |
| 4468246 | Same cases with organized output; five-bead restart 100→110 | COMPLETED 0:0, 24 s |

Verified target checkpoints, finite properties/positions/momenta, all expected
client handshakes, 11 evolving XYZ frames per bead and unchanged parent files
after continuation. Small tests used six CPUs, 2 GiB, five-minute job limits.

Local original evidence remains under `TRPMD/Silicon/results/` and
`TRPMD/Silicon/validation/` on the original Arrhenius checkout. Shared knowledge
stores a sanitized snapshot of checker results and file hashes, not raw outputs.

Lessons: install MPICH if the wheel cannot find `libmpi.so.12`; use the same
runtime for client/server; wait for the socket, not a fixed startup delay.
LAMMPS's `Got EXIT message from i-PI` ERROR can indicate normal shutdown only
when server exit, target checkpoint and finite output independently pass.
No-error client logs are valid too. Other errors still need investigation.

The bundled portable export changes only runtime defaults and the default cell/
step count to the documented small case. It still needs target-site validation.
These observations do not establish equilibration, bead/timestep convergence,
exact stochastic restart equivalence, long-run stability, GPU or LUMI support.
