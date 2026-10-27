# Reusable TRPMD setup

For "set up TRPMD on another machine" or "run TRPMD with SNAP silicon 110",
read `README.md`, `Silicon/README.md` and `Silicon/template.json`. This is the
canonical saved template; "Si(110)" and "silicon 110" select `Silicon/`.

Copy into a new destination before adapting it. Keep material folders separate
from run outputs, and keep the organized run layout described in the README.
Use Python 3.11+ and the pinned runtime or inspect the target machine's compatible
engines. Configure site resources from the actual target, not saved job records.
Start with the documented small test when authorized; setup alone does not
authorize a submission. Preserve native physics inputs and model hashes.

Read `research-knowledge/policies/trpmd-project-template.md` and the project card
`research-knowledge/projects/trpmd-si110-snap.md` from the configured knowledge
checkout when available. Keep new lessons scoped to site/runtime and evidence;
never copy the original machine's runtime, trajectories, checkpoints or raw logs.
