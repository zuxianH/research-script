# Yoshioka EELS development

This folder is the development source for distribution `yoshioka-eels` and
Python namespace `yoshioka`. The original source, environment, physical inputs,
checkpoints and completed outputs remain under
`/home/zuxian/Documents/PhD/Yoshioka`. Do not edit those frozen sources while
developing the package. Keep new numerical outputs under `PhD`, with explicit
fresh run directories and dependency manifests.

Read `README.md` and `docs/research-status.md`. Packaging tests do not establish
physical convergence. Preserve the original model identifiers, numerical
criteria, failures, uncertainty, contact events, amplitude interference,
weights and resource guards. The complete-pair bulk benchmark remains
NOT_ACCEPTED and has no corrected complete-net production spectrum.

Installed solver code must not import `benchmarks`. Those modules are research
checkout tools. Preserve relative compatibility links in `benchmarks` and the
existing `import yoshioka` API. Local data links are ignored by Git; create them
with `scripts/link_research_data.py`. Historical source-hashed resume commands
use the original checkout, not modified package sources.

Run relevant scientific regressions after solver changes. The full integration
suite needs the saved research data and material-reference dependencies; the
original `.venv/bin/python` may be used as an interpreter without installing
anything into that environment. Test wheel installation from outside this
checkout after changes to package boundaries. Record evidence under `PhD` and
checkpoint the managed task through the repository's pinned workflow.
