# Yoshioka EELS package development

This directory is a Python package: distribution `yoshioka-eels`, import
namespace `yoshioka`. Keep reusable Python code, package metadata, portable
tests, documentation, and small examples here.

Do not add benchmark runners, saved results, result-store links, research
reports, machine-specific data-link scripts, or material input files to this
directory. Research sources and completed outputs live separately under
`/home/zuxian/Documents/PhD/Yoshioka`; preserve those frozen records. New
calculations use fresh external output directories and dependency manifests.

Read `README.md` and `docs/api.md`. Preserve the existing import API, physical
operators, SI conventions, coherent amplitudes, weights, contacts, numerical
tolerances, and resource guards. Package checks do not establish physical
convergence or supersede historical scientific decisions.

Installed code must be independent of research runners and local data. Keep
optional engines optional at import time. Tests use synthetic inputs and
independent analytical references; never require a saved research run.

Run the relevant unit tests after changes and test wheel installation outside
the checkout after package-boundary edits. Store verification logs under PhD,
and checkpoint the managed task through the repository's pinned workflow.
