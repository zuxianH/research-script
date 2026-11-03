# Leading two phonon EELS

Two insertion approximation.

[All benchmark groups](../README.md)

| Script | Purpose |
|---|---|
| [two_phonon_analysis.py](two_phonon_analysis.py) | Raw process-resolved reductions and independent-batch uncertainty. |
| [two_phonon_eels.py](two_phonon_eels.py) | Bounded/resumable continuous-BZ leading two-phonon silicon EELS. |
| [two_phonon_finalize.py](two_phonon_finalize.py) | Standalone methods, measured-cost report and immutable reproduction audit. |
| [two_phonon_report.py](two_phonon_report.py) | Measured two-phonon convergence, figures, conditional reference and report. |
| [two_phonon_validation.py](two_phonon_validation.py) | Independent two-mode Fock propagation and actual-q source controls. |

Use the existing `python -m benchmarks.<module>` commands from the project root. The grouped file path is for browsing and editing.

[Corresponding results](../../benchmark_results/14_two_phonon_eels/README.md)

## Complete leading material workflow

Use the flat module aliases from the project root: `benchmarks.complete_leading_eels` for source construction and bounded pair batches; `benchmarks.complete_leading_guard` for live input/result integrity; `benchmarks.complete_acceptance` for every gain/window/shape gate; `benchmarks.complete_pair_controls` for paired electron/slice controls; and `benchmarks.complete_leading_report` for the report and PNG figures. The `_analysis.json` gate fields are preliminary monitors; `_acceptance.json` contains the authoritative complete gate set.

See the [versioned resume entry point](../../benchmark_results/14_two_phonon_eels/complete_leading_20261102_v1/resume.sh).

## Pair-convergence continuation

The flat aliases `benchmarks.pair_electron_diagnostics`, `benchmarks.pair_background_setup`, `benchmarks.pair_sampling_audit`, `benchmarks.pair_finite_validity`, and `benchmarks.pair_convergence_reporting` preserve the normal project-root invocation. The diagnostic runner separates retained reciprocal support, product support, depth, and local action tolerance, and saves complex direct/sequential components. The reporting utility performs no propagation.

Read the [executed continuation report](../../benchmark_results/14_two_phonon_eels/pair_convergence_20261102_v1/report.md) before choosing settings. Its [resume script](../../benchmark_results/14_two_phonon_eels/pair_convergence_20261102_v1/resume.sh) verifies the sealed delivery and writes future bounded representative calculations to a separate result directory.
