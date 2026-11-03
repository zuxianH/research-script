# Research scope retained during packaging

The authoritative latest continuation is in the original result store:

```text
/home/zuxian/Documents/PhD/Yoshioka/benchmark_results/14_two_phonon_eels/
  pair_convergence_continuation_20261102_v1/
```

Its `checkpoint.json`, `benchmark.json`, `continuation_decision.json`,
`dependency_invalidation.json`, `next_actions.json`, and `final_manifest.json`
retain the numerical evidence and guarded continuation commands. Packaging
does not supersede these records or rerun their production cases.

- Complete-pair four-repeat bulk benchmark: **NOT_ACCEPTED**; no corrected
  complete-net production spectrum. Selected reevaluations are diagnostics.
- The representative-panel support comparisons remain statistically unresolved
  for the complete vector. Local failures and weak-detector depth errors remain.
- Preserve coherent direct/contact plus both sequential orderings, projected
  Hamiltonian conventions, ordered-pair weights, physical axes, explicit plane
  contacts and all 56 detector/window observables.
- Criterion SHA-256:
  `d4849e737ffa48a899fbe4e239bb95d1818e110eb10204d8cbe3148e5d974305`.
- The tested historical control variate worsened variance–cost by about 3.09
  and improved only 13/56 window products; it is not a production default.
- The finite-model one-window agreement and its empirical envelope remain
  separate. The −0.021646% difference is unresolved, not a measured higher-order
  correction. Low-resolution thermal bookkeeping does not validate the
  high-resolution finite-temperature model. Nonzero q_z support and active u_z
  support remain distinct.
- B7, the original 19.962 nm specimen, full nonlinear bath convergence, thermal
  stability, intrinsic linewidths and calibrated TACAW/FRFPMS comparisons remain
  separate open milestones. Exact metadata for the original TACAW arrays is
  unavailable locally.
- Alternative dispersion and Born-tail diagnostics retain their own model
  identifiers and cannot promote the original benchmark to PASS.

The package migration records and test logs are stored separately at
`/home/zuxian/Documents/PhD/Yoshioka/package_migration_20261103_v1`.
The original source, scientific environment and completed result directories
are preserved. A software regression pass establishes packaging compatibility,
not material accuracy, speed advantage or physical convergence.

## Separate SNAP 10 K finite-bath pilot

A new finite model, `SNAP_Si110_T10_finite1x1x4_GammaXY_nonlinear_PHP_v1`, uses
fresh supplied SNAP forces and a 16-atom periodic nuclear reference with two
active transverse Gamma optical coordinates. The p=2 cutoff retains six target
states and repeated scattering, with full nonlinear displacement phases.

The local [report](../benchmark_results/15_snap_two_mode/si110_snap_T10_p2_v1/REPORT.html)
records actual spectra, channel probabilities, independent p/grid/depth checks,
failures, resource measurements and a conditional raw SNAP/TRPMD comparison.
The pilot remains a separate finite optical approximation. Neither that result
nor software tests change the NOT_ACCEPTED bulk decision, corrected complete-net
production count zero, rejected controls or the open milestones above.
