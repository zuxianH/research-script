# Affordable approximate silicon EELS

The new workflow computes bulk-q harmonic Born and elastic-distorted one-source spectra at 300 K and exploratory 1000 K. It uses a 1.536 nm Si [110] slab. The primary numerical representation is 49×49 unit-cell pixels and 385 longitudinal steps, with 2 meV display FWHM.

- [Measured report](benchmark_results/approximate_eels_20261102_v1/report.md)
- [300K spectrum](benchmark_results/approximate_eels_20261102_v1/Si_300K_EELS.png)
- [Conditional TACAW overlays](benchmark_results/approximate_eels_20261102_v1/TACAW_broad_detector_comparison.png)
- [Methods and limits](benchmark_results/approximate_eels_20261102_v1/methods.md)
- [Exact reproduction/resume commands](benchmark_results/approximate_eels_20261102_v1/reproduce_and_resume.sh)

Dense Born quadrature plus an independently sampled distorted-minus-Born correction preserves the actual scattering weights while reducing variance. Individual Bloch sectors and channel energies remain explicit. Full thermal occupations are handled analytically for the linear harmonic source.

The separate two-interaction solver uses ordered oscillator operators and Wick thermal contractions. Its reduced optical-triplet silicon demonstration is not a full bulk-q repeated-interaction result, and it omits direct nonlinear multiphonon vertices.

The TACAW energy axis and physical acceptance remain conditional. Main features can agree while the shapes and unmatched features differ. Temperature-fitted candidates retain their original independent validation failures. Historical strict benchmark statuses and results are preserved.
