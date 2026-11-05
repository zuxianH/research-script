# All internal modes in finite Si [110]

`run.py` prepares a SNAP-force harmonic fit with hiPhive and propagates the
full three-dimensional rigid-ion Lobato displacement phases in a total phonon
occupation basis. The working single-oscillator exponential is evaluated with
the configured `working_dimension` before projection. Active longitudinal
displacements are retained. This differs from the transverse-only fast action.

The initial use is the four-atom 1×1×1 cell, 10 K, 60 keV plane wave, all nine
internal modes, p=2 (55 channels), N=97, M=193, 769 depth-field points, 11
propagation slices, D=32, and a 0–30 mrad detector. The user confirmed N=97
directly on this cell; the historical 481-grid/97-sector relation concerned a
different laterally repeated specimen. The three rigid translations are removed.

The 9.85 Å harmonic cutoff covers twice the supplied 4.9 Å SNAP neighbor cutoff.
A 4×6×6 force-training cell separates harmonic images. Eight independent paired
random displacements train the symmetry-constrained second-order hiPhive model;
two pairs and direct four-atom finite differences at 0.005/0.0025 Å validate it.
Failed fitting, force, frequency, Hermiticity, memory, or norm checks stop the run.
Static zero-pressure geometry omits thermal and zero-point volume expansion.

The declared finite Fourier field is evaluated at propagation midpoints by
frequency folding. The transverse FFT embeds the retained reciprocal rectangle
in M=2N−1; every Hamiltonian action projects back without cyclic wraparound.
The solver uses full-Hamiltonian exponential actions and saves each completed
slice separately for each appreciably populated thermal initial state. Thermal
weights retain their absolute normalization; omitted initial mass is bounded.
Reported detector probabilities have no detector-specific renormalization.

Independent checks compare the nine-mode action with the existing dense
working-Fock reference, compare longitudinal field folding at independent
midpoints, and compare an exponential action against a dense matrix exponential:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s experiments/yoshioka/benchmarks/16_snap_all_modes -p 'test_*.py' -v
```

Use a fresh external run directory containing `inputs/config.json`, the supplied
structure in `inputs/`, and SNAP coefficient/parameter files in `models/`:

```bash
python run.py prepare --output /absolute/fresh/run
python run.py run --output /absolute/fresh/run
```

Preserve preparation, field caches, checkpoints, executed sources, environment
and validation records together. A completed result is immutable. Incomplete
field construction is deliberately not silently reused. Run configuration and
source hashes bind field reuse and propagation checkpoints. Generated arrays,
logs, plots and model fits remain outside this repository. Spectral output is
−180 to 180 meV at 0.05 meV spacing with 2 meV Gaussian display FWHM. Plots are
PNG only. These checks validate the implementation of a finite model; the
requested 11-slice calculation has no independent convergence certificate.
