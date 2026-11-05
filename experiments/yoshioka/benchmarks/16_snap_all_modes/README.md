# All internal modes in finite Si [110]

`run.py` prepares a SNAP-force harmonic fit with hiPhive and propagates the
full three-dimensional rigid-ion Lobato displacement phases in a total phonon
occupation basis. The working single-oscillator exponential is evaluated with
the configured `working_dimension` before projection. Active longitudinal
displacements are retained. This differs from the transverse-only fast action.

The current run uses the four-atom 1×1×1 cell, 10 K, 60 keV plane wave, all nine
internal modes, p=2 (55 channels), N=11, M=21, 193 depth-field points, 11
propagation slices, D=32, and a 0–30 mrad detector. The cancelled N=97/F=769
run is preserved separately. The three rigid translations are removed.

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

Future final wave output is `exit_probabilities.npz`: `exit_probabilities` is a
float64 array of unweighted |A|² with axes (initial state, final phonon channel,
ky, kx). It retains channel occupations, initial-state indices, momentum/angle
coordinates, detector masks/bounds, mode energies, full thermal weights and
the executed initial-state weights. Apply thermal weights once when forming
spectra. Float64 probabilities use half the uncompressed bytes of complex128
amplitudes; the compressed size depends on the data.

Propagation and unfinished restart checkpoints remain complex. After all final
outputs and the completion record are saved, verified completed checkpoint waves
are removed and their JSON records point to the final probability file. Existing
completed runs and their frozen sources remain unchanged. The probability save
supports momentum probabilities and EELS; phase-dependent reanalysis requires
complex amplitudes and must be explicitly requested for a future run.

Field construction uses deterministic four-transfer tiles. One parent process
assembles and verifies independent chunk files produced by 1, 2, or 4 worker
processes; each process uses one BLAS thread. Completed tiles have checksums
and can be reused after interrupted construction. The parent fills conjugate
partners only after checking exact canonical coverage. Propagation slices stay
sequential; processes do not split the coupled wave into independent channels.

Independent checks compare the nine-mode action with the existing dense
working-Fock reference, compare longitudinal field folding at independent
midpoints, compare an exponential action against a dense matrix exponential,
verify agreement across 1/2/4 workers, and reject corrupted reusable chunks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s experiments/yoshioka/benchmarks/16_snap_all_modes -p 'test_*.py' -v
```

Use a fresh external run directory containing `inputs/config.json`, the supplied
structure in `inputs/`, and SNAP coefficient/parameter files in `models/`:

```bash
python run.py prepare --output /absolute/fresh/run
python run.py benchmark-and-run --output /absolute/fresh/run
```

Preserve preparation, field caches, checkpoints, executed sources, environment
and validation records together. A completed result is immutable. Phonon reuse
is bound to a phonon-only input contract and artifact checksums; changing only
electron or execution settings does not require refitting. Preserve original
preparation records when adopting older fits into this contract.

The benchmark measures full construction, process startup, assembly and hashing
for 1/2/4 workers and uses the fastest numerically equivalent result. A partially
completed benchmark without a full-build timing is preserved and rejected as a
timing sample. A completed selection is reusable. Resume completed selections
and propagation with the `run` stage. Run configuration and source hashes bind
field reuse and propagation checkpoints. Generated arrays,
logs, plots and model fits remain outside this repository. Spectral output is
−180 to 180 meV at 0.05 meV spacing with 2 meV Gaussian display FWHM. Plots are
PNG only. These checks validate the implementation of a finite model; the
requested 11-slice calculation has no independent convergence certificate.
