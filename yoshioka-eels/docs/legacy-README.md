# Yoshioka propagation benchmarks

Current continuation (2 November 2026): [pair-observable discretization and finite-material validity](benchmark_results/14_two_phonon_eels/pair_convergence_20261102_v1/report.md). The final regression has 247 passing tests. Circular-product aliasing is demonstrated and a projected, dealiased action is implemented; retained-support, weak-detector depth, and bulk statistical gates remain open. The report preserves separate finite-model validity limits, the original thick-material target, linewidths, and the missing TACAW metadata.


Benchmark 1 validates free-electron paraxial envelope propagation before adding
atomic potentials, phonons, or channel couplings. The existing `Silicon/`
directory is reserved for later work.

Run from this directory with Python 3.10 or newer and NumPy, SciPy, Matplotlib:

```bash
cd /home/zuxian/Documents/PhD/Yoshioka
python3 -m unittest discover -s tests -v
python3 -m benchmarks.free_electron
```

If dependencies are missing, install this project in your chosen environment:

```bash
python3 -m pip install -e .
```

The benchmark writes `benchmark_results/free_electron/results.json`, `report.md`,
and three PNG figures. It returns exit status 1 if any acceptance criterion fails;
the tolerances are fixed, including for customized runs. Read the
[generated benchmark report](benchmark_results/free_electron/report.md) for
equations, measured results, boundary and sampling estimates, and limitations.
Results include the 64², 128², and 256² Gaussian comparison at fixed box size.

All requested parameters are configurable. For example:

```bash
python3 -m benchmarks.free_electron \
  --energy-kev 60 --lx-nm 10 --ly-nm 10 --nx 256 --ny 256 \
  --width-nm 0.2 --distances-in-zd 0 0.25 1 2 \
  --mode-x 3 --mode-y -5 --composition-steps 100 \
  --refinement-grids 64 128 256 --output benchmark_results/free_electron
python3 -m benchmarks.free_electron --help
```

The reusable implementation is `yoshioka/free_propagation.py`. Its public API
uses SI units and arrays shaped `(..., ny, nx)`; leading dimensions are independent
waves or channels. For example, with `T_j` in joules and lengths in metres:

```python
from yoshioka import FreePropagator, TransverseGrid, electron_wavenumber

grid = TransverseGrid(lx_m=10e-9, ly_m=10e-9, nx=256, ny=256)
propagator = FreePropagator(grid, electron_wavenumber(T_j))
propagated = propagator.propagate(initial_wave, distance_m)
```

The reference solutions live separately in `benchmarks/reference_solutions.py`.
The Gaussian reference is the infinite-domain solution, while the FFT is
periodic. The benchmark retains the specified global phase and does not include
the longitudinal carrier. Passing this benchmark validates free propagation and
the absence of unintended channel mixing; it does not yet validate
electron–phonon scattering or EELS normalization.

## Benchmark 2: static Si [110] elastic diffraction

The second benchmark reads `Si_110_temp300.lmp` unchanged, explicitly assigns
LAMMPS type 1 to Si, and verifies the cell, fourfold periodic coordination, and
projected dumbbells. The coordinates are interpreted as ångströms; the filename
does not trigger thermal displacements or smearing.

Create an isolated reference environment and run all tests and the benchmark:

```bash
cd /home/zuxian/Documents/PhD/Yoshioka
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-benchmark2.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m benchmarks.elastic_si110
```

The pinned environment contains abTEM 1.0.11 and ASE 3.29.0. The optional project
extra `elastic-reference` also declares the direct reference dependencies. The
SI solver itself depends only on NumPy and SciPy; tests needing abTEM are skipped
when it is absent.

The run begins with the 52-atom, 4.991948 nm specimen at 128² and eight finite
Lobato slices per original z repeat. It then measures every combination of
64²/128²/256², 8/16/32 slices per repeat, and 13/26/52 z repeats. If the requested
physical convergence fails, two additional thick-specimen refinements are run
to investigate it. Use `--help` for configuration or `--skip-extra-refinement`
to omit only those extra diagnostics.

Outputs are in `benchmark_results/elastic_si110/`:

- `results.json`: raw-phase agreement, beam and detector probabilities,
  convergence pairs, package versions, native and SI conventions, and checks.
- `report.md` and eight PNG figures: structure, potential, wave intensities,
  diffraction, differences, thickness series, convergence, and norm histories.
- `specimens.npz`: the exact repeated atomic positions and cells.
- `shared_inputs/*.npz`: actual unit-cell potential arrays, entrance waves,
  slice thicknesses, and explicit complete-specimen slice maps.
- `waves/*.npz`: both complex exit waves, intensities, angular-region masks,
  and per-slice norms.

Read the [measured elastic report](benchmark_results/elastic_si110/report.md)
before interpreting the diffraction as converged. Implementation agreement and
physical convergence have separate checks. The default run returns exit status
1 if either fails, while preserving all diagnostic outputs and fixed tolerances.

The finite reference potential uses abTEM's periodic bulk-image convention at
the exact specimen faces, without vacuum padding. Unit slices are reused
losslessly along z, verified against a direct repeated-atom potential build.
To reconstruct the exported SI potential of the 13-repeat specimen:

```python
import numpy as np

with np.load("benchmark_results/elastic_si110/shared_inputs/n128_s8.npz") as data:
    native = data["potential_v_angstrom_sxy"][data["slice_unit_indices_r13"]]
    projected_potential_si = native.swapaxes(-2, -1) * 1e-10  # (slice,y,x), V m
    thicknesses_m = data["slice_thickness_m_r13"]
```

The integrated potential receives no further thickness factor. Both solvers use
the same entrance probability, transmission-then-propagation order, and numerical
antialias filtering. No mean potential is removed and no output is independently
renormalized. Passing checks propagation through the shared static potential;
it does not independently validate potential generation, phonon scattering,
thermal averaging, or EELS spectral normalization.

## Benchmark 3: exact T=0 harmonic-oscillator response

This benchmark applies `F(Q) = exp(-i Q u)` once to the oscillator vacuum and
checks complex transition amplitudes, unbroadened weights, energy losses,
Debye–Waller factors, general matrix elements, and the time correlation over
one period. It evaluates coordinate overlaps with Gauss–Hermite quadrature and
independently exponentiates the complex128 Fock coordinate matrix. The numerical
routes do not use the analytical Poisson coefficients.

Run the benchmark and all Benchmark 1–3 tests in the existing environment:

```bash
cd /home/zuxian/Documents/PhD/Yoshioka
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m benchmarks.harmonic_response
.venv/bin/python -m benchmarks.harmonic_response --help
```

NumPy, SciPy, and Matplotlib suffice for Benchmark 3; it uses no abTEM potential,
atomic structure, MD, or occupation sampling. `requirements-benchmark2.txt` pins
the environment used for the recorded results. Constants follow the installed
`scipy.constants` configuration documented for Benchmark 2 (CODATA 2022 in this
environment); actual values, units, uncertainties, package versions, and the
constant-table source hash are recorded in the new results.

Defaults are an illustrative mass of 28.085 u, oscillator quantum of 60 meV,
and positive angular Q corresponding to eta=0.1. These are not a complete
crystalline Si model. The reusable physical API uses SI units. Conversion is
`M_kg = M_u * atomic_mass`, `E_J = E_meV * 1e-3 * elementary_charge`,
`omega = E_J / hbar = 2*pi*f`, and `Q_m_inv = Q_inv_angstrom * 1e10`.
There is no extra factor of `2*pi` in the scattering phase. The quadrature uses
`x = u / ell`, with `ell = sqrt(2)*u_zp`.

For example, change the oscillator, selected coupling, and Gaussian width in a
separate output directory:

```bash
.venv/bin/python -m benchmarks.harmonic_response \
  --mass-amu 28.085 --energy-mev 60 --eta 0.5 --fwhm-mev 2 \
  --output benchmark_results/harmonic_response_eta05
```

Use signed `--q-inv-angstrom` instead of `--eta` to specify Q. If both are
supplied, inconsistent values are rejected. The eta sweep, physical cutoffs,
working dimensions, quadrature orders, and acceptance tolerances are configurable.
The script exits 1 for failed acceptance checks and saves their measured errors;
it never silently relaxes tolerances or normalizes incomplete retained weights.

The public numerical functions live in `yoshioka/harmonic_oscillator.py`:

```python
from yoshioka import (
    HarmonicOscillator, fock_scattering_matrix,
    quadrature_scattering_block, quadrature_vacuum_amplitudes, resolve_coupling,
)

oscillator = HarmonicOscillator.from_amu_mev(28.085, 60)
Q, xi, eta = resolve_coupling(oscillator, eta=0.1)
p, D = 7, 64
vacuum_fock = fock_scattering_matrix(xi, D)[:p+1, 0]
vacuum_quad = quadrature_vacuum_amplitudes(xi, p, 128)
general_block = quadrature_scattering_block(xi, n_max=6, quadrature_order=128)
energy_losses_j = oscillator.energy_losses_j(p)  # n*hbar*omega; elastic at zero
```

Physical retained occupations `n=0..p` and working dimension `D` are separate.
Exponentiating the prematurely truncated coordinate matrix can produce a
normalized state with wrong retained coefficients. The benchmark demonstrates
this explicitly and evaluates omitted weights using a stable Poisson survival
function. A finite extracted block need not be unitary.

The recorded default run passes all Benchmark 3 acceptance checks. Across
eta=0, 0.01, 0.1, 0.5, 1, 4, the adaptive cutoffs are respectively
p=0, 4, 7, 11, 14, 25, with exact omitted weight below 1e-12. D=32 and 32
quadrature nodes are sufficient among the tested settings for these adaptive
vectors and the 0..6 general block. Studying every requested cutoff through
p=32 requires D=64; at eta=4 it also requires 64 quadrature nodes. Final
evaluations use D=128 and 256 nodes, with independent refinement checks.

Outputs are separate in `benchmark_results/harmonic_response/`:

- `results.json`: SI inputs, constants and sources, complex amplitudes, errors,
  stable tails, moments, convergence sequences, and separate acceptance statuses.
- `arrays/*.npz`: unbroadened line energies, numerical/reference amplitudes and
  weights, general matrix blocks, and correlations.
- `broadened_spectrum.npz`: SI energy grid and elastic/inelastic densities in
  J^-1. Plots convert density to meV^-1; Gaussian area equals retained weight.
- Five PNG figures and a [measured report](benchmark_results/harmonic_response/report.md)
  with exact reproduction commands and the deliberate truncation diagnostic.

Unbroadened T=0 lines have nonnegative energy loss. Gaussian tails below zero
are broadening tails. The elastic amplitude is `exp(-eta/2)` and its weight is
`exp(-eta)`; oscillator weights exclude atomic scattering form factors and
electron-flux factors, so they are not absolute detector probabilities.

Existing Benchmark 1–2 results are preserved. Benchmark 2's recorded physical
convergence status remains **FAIL**. Passing Benchmark 3 validates fixed-Q
harmonic matrix elements and spectral weights. Repeated scattering, detector
integration, momentum-dependent prefactors, and general coupled electron–phonon
multislice propagation require separate validation; they need not produce
Poisson weights.

## Benchmark 4: finite-temperature harmonic-oscillator response

Benchmark 4 reuses the validated Benchmark 3 oscillator matrices and forms the
thermal probability sum `W_ell = sum_(n-m=ell) p_m * abs(F_nm)**2`. Positive
`ell=n-m` is electron energy loss, negative ell is gain, and ell=0 is the
complete zero-energy-transfer line. The thermal sum is not a propagation solver.

Run all regressions and the new benchmark in the existing environment:

```bash
cd /home/zuxian/Documents/PhD/Yoshioka
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m benchmarks.thermal_harmonic_response
.venv/bin/python -m benchmarks.thermal_harmonic_response --help
```

The default 36 cases cover T=0, 10, 100, 300, 600, 1000 K and
eta=0, 0.01, 0.1, 0.5, 1, 4. The selected illustrative spectrum uses 300 K,
eta=0.1, mass 28.085 u, and quantum energy 60 meV. The independent reference
uses scaled Bessel/logarithmic evaluation of the Skellam weights, with explicit
eta=0 and T=0 branches and stable positive Poisson-convolution tail sums.
The numerical calculation uses Fock-state transitions, with a separate
Gauss–Hermite overlap check; it does not use the exact weights to generate
numerical gain or loss lines. Operator matrices are cached across temperatures.

The reusable thermal API is `yoshioka/thermal_response.py`:

```python
from yoshioka import (
    HarmonicOscillator, ThermalOscillatorState,
    fock_scattering_matrix, resolve_coupling, transition_sum,
)
from yoshioka.thermal_response import final_state_cutoff

oscillator = HarmonicOscillator.from_amu_mev(28.085, 60)
Q, xi, eta = resolve_coupling(oscillator, eta=0.1)
thermal = ThermalOscillatorState(oscillator.quantum_energy_j, 300)
m_max = thermal.initial_cutoff(1e-13)
p_m = thermal.probabilities(m_max)  # Retained probabilities are not normalized.
F = fock_scattering_matrix(xi, working_dimension=192)
n_max = final_state_cutoff(F, p_m, tolerance=1e-12)
response = transition_sum(F, p_m, n_max)
energy_losses_j = response.ell * oscillator.quantum_energy_j
```

Initial thermal cutoff, final-state cutoff, working matrix dimension, signed
energy window, and quadrature order are varied independently. Exact initial
omitted weight targets 1e-13, final omitted weight targets 1e-12, and the
independently chosen signed line window omits less than 1e-12. The signed final
norm-deficit estimate is saved even when roundoff makes it slightly negative.
A finite-dimensional displacement matrix can be unitary while underresolved.
No retained distribution or spectrum is renormalized.

The default initial cutoffs are m_max=0, 1, 4, 12, 25, 42 in temperature order.
The worst default case, eta=4 at 1000 K, requires n_max=65 and a signed line
window [-21,33]. Among tested settings it needs D=96 and 128 quadrature nodes.
Final evaluations use D=192 and 256 nodes with independent refinement checks.
All per-temperature/per-eta settings are recorded in the generated report.

Parameters, signed Q or eta, temperatures, sweeps, numerical sequences,
tolerances, and Gaussian FWHM are configurable. For example:

```bash
.venv/bin/python -m benchmarks.thermal_harmonic_response \
  --temperature-k 600 --eta 0.5 --fwhm-mev 2 \
  --output benchmark_results/thermal_harmonic_response_T600_eta05
```

Outputs in `benchmark_results/thermal_harmonic_response/` include
`results.json`, cached operator matrices and per-case NPZ arrays, six figures,
and the [scientific report](benchmark_results/thermal_harmonic_response/report.md).
The report separates line-weight acceptance, cutoff convergence, detailed
balance, coherent amplitude versus full zero-energy weight, net one-quantum
lines versus leading one-phonon terms, and Gaussian-area checks. Densities use
J^-1 internally and meV^-1 in plots. Gaussian tails are distinct from physical
negative-energy gain lines. Finite windows are recorded without area rescaling.

This uses the existing NumPy/SciPy/Matplotlib dependencies and the same recorded
SciPy physical constants as earlier benchmarks, adding the Boltzmann constant.
No structure, potential, MD, or finite-temperature sampling is required.
Benchmark 2's saved physical-convergence status remains **FAIL**. No validated
Benchmark 3B channel integrator is present, so optional channel propagation is
explicitly **NOT_PERFORMED**, with no change to a separate 3B status. Passing
Benchmark 4 validates thermal oscillator response and probability averaging;
full finite-temperature material EELS accuracy requires separate validation.

## Benchmarks 5–7: MOD phonons, Born channels, and EELS convergence

Install the additional pinned lattice dependencies without changing earlier
benchmark packages, then run the material calculation:

```bash
cd /home/zuxian/Documents/PhD/Yoshioka
.venv/bin/python -m pip install -r requirements-lattice.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.si_lattice_dynamics \
  --config benchmarks/configs/lattice.json
```

Benchmark 5 uses the supplied `Si.tersoff.mod` with LAMMPS `tersoff/mod` in metal
units. It preserves both original input files and creates a separate equilibrium
structure. The installed LAMMPS wheel uses its matching MPICH runtime from the
virtual environment; no system MPI replacement or incompatible library symlink
is used. The `temp300` input filename does not trigger thermal motion.

The equilibrium lattice parameter is 5.429000041 Å. Independent central
differences use displacements 0.02, 0.01, 0.005, and 0.0025 Å in 3×4×4 and
4×5×5 oriented-cell supercells. The measured harmonic range is 3.83888 Å,
exceeding the 3.3 Å pair cutoff through many-body coupling. Raw and corrected
force constants are kept separately. A second-order hiPhive model is tested
against independent held-out MOD forces and force constants while training
size, displacement, interaction radius, supercell, and regression settings vary.

The [material report](benchmark_results/si_lattice_dynamics/report.md) and
`manifest.json` document the equilibrium mapping, primitive/oriented folding,
Fourier and eigenvector conventions, Phonopy-compatible arrays, trained model,
dispersion, DOS, and quantum covariances at 0 and 300 K. Force supercell size is
separate from covariance q sampling. This benchmark passes numerical/model
reproduction checks; agreement with MOD is not agreement with experiment or DFT.

Run the actual one-phonon longitudinal source calculation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.one_phonon_born \
  --config benchmarks/configs/born.json
```

Benchmark 6 uses the same finite atomic segment and phonon gauge in numerical
propagation and independent Born references. Its initial bulk-periodic model
contains 32 atoms and all 93 physical modes after removing only three rigid
translations. Lobato Si, with interaction energy `h=-eV`, supplies the electron
potential; MOD is used only for phonons. Every displacement component, including
longitudinal coupling, is retained. The Debye–Waller covariance includes all
physical modes. A radial single-atom quadrature checks Fourier normalization,
charge sign, units, and the angular/cycles wavevector conversion.

The source solver uses Benchmark 1 free propagation, the phonon Hamiltonian
phase, and numerical Simpson source increments. It is compared with an
independent finite-window quadrature and a separately padding-validated
full-space atomic Born sum at `Qz=-kappa`. Actual complex amplitudes, detector
and mode weights, cancellation, degenerate rotations, incident-area dependence,
and g² scaling are tested. The undepleted incident weight plus first-order
inelastic weight is not required to equal one. The
[Born report](benchmark_results/one_phonon_born/report.md) and `manifest.json`
separate numerical checks from this physical approximation.

Run or resume the controlled convergence workflow:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.si_eels_convergence \
  --config benchmarks/configs/convergence.json
```

Benchmark 7 keeps the 32-atom specimen, thickness, transverse physical cell,
60 keV illumination, T=0, detectors, absolute energy window, and 2 meV FWHM
fixed during numerical sequences. It refines electron spacing, longitudinal
integration, and padding before bulk phonon quadrature, then tests joint
refinement. Primitive reciprocal meshes use all q points and six branches.
Outgoing channels have explicit Bloch shifts; arbitrary phonon phases are not
placed in an incompatible periodic FFT box. Primitive displacements contain
the required `1/sqrt(Nq)` exactly once, independently verified against full-model
normalization. The actual atom list and thickness do not grow with the q mesh.

The independently converged Benchmark 5 covariance is held fixed while the
transition quadrature is refined. Primitive covariance sampling is checked
separately. Predeclared detector areas, peak groups, qx regions, exact-forward
contribution, and all 8×8 momentum bins remain in the error tables. Spectra
are absolute; no dominant elastic line, per-run normalization, increased
broadening, or unsupported extrapolation is used to hide numerical errors.

The default convergence limits are 1800 seconds, 2 GiB resident memory, and 30
new EELS configurations. Per-q memory-mapped checkpoints resume interrupted
work exactly. Runtime checks stop refinement between q sectors; finalizing
completed arrays into spectra and reports adds separately reported processing
time. To change resource limits without changing the physical model:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.si_eels_convergence \
  --config benchmarks/configs/convergence.json \
  --runtime-seconds 1800 --memory-gib 2 --max-refinements 30
```

Cache reuse checks the physical model, numerical input checksums, validation
statuses, constants, environment, and calculation source versions. Original and
current manifests are archived. A correction to reporting metadata alone does
not invalidate numerical work when those scientific inputs are unchanged.

The [convergence report](benchmark_results/si_eels_convergence/report.md) and
machine-readable results give separate material, electron, phonon/covariance,
observable, and joint-refinement statuses. An unfinished category remains
`INCOMPLETE`, with its actual remaining error; best computed settings are not
called accepted unless all declared gates pass. No numerical occupation p sweep
inside this one-phonon formula establishes multiphonon accuracy. Thermal
material EELS and a dynamical one-phonon source solver are separate extensions.

Reusable code is in `yoshioka/mod_forces.py`, `lattice_dynamics.py`, and
`electron_phonon.py`. Exact configurations are in `benchmarks/configs/`.
Run all earlier and new regressions with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

Earlier results and statuses are preserved: Benchmark 2 remains physically
underconverged, Benchmark 3B remains unavailable, and Benchmark 4 remains PASS.
These calculations do not establish general Yoshioka or full material EELS
accuracy.
