# API guide

The top-level `yoshioka` namespace exports the entry points below. Additional
APIs are available from the listed modules; use `help()` for signatures and
docstrings.

| Area | Entry points or modules |
| --- | --- |
| Free electron propagation | `TransverseGrid`, `FreePropagator`, `electron_wavenumber`, `grid_norm_squared` |
| Elastic propagation | `ElasticPropagator`, `electron_speed`, `interaction_coefficient` |
| Oscillator states | `HarmonicOscillator`, `resolve_coupling`, `fock_scattering_matrix`, `quadrature_scattering_block`, `quadrature_vacuum_amplitudes` |
| Thermal response | `ThermalOscillatorState`, `ThermalTransitionSum`, `transition_sum`; further utilities in `thermal_response` |
| Coupled channels | `coupled_channels`, `distorted_channels`, `matrix_free_channels`, `coordinate_channel_action` |
| Material Hamiltonians | `reciprocal_material`, `projected_material_action`, `finite_snap_bath` |
| Electron–phonon interactions | `electron_phonon`, `dynamical_source`, `bulk_thermal_source`, `thermal_dyson_words` |
| Two-phonon propagation | `two_phonon`, `leading_two_phonon`, `pair_galerkin`, `pair_planar_fields`, `pair_planar_propagation` |
| Depth integration | `longitudinal_integration`, `depth_fields`, `fourier_depth_fields`, `pair_depth_controls` |
| Sampling and detectors | `integration_measures`, `phonon_pair_importance`, `stratified_pairs`, `detector_cells` |
| Spectral analysis | `channel_spectra`, `spectral_data`, `mode_spectra`, `thermal_model_validation` |
| Lattice and force interfaces | `lattice_dynamics`, `mod_forces`, `snap_forces`, `molecular_dynamics`, `molecular_dynamics_checked` |
| Checkpoint utilities | `checkpointed_channel_action`, `provenance` |

For example:

```python
from yoshioka.coupled_channels import channel_basis, thermal_channel_weights
from yoshioka.channel_spectra import transition_lines
from yoshioka.lattice_dynamics import RealSpaceForceConstants
```

## Dependencies

Core propagation and oscillator operations use the base dependencies.
`LobatoSilicon.from_abtem_table()` reads the silicon potential coefficients
provided by the optional `elastic-reference` extra. Force-engine and lattice
workflows use the `lattice-reference` extra and caller-supplied input files.
Importing the package does not start those engines or load local research data.

## Numerical contracts

Read the [unit conventions](usage.md) before composing modules. Preserve
complex phases when combining coherent amplitudes. Keep thermal weights,
sampling weights, retained probability, and omitted probability explicit.
Contact terms and sequential contributions are model-specific parts of the
interaction, and should not be dropped merely to simplify a calculation.

Resource guards and checkpoint fingerprints are part of the numerical APIs.
Use fresh output paths when changing a calculation's physical inputs or
numerical settings. Unit tests use independent references and small synthetic
systems; physical convergence remains a property of each chosen calculation.

## Exact lateral translation sector

`lateral_symmetry.LateralTranslationSector((rx, ry), (ny, nx))` validates a
laterally repeated finite bath and maps its zero-Bloch electron sector to the
full reciprocal grid. `reduce_bath(bath)` retains the supplied full-specimen
active displacement normalization. `expand(coefficients)` inserts zeros in
unreachable beams without scaling the reachable complex coefficients. The
unit incident probability, physical momenta and detector probabilities are
preserved. This reduction requires a plane wave in this sector and a fully
translation-invariant interaction; it does not apply to arbitrary probes or
non-Gamma active displacements. The current projected action uses a square
reduced sector; the expanded full grid may be rectangular.

`ProjectedMaterialAction(..., coalesce_profiles=True)` sums longitudinal
profiles before storage when displacement vectors, inactive covariances, and
transverse positions are exactly identical. This shares both stored fields
and oscillator phase blocks. The cache contract records the actual grouping.
Default construction retains the original atom-by-atom representation.

`finite_optical_bath(model, repetitions, temperature_k)` recomputes all
`3*N-3` finite periodic modes and the local quantum covariance for the supplied
specimen, subtracting its two specimen-normalized active contributions once.
Translation-equivalent local covariances are checked to relative `1e-10` and
averaged at roundoff level. Raw covariances and the translation residual remain
in the returned evidence. This is local inactive Gaussian averaging, not
explicit scattering through all phonon modes.

Resource preflight includes one fully resident mapped profile during construction
and conversion, in addition to phase tensors, transfer tiles and FFT buffers.
Read-only mapped profiles are dropped after construction and each slice read.

## Finite-mode screening and all-direction active modes

`finite_mode_screen.commensurate_bath` constructs both real standing waves and
complex traveling waves in the force-constant model's cell gauge. The former
are specimen-normalized real coordinates; the latter label commensurate q and
branch. Only the three known Gamma translations are removed. A supplied bath
must be checked against the independent dense finite dynamical matrix.

`finite_mode_screen.born_mode` is an absolute single-contraction Born screen,
including longitudinal displacements and quantum creation/annihilation factors
supplied by the caller. Each atomic vertex is
`V(K) exp[-K C_full K / 2] (-i K.d_mu*)`. The full finite-cell quantum local
covariance appears once. This is the first contraction of the Gaussian
correlation function, rather than a cutoff on thermal occupation or frequency.
It has an infinite-z integral and no elastic multiple scattering. It does not
predict nonlinear two-phonon probabilities.

`sector_screen.source_profiles` and `distorted_sector` retain elastic scattering
before and after a single insertion. This focused sector screen currently
requires five repeats in each lateral direction. The source finite Fourier
field retains all three displacement directions. Every outgoing sector keeps
exactly the beams present in the declared full-supercell support. Local Taylor
order, midpoint depth sampling, and kinetic splitting require controls. A unit
incident Fourier coefficient represents unit full-cell probability.

`longitudinal_material_action.LongitudinalMaterialAction` forms the full
nonlinear projected atom interaction at every three-dimensional Fourier K.
Its infinite-oscillator block uses associated Laguerre polynomials, validated
against finite oscillator exponentials and independent Hermite quadrature.
Longitudinal motion shifts the atomic field, not just a transverse phase. Only
real standing-wave baths are accepted by this action. Its summed C-squared
field is streamed to disk with checksummed construction checkpoints; the
propagated Hamiltonian remains Hermitian.

`momentum_material_action.MomentumMaterialAction` supports complex traveling
modes and the exact joint electron/phonon lateral momentum sector. With an
initial plane wave and phonon vacuum, channel n carries electron residue
`r_n = -sum(n_mu h_mu) mod R`. Its Fourier transfer includes
`(r_out-r_in)/R`, kinetic energies use those shifted physical momenta, and masked
edge beams preserve the complete declared support. `expand` inserts the
coefficients without scaling. Complete q partners and translation-invariant
inactive covariance are required. A portable check rotates the phonon Fock
basis into real standing modes and compares the entire action with an
independent full-electron-cell calculation. This reduction differs from an
electron-only zero sector. Its current runner controls use the vacuum only
where an explicit omitted thermal probability bound is negligible; that does
not justify a vacuum assumption for an arbitrary acoustic active set.

For either nonlinear action, the caller subtracts the selected modes' quantum
covariance from the full bath and applies the remaining local Gaussian average
once. Local inactive averaging is not explicit correlated inactive dynamics.
The total occupation cutoff defines the projected space, separately from any
initial thermal ensemble. For populated initial states, a classification by
`sum(abs(n_final-n_initial))` differs from final occupation, net energy loss,
and scattering-event count; callers must state their convention. No screening
or package test establishes an active-mode cutoff or physical convergence.

`active_modes.normal_order.NormalOrderedMomentumAction` is an exact p≤2
construction variant of the joint momentum action. It retains the Gaussian
`exp[-sum(abs(K.d_mu)**2)/2]` and evaluates the remaining occupation polynomials
only where occupations are nonzero. Combining that Gaussian with the inactive
Gaussian is an algebraic identity. No displacement expansion is made. Independent
complex Laguerre-block tests and full material-action comparisons validate it.
It currently retains the same C-squared field storage as the reference action;
removing that storage needs a further matrix-free lowering/raising action.
