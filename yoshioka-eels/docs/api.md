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
