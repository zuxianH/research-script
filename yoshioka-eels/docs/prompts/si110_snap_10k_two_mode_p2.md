# Implement SNAP Si [110] at 10 K: two-mode coupled-channel Yoshioka, p=2

Work on the existing package. Implement and execute the bounded local pilot
described below, producing channel-resolved EELS spectra and a qualitative
comparison with the specified SNAP/TRPMD reference. Carry the task through to
actual arrays, CSV tables, PNG figures and a readable report. Keep incomplete
and failed convergence checks explicit.

## Paths and preservation

- Canonical development code:
  `/home/zuxian/Documents/research-script/yoshioka-eels`
- Original frozen project and result store:
  `/home/zuxian/Documents/PhD/Yoshioka`
- Fresh experiment output:
  `/home/zuxian/Documents/PhD/Yoshioka/benchmark_results/15_snap_two_mode/si110_snap_T10_p2_v1`
- Initial structure:
  `/home/zuxian/Documents/PhD/Yoshioka/Si_110_temp300.lmp`
  (the filename does not supply a 10 K equilibrium structure).
- SNAP coefficient file:
  `/home/zuxian/Documents/PhD/Yoshioka/SNAPotential.snapcoeff`
- SNAP parameter file:
  `/home/zuxian/Documents/PhD/Yoshioka/SNAPotential.snapparam`
- Read-only qualitative reference:
  `/home/zuxian/Documents/PhD/HiPhive_TRPMD/Si_5x7x52_10K_beads165_SNAP.npy`

Keep all new source, small configuration and tests in the package. Keep arrays,
figures, logs, checkpoints, source snapshots and numerical manifests under the
fresh PhD output. Preserve the original source, scientific environment, all
completed runs, compatibility paths and the organized result navigation. Use a
versioned new output if the specified experiment directory has incompatible
contents. Do not install anything into the original `.venv`.

Expected supplied SNAP hashes:
- snapcoeff: `04c04279a25fb52893be5ac2ef2ec48d2134168e38dff20234ce85a920e0123e`
- snapparam: `9a619a0c1e76bf81b8ef9384f6e97ee47b61a643fab02ba16221c73381aee0da`
Keep rcutfac=4.9, twojmax=8 and all other supplied fitted settings unchanged.

Read the repository/package AGENTS.md, README.md, docs/research-status.md and the
configured research workflow. Prepare an isolated task worktree with the exact
workflow pin. The machine-local override is:
`RW_WORKFLOW_PATH=/home/zuxian/Documents/research-workflow-pins/01f3a8ca70ef69161b827e768559d472d7bb9e07`.
Suggested new task ID: `yoshioka-snap10k-p2`. Integrate reviewed local source
into the canonical package when the destination is clean; preserve concurrent
changes and do not push remotely or submit scheduler jobs.

The package state inspected while preparing this brief was commit
`fdb37ebfc3fc2761e8919e6645b3b207c93a18b1`. This brief is a new task definition,
not evidence that SNAP phonons or a SNAP/10 K spectrum have already been run.

## Selected physical model

Use a **two-active-mode coupled-channel Hamiltonian with total occupation
cutoff n_a+n_b <= 2**, retaining exactly:

`(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)`.

Here p=2 limits target states. It is not a limit of two scattering events and
is not the historical leading changed-pair calculation. Propagate the projected
Hamiltonian P H P, retaining repeated scattering inside this six-state basis.
Keep the full nonlinear displacement dependence of the atomic electron
potential within this model. Do not silently substitute a Taylor-order-two,
DWBA-only or classical-trajectory calculation.

Proposed modes a,b are two real transverse Gamma optical modes, with fixed
orthogonal x/y laboratory polarizations and z along [110]. Select them from
the newly calculated SNAP modes; do not assume frequencies or reuse Tersoff
eigenvectors. Resolve degeneracies by a reproducible physical polarization
projection and save the actual vectors, frequencies, phases and normalization.
Record that individual (2,0)/(1,1)/(0,2) populations are basis-dependent within
an exactly degenerate mode subspace; their summed sector is also required.

This is a reduced finite-bath optical model. It does not activate the complete
silicon Brillouin zone or all displacement polarizations. Existing scalable
nonlinear actions support transverse active displacements; preserve that
scope. Nonzero q_z support elsewhere does not establish active u_z support.

Use SNAP for nuclear forces and harmonic phonons; use the existing Lobato
electron potential for scattering. Define a new model identifier. Old MOD
force constants, thermal covariance, acceptance results and production arrays
must not be reused as SNAP physical inputs.

## Parameters

| Parameter | Pilot | Independent checks / subsequent targets |
| --- | --- | --- |
| Temperature | 10 K, quantum harmonic statistics, zero-point motion | Retain exact thermal weights and omitted mass |
| SNAP files | Supplied pair, unchanged | Explicit LAMMPS SNAP interface; no fallback |
| Lattice | SNAP static zero-pressure equilibrium and relaxed basis | State the approximation of using a static equilibrium volume at 10 K |
| Nuclear mass | 28.085 u | Record units and isotope convention |
| FC supercell | 4x6x6 repetitions of the four-atom oriented cell, 576 atoms | 5x7x7, 980 atoms, when needed for range/cell verification |
| Central force displacement | +/-0.005 Angstrom | +/-0.0025 and +/-0.01 Angstrom |
| Active modes | Two transverse Gamma optical coordinates | Fixed polarization convention, correct specimen normalization |
| Total phonon cutoff | p=2, six states | Selected p=3 (10 states), then p=4 (15 states) |
| Electron energy / illumination | 60 keV, plane wave along [110] | Same incident probability convention throughout |
| Specimen | Four oriented z repeats, approximately 1.54 nm | Save exact relaxed atom list/cell/thickness; do not switch to 52 repeats just to match the reference filename |
| Vacuum padding | 1.2 nm on each beam-direction side | Record sensitivity if needed |
| Retained transverse support | N=33, full 33x33 reciprocal rectangle | N=65 then N=97 only after bounded field storage and cost preflight |
| Auxiliary FFT grid | M=65, current minimum M=2N-1 | Minimum M=129 for N65; M=193 for N97; preserve projection at every H action |
| Longitudinal field points | 769 | Refine separately from propagation |
| Propagation intervals | 769 | First compare 1537 at fixed field; independently vary field representation at fixed propagation |
| Internal oscillator working dimension | 32 | Compare 48; this does not change physical p=2 |
| Independent oscillator quadrature | Reuse existing Gauss-Hermite control where useful | Orders 24 and 32 for a representative complex-amplitude check |
| Detectors | 0-20, 0-5, 5-10, 10-20 mrad | Use physical electron momentum axes, retain absolute weights |
| Output energy axis | -180 to +180 meV, 0.05 meV spacing | Save native transition lines before interpolation/broadening |
| Gaussian broadening | 2 meV FWHM | Imposed display/instrument width, not an intrinsic lifetime |
| Resources | One scientific worker, one CPU/BLAS/FFT thread, 4 GiB RSS | Maximum 600 s per scientific invocation; atomic propagation checkpoints, e.g. every 16 steps |

These are pilot settings, not a convergence certificate. Freeze the chosen
finite-model criteria before interpreting results, retain existing applicable
numerical checks, and do not transfer the bulk weak floor into a different
finite-observable definition without justification.

## Implementation and bounded execution

1. Inspect and reuse the current coupled-channel, reciprocal/coordinate action,
   checkpoint and spectral APIs. Core installed code must not import benchmarks.
   Add a small explicit SNAP force interface alongside the MOD interface; do
   not change the MOD implementation or silently reinterpret its inputs.
   Use LAMMPS metal units and the supplied element mapping Si. Check energy,
   force, structure and unit consistency, then compute new SNAP force constants.
   Verify force-displacement/cell convergence, symmetry, sum-rule corrections
   and stability. Retain raw arrays/correction norms; do not clip unstable modes.

2. Define the normalized active modes on the actual four-repeat reference
   specimen. A unit-cell normalized vector cannot simply be repeated without
   the specimen normalization. Compute the matching 10 K inactive covariance
   from the same declared harmonic bath, subtract the active-mode contribution
   consistently, and count inactive averaging once. For the initial finite
   reference, a compatible commensurate periodic bath is preferred. Record
   boundary conditions and omitted rigid translations. Do not mix bulk-BZ
   weights and finite-mode weights silently, or apply an extra Debye-Waller
   factor to active modes already treated explicitly.

3. Obtain projected nonlinear interaction matrix elements from the existing
   converged oscillator construction. Do not replace P exp(-i Q.u) P with
   exp(-i Q.(P u P)); these are different truncations. Preserve SI conventions,
   target energies, electron phase, elastic background, projection and exact
   atomic-plane locations. Integrate the declared Hamiltonian, not a changed
   dispersion or fitted optical potential.

4. Fix memory preflight using the actual atom count, profiles, quadrature nodes,
   work buffers and construction-time copies. With 16 atoms, full complex
   profiles alone are approximately 0.775 GiB at N33/M65/F769, 6.10 GiB at
   N65/M129/F1537 and 27.29 GiB at N97/M193/F3073. These are estimates, not RSS
   measurements. Use streamed/windowed/memory-mapped fields before attempting
   larger cases; verify their action against the small full-storage action at
   the complex-amplitude level. Do not bypass guards or launch an allocation
   known to exceed them. Keep field and propagation controls independent.

5. Implement thermal initial-state handling explicitly. The historical
   checkpoint helper starts from vacuum, so check this rather than merely
   setting a temperature label. Use unrenormalized harmonic weights. Initial
   states may be omitted only with a recorded omitted-probability bound
   appropriate for the reported absolute errors. At 10 K optical states will
   likely be vacuum-dominated, but demonstrate this from the SNAP frequencies.

6. Run the bounded p=2 N33 pilot and necessary focused checks. Save partial
   progress and resume exact committed prefixes when an invocation hits its
   limits. Perform p=3 on the same bath/settings and selected p=4 when useful.
   Independently test field/propagation and matrix-element resolution. Advance
   electron support only when memory/cost preflight and a validated bounded
   implementation permit it. Preserve failing or unresolved comparisons.
   Do not turn this into a large unrelated convergence campaign.

## Required channel-resolved EELS outputs

- Save complex exit coefficients for each executed initial and final state,
  physical momentum axes/masks and all six p=2 final-channel probabilities.
- Build thermal spectra from transitions i -> f with loss E_f-E_i. Retain
  initial/final labels and all absolute weights before broadening.
- Sum amplitudes coherently inside one final-state channel. Sum probabilities
  over orthogonal final states and incoherent thermal initial states. Do not
  assign independent positive probabilities to interfering interaction paths.
- Distinguish returning-state elastic transitions from changed-state
  transitions, including zero-energy transfers between degenerate states.
  Save inclusive and changed-state spectra separately; do not classify every
  zero-energy transition as coherent elastic scattering.
- Produce detector-resolved absolute spectra and channel components; a
  six-channel probability table; a (2,0)/(1,1)/(0,2) comparison; and the summed
  two-phonon sector. Save CSV and NPZ arrays, not plots alone. Record any
  fractions and their denominator while retaining original absolute values.
- Save PNG figures only: total/channel spectra, channel-weight bars, detector
  comparison and relevant p/grid/depth differences. Use the same plotting
  factor for all components of a displayed total curve.

## Qualitative comparison with the supplied SNAP/TRPMD array

Inspect this reference independently of the older uncalibrated Tersoff arrays.
The newly inspected file is a read-only float64 array, shape (600,267,266),
340905728 bytes, SHA256:
`46f97e1d7c651b9367d6e34c34adbf44f8d05dc67a64db3902375e714eea41b0`.

Useful local analysis evidence is:
`/home/zuxian/Documents/PhD/HiPhive_TRPMD/plot_paper_10K_diffbeads.ipynb`
with SHA256
`00011d1959ffc2f1f345d5389c0a2dd1d1f548ae4dcbca6de7b5cd7cf901891d`.
Its inspected cells explicitly include beads=165 and assume:

- Array order (energy,y,x), with an FFT-centered energy axis.
- `timestep_fs=2.5`, `stride=3`, hence effective 7.5 fs sampling for analysis.
- `E_meV = fftshift(fftfreq(600,d=0.0075 ps))*4.13566769692386`:
  spacing about 0.919037266 meV, range -275.711180 to +274.792143 meV.
- Circular detector center (x,y)=(133,133), radius 20 **pixels**.
- `apply_kubo=False` in that notebook.
- The notebook normalizes its aperture curve by the full momentum/energy sum
  and then applies sigma=1 meV Gaussian smoothing. That sigma corresponds to
  FWHM about 2.355 meV, not the model's 2 meV FWHM.

These are inspected notebook analysis conventions, not proof of the exact
generation axes, Fourier sign, pixel-to-angle calibration, native window,
incident normalization or prior quantum correction. Inspect nearby generation
evidence and eels-studio-config.json, record provenance/status per field and
verify anything that can be verified. Never infer 20 pixels = 20 mrad.
Do not apply a second Kubo correction or silently relabel assumed axes verified.

Produce the requested qualitative figure even if absolute metadata remains
unavailable: use the documented notebook conventions as explicitly conditional
analysis assumptions, with pixel ROI and energy-sign uncertainty labelled.
If physical detector calibration is recovered, use a matched angular detector;
otherwise label the different acceptances in the legend/caption. If the energy
axis cannot be adopted consistently, retain an array-index reference panel
alongside the physical Yoshioka spectrum and explain the localized limitation.

Start from the raw read-only array when constructing comparison curves. Save
the raw reduction, every transformation, the native axis and display factors.
Retain an unscaled/raw-intensity view with separate units where needed. A
separate display-only area-normalized overlay on a declared common positive-
loss interval, initially 5-160 meV, is allowed for qualitative shape inspection.
Use one factor per total curve, record it and apply it to all its components;
do not normalize individual peaks to match. Do not shift energies to align
peaks, adjust phonon frequencies, subtract backgrounds or fit detector settings
to improve agreement. Use conservative rebinning and documented forward
broadening; do not claim that matching an added Gaussian matches unknown
native resolution. Preserve native-window/cropped mass and negative values.

The reference filename describes a 5x7x52, 165-bead TRPMD calculation, while
the new pilot has four repeats and two active Gamma optical modes. Verify
geometry where possible and prominently state this model/thickness mismatch.
The new model will have a few sharp harmonic transitions and cannot reproduce
the full dispersive/acoustic spectrum or intrinsic anharmonic linewidths.
No absolute-intensity accuracy, matched-thickness agreement, material accuracy,
runtime advantage or bulk acceptance follows from the qualitative overlay.

## Verification, records and completion

Use meaningful tests: SNAP force/units and derivative consistency; normalized
mode construction and inactive-covariance accounting; a small independent
dense/quadrature check of the nonlinear projected action; correct six-state
enumeration and p controls; thermal initial weights; detector/channel sums;
transition-energy signs; streamed/full-storage agreement; exact checkpoint
resume; and area-preserving spectral processing. Reuse existing coverage.
Run the applicable scientific suite with suitable material dependencies and
verify the installed package remains independent of benchmark imports.

All scientific arrays and the qualitative reference remain immutable inputs.
Freeze the new input/config/source/dependency hashes and per-case manifests;
retain raw logs, resource stops, failures, uncertainty, measured runtime/RSS,
completed counts and active-job status. Checkpoint through the pinned workflow.

Keep the old complete-pair four-repeat bulk benchmark NOT_ACCEPTED and its
corrected complete-net production count zero. This finite SNAP p=2 task cannot
promote it. Preserve the rejected Born-tail/control-variate results and the
separate nonlinear bath, longitudinal-displacement, thermal stability,
linewidth and external-calibration milestones. Do not restart old cases.

Finish with actual labelled pilot EELS spectra and a qualitative reference
comparison whenever the bounded pilot succeeds, even if refinements fail.
Include an executed parameter table, model/mode identifiers, exact source
revision, uncertainties, reproducible local commands and guarded resume
instructions. Mark the pilot numerically unconverged unless its checks pass.
If a genuine blocker prevents a spectrum, save the working code, exact failed
step and smallest remedy; do not manufacture a plot from assumed frequencies.

Add a clear link to the new experiment/report in the existing results start
page/Current_results view without altering sealed run directories. Report the
canonical code and result paths, then leave a code-backed local handover.
