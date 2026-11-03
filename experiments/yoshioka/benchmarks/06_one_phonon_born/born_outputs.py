"""Absolute spectra, momentum maps, convergence plots, and Benchmark 6 report."""

import numpy as np


def write_outputs(results,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=results['selected_case'];config=results['parameters'];controls=results['controls'];figures=[]
    with np.load(output/selected['artifact']) as data:arrays={name:data[name] for name in data.files}
    def save(fig,name):fig.savefig(output/name,dpi=175,bbox_inches='tight');plt.close(fig);figures.append(name)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for name in ('full_0_20_mrad','central_0_5_mrad','annulus_5_10_mrad','annulus_10_20_mrad'):
        axes[0].plot(arrays['energy_grid_mev'],arrays[f'spectrum_{name}_per_mev'],label=name)
        axes[0].plot(arrays['energy_grid_mev'],arrays[f'reference_spectrum_{name}_per_mev'],'--',linewidth=.8)
    axes[0].set(xlabel='Electron energy loss (meV)',ylabel='Absolute inelastic weight / meV',title='Numerical solid; independent Born dashed')
    axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    mode=selected['mode_errors'];energies=[row['energy_mev'] for row in mode]
    axes[1].semilogy(energies,[row['reference_weight'] for row in mode],'o',markerfacecolor='none',label='Born')
    axes[1].semilogy(energies,[row['numerical_weight'] for row in mode],'x',label='Propagated channels')
    axes[1].set(xlabel='Mode energy (meV)',ylabel='Mode-resolved inelastic weight',title=f"All {len(mode)} physical modes retained")
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    save(fig,'absolute_spectra_and_mode_weights.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    numerical=np.sum(abs(arrays['amplitudes'])**2,axis=0);reference=np.sum(abs(arrays['born_full_space_amplitudes'])**2,axis=0)
    q=arrays['qxy_m_inv']*1e-10
    for ax,values,title in zip(axes,(reference,numerical,numerical-reference),('Born intensity','Propagated intensity','Numerical − Born')):
        im=ax.scatter(q[:,0],q[:,1],c=values,cmap='coolwarm' if '−' in title else 'viridis',s=65,marker='s')
        ax.set(xlabel='Angular qx (Å⁻¹)',ylabel='Angular qy (Å⁻¹)',title=title,aspect='equal')
        fig.colorbar(im,ax=ax,label='Weight per discrete electron Fourier mode')
    save(fig,'momentum_intensity_and_difference.png')
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    for ax,category,xkey,xlabel in zip(axes,('transverse','longitudinal','padding'),('dx_angstrom','propagation_step_angstrom','padding_angstrom'),('Transverse Δx (Å)','Propagation step (Å)','Vacuum padding per side (Å)')):
        rows=results['cases'];rows=[row for row in rows if row['category']==category]
        ax.loglog([row[xkey] for row in rows],[row['amplitude_relative_vector_error'] for row in rows],'o-')
        ax.axhline(config['amplitude_relative_tolerance'],color='k',linestyle='--',label='Amplitude target')
        ax.set(xlabel=xlabel,ylabel='Complex amplitude-vector error vs Born');ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'electron_and_padding_convergence.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    rows=controls['g_scaling'];axes[0].plot([row['g']**2 for row in rows],[row['probability'] for row in rows],'o-',label='Numerically propagated')
    axes[0].plot([row['g']**2 for row in rows],[row['quadratic_expected'] for row in rows],'k--',label='P(g=1) g²')
    axes[0].set(xlabel='g²',ylabel='Total inelastic weight',title='Undepleted first-order source');axes[0].legend(fontsize=8)
    history=arrays['source_norm_history'];axes[1].plot(np.linspace(0,selected['interaction_length_angstrom'],len(history)),history)
    axes[1].set(xlabel='Propagation distance (Å)',ylabel='Computed inelastic-channel weight',title='Actual longitudinal source integration')
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'g_scaling_and_longitudinal_history.png')
    lines=['# Benchmark 6: plane-wave one-phonon Si scattering in the first-Born limit','',f"Overall status: **{results['status']}**; numerical checks: **{results['numerical_status']}**; Benchmark 5 material status: **{results['material_status']}**.",'',
           'This benchmark connects the validated harmonic phonons to actual longitudinal electron-channel propagation. It uses all physical modes of a bulk-periodic phonon model, a finite atomic segment along z, and periodic transverse electron boundaries. It is not a relaxed free-surface calculation. MOD supplies lattice dynamics only; Lobato supplies the beam-electron interaction.','',
           '## Physical model and SI conventions','',str(results['model_specification']),'',
           'Full-model mass-weighted eigenvectors obey sum_iα |e_iαλ|²=1. d_iλ=sqrt[hbar/(2Mi omegaλ)] e_iλ; no further 1/sqrt(N) factor is applied. Only the three explicitly known rigid translations are removed. The Debye–Waller covariance includes every physical mode of this declared model, not just an excited mode. Site equivalence and diagonal covariance are verified rather than assumed; the unequal Cartesian diagonal entries are retained.',
           f"Covariance verification: {selected['covariance_diagnostic']}. Atom positions and eigenvectors in modes.npz preserve the exact common gauge used by numerical and reference amplitudes.",
           'The Lobato table uses cycles/Å and scattering factors in Å. With angular Q, h_tilde(Q)=−(2π hbar²/me) f_Lobato(Q/(2π)), in J m³. h=−e V is negative for the positive electrostatic atom potential. All electron kinematic constants use the project SciPy configuration; older ASE constants are not substituted. Direct radial quadrature of the real-space atom potential verifies its full 3D Fourier transform, signs, units, and 2π factors.',f"Single-atom audit: {results['single_atom_audit']}",'',
           '## Independent coupling and longitudinal propagation','',
           'Numerical coupling construction inverse-transforms the DW-convolved atomic potential along z and applies an actual fourth-order finite-difference gradient. The transverse gradient uses the exactly equivalent fourth-order stencil Fourier symbol. Actual finite-atom position/polarization sums assemble all three Cartesian displacement components. This numerical construction is checked against the independent atomic full-space Fourier expression G_tilde=−i sum_i h_tilde exp(−Wi) exp(−iQ·Ri) (Q·d_iλ*).',
           'The longitudinal solver advances initially empty channels by exact Benchmark 1 homogeneous free propagation, the phonon phase exp(−i omega dz/v), and Simpson endpoint/midpoint source increments. The incident plane wave remains chi0=1/sqrt(A). The observable electron Fourier modes are propagated as a Galerkin subset: they decouple exactly in this undepleted plane-wave first-Born problem, so unobserved Fourier modes do not feed them. All physical phonon modes remain present. This optimization is not a dense coupled-channel or depleted-beam approximation.',
           'The independent finite-window reference uses analytical Gaussian-convolved Yukawa plane kernels and their exact derivatives, followed by adaptive Gauss–Kronrod longitudinal quadrature. It uses neither the sampled source grid nor the channel integrator. From c(L)=−ig exp(−i kappa L)/(hbar v) integral_0^L exp(+i kappa z) G_q(z) dz and the exp(−iQz) Fourier transform, the full-space expression samples Qz=−kappa. This sign is checked by a pure longitudinal single-atom source.',
           f"Finite-window reference: {controls['finite_window_quadrature']}. Numerical vs finite-window complex amplitude error={controls['finite_window_relative_amplitude_error']:.6e}; finite-window vs full-space difference={controls['finite_window_vs_full_space_relative_amplitude_difference']:.6e}. No arbitrary phase alignment is applied.",
           f"Independent coupling Fourier errors: {controls['coupling_fourier_relative_errors']}. Longitudinal/area/cancellation control: {controls['longitudinal_single_atom']}. Degenerate-subspace control: {controls['degenerate_subspace']}.",'',
           '## Computed settings and errors','',
           '| Category | Grid | Δx (Å) | z sample spacing (Å) | Propagation step (Å) | Padding (Å) | Relative amplitude error | Inelastic weight |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in results['cases']:lines.append(f"| {row['category']} | {row['grid']} | {row['dx_angstrom']:.6f} | {row['longitudinal_sample_spacing_angstrom']:.6f} | {row['propagation_step_angstrom']:.6f} | {row['padding_angstrom']} | {row['amplitude_relative_vector_error']:.6e} | {row['numerical_inelastic_weight']:.9e} |")
    lines.extend(['',f"Selected absolute inelastic weight={selected['numerical_inelastic_weight']:.12e}. Maximum mode-intensity error={selected['maximum_mode_intensity_error']:.6e}; maximum detector/peak-group acceptance error={selected['observable_errors']['maximum_acceptance_error']:.6e}, at {selected['observable_errors']['worst_observable']}. Weak probability scale={config['weak_probability_scale']:.1e}; every weak observable also has its absolute error recorded. No spectrum is normalized to unit area.",'',
                  'The undepleted incident probability plus first-order inelastic probability is not required to equal one. The g sweep tests quadratic first-order scaling while retaining the phonon Hamiltonian. The full momentum disk, fixed annuli, qx half-regions, exact forward mode, and energy groups are fixed before refinement; none is removed because it fails. The primary lines have positive phonon energies at T=0. Any Gaussian tail below zero is broadening, not a physical gain transition.',
                  f"Final spatial/integration/padding comparisons: { {name:[(row['coarse'],row['fine'],row['maximum_acceptance_error'],row['worst_observable']) for row in rows] for name,rows in results['refinement_comparisons'].items()} }",'',
                  '## Checks and reproduction','','| Check | Status |','|---|---|'])
    for name,value in results['checks'].items():lines.append(f"| {name} | {'PASS' if value else 'FAIL'} |")
    lines.extend(['',f"Costs: {results['resource_costs']}. Optional complete-propagator check: {results['optional_complete_propagator']}. Earlier statuses preserved: {results['prior_benchmark_statuses_preserved']}.",'',
                  '```bash','cd /home/zuxian/Documents/PhD/Yoshioka','.venv/bin/python -m benchmarks.si_lattice_dynamics --config benchmarks/configs/lattice.json','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.one_phonon_born --config benchmarks/configs/born.json','.venv/bin/python -m unittest discover -s tests -v','```','',
                  'manifest.json supplies Benchmark 7 with explicit material provenance, physical model, units, constants, computed arrays, checksums, and statuses. Agreement validates this declared harmonic one-phonon coupling and first-order longitudinal source propagation. It does not establish multiphonon, repeated inelastic scattering, elastic dynamical diffraction, finite-temperature material EELS, or general Yoshioka accuracy. The small finite phonon model still requires the separate sampling study in Benchmark 7.',''])
    for name in figures:lines.extend([f'![{name}]({name})',''])
    (output/'report.md').write_text('\n'.join(lines))
    return figures
