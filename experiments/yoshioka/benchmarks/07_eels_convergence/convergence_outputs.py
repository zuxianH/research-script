"""Actual convergence/error/cost reporting; incomplete categories stay explicit."""

import time

import numpy as np


def write_outputs(results,output):
    output_started=time.monotonic()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=[];spec=results['model_specification'];config=results['parameters']
    complete=[row for row in results['cases'] if row['status']=='COMPLETE']
    def save(fig,name):fig.savefig(output/name,dpi=170,bbox_inches='tight');plt.close(fig);figures.append(name)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,category in zip(axes.flat,('electron_grid','longitudinal','phonon_sampling','padding')):
        rows=results['comparisons'].get(category,[])
        if rows:
            if category=='electron_grid':xx=[spec['transverse_lengths_m'][0]*1e10/row['fine_settings']['grid'] for row in rows];xlabel='Fine transverse Δx (Å)'
            elif category=='longitudinal':xx=[2*(spec['specimen_thickness_m']*1e10+2*row['fine_settings']['padding_angstrom'])/row['fine_settings']['samples'] for row in rows];xlabel='Fine propagation step (Å)'
            elif category=='padding':xx=[row['fine_settings']['padding_angstrom'] for row in rows];xlabel='Fine vacuum padding per side (Å)'
            else:xx=[row['fine_settings']['mesh'] for row in rows];xlabel='Fine primitive reciprocal mesh n³'
            for yy,label,style in (([row['spectral_changes']['maximum_relative_L1'] for row in rows],'Spectrum L1','o-'),([row['observable_changes']['maximum_acceptance_error'] for row in rows],'Worst predeclared observable','s-')):
                positive=np.array(yy)>0
                ax.semilogy(np.array(xx)[positive],np.array(yy)[positive],style,label=label)
            ax.axhline(.01,color='k',linestyle='--',label='1% target')
            ax.set(xlabel=xlabel,ylabel='Successive-refinement change')
            ax.legend(fontsize=8)
        else:ax.text(.5,.5,'No completed comparison',ha='center',transform=ax.transAxes)
        ax.set_title(category.replace('_',' '));ax.grid(alpha=.2)
    save(fig,'numerical_convergence.png')
    rows=results['covariance_study']
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),layout='constrained')
    if rows:
        xx=[row['mesh'][0] for row in rows]
        trace=[np.trace(row['covariance_m2'])*1e20 for row in rows]
        axes[0].plot(xx,trace,'o-',label='Primitive BZ quadrature')
        axes[0].axhline(np.trace(spec['covariance_m2'])*1e20,color='k',linestyle='--',label='Converged Benchmark 5 input')
        axes[1].semilogy(xx,[row['relative_difference_from_independent_B5_covariance'] for row in rows],'o-')
        axes[1].axhline(config['covariance_relative_tolerance'],color='k',linestyle='--')
    axes[0].set(xlabel='Primitive mesh n³',ylabel='Trace of zero-point covariance (Å²)');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Primitive mesh n³',ylabel='Relative covariance difference from B5')
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'phonon_covariance_convergence.png')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    phonons=[row for row in complete if row['settings']['grid']==config['selected_grid'] and row['settings']['samples']==config['selected_samples'] and row['settings']['padding_angstrom']==config['selected_padding_angstrom']]
    # The fixed electron base q mesh may have been reused from an electron case.
    if not phonons and complete:phonons=complete[-1:]
    for row in phonons:
        with np.load(row['artifact']) as data:
            for ax,name in zip(axes.flat,('full_0_20_mrad','central_0_5_mrad','annulus_5_10_mrad','annulus_10_20_mrad')):
                ax.plot(data['energy_grid_mev'],data[f'spectrum_{name}'],label=f"{row['settings']['mesh']}³")
                ax.set(title=name,xlabel='Energy loss (meV)',ylabel='Absolute inelastic weight / meV')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
    save(fig,'absolute_phonon_sampling_spectra.png')
    if complete:
        first=phonons[0] if phonons else complete[0];last=phonons[-1] if phonons else complete[-1]
        with np.load(first['artifact']) as data:a=data['momentum_histogram']
        with np.load(last['artifact']) as data:b=data['momentum_histogram']
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,value,label in zip(axes,(a,b,b-a),(f"q={first['settings']['mesh']}³, grid={first['settings']['grid']}²",f"q={last['settings']['mesh']}³, grid={last['settings']['grid']}²",'Fine − coarse')):
            image=ax.imshow(value.T,origin='lower',extent=(-20,20,-20,20),cmap='coolwarm' if '−' in label else 'viridis')
            ax.set(xlabel='qx/k (mrad)',ylabel='qy/k (mrad)',title=label)
            fig.colorbar(image,ax=ax,label='Absolute weight per fixed momentum bin')
        save(fig,'momentum_sampling_difference.png')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained')
    if complete:
        for category in sorted(set(row['category'] for row in complete)):
            chosen=[row for row in complete if row['category']==category]
            axes[0].plot([row['q_points'] for row in chosen],[row['elapsed_seconds'] for row in chosen],'o',label=category)
    axes[0].set(xlabel='Primitive q points',ylabel='Computed case wall time (s)');axes[0].grid(alpha=.2)
    if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=8)
    rows=results.get('energy_sampling_comparisons',[])
    if rows:
        axes[1].loglog([row['fine_step_mev'] for row in rows],[max(row['maximum_relative_L1'],1e-17) for row in rows],'o-')
        axes[1].axhline(.01,color='k',linestyle='--')
    axes[1].set(xlabel='Energy sampling ΔE (meV)',ylabel='Spectrum L1 change; Gaussian FWHM fixed');axes[1].grid(alpha=.2)
    save(fig,'computational_cost_and_energy_sampling.png')
    lines=['# Benchmark 7: controlled numerical convergence of Si phonon EELS','',f"Overall status: **{results['status']}**. {results['conclusion']}",'',
           '## Fixed physical calculation','',
           'The specimen positions, transverse physical cell, thickness, force constants, electron potential, 60 keV incident plane wave, temperature, detector masks, momentum bins, energy window, and 2 meV Gaussian FWHM are fixed within every numerical sequence. model_specification.json records their SI values and provenance. No absolute spectrum is normalized to unit area. No elastic zero-loss contribution enters the inelastic L1 denominator.',
           f"Specimen thickness={spec['specimen_thickness_m']*1e10:.8f} Å; transverse lengths={[x*1e10 for x in spec['transverse_lengths_m']]} Å. Force-constant supercell={spec['force_constant_model']['force_supercell']}; fitted range={spec['force_constant_model']['fitted_range_angstrom']} Å. This force supercell is separate from phonon q sampling. Material status={spec['force_constant_model']['status']}.",
           'The calculation remains harmonic, one-phonon, undepleted first-Born source propagation. There is no dynamical one-phonon source solver to merge with these results, and no actual coupled-channel occupation p cutoff. No p sweep is claimed to establish multiphonon accuracy. Multiphonon, repeated-inelastic, and elastic dynamical scattering remain unassessed.','',
           '## Phonon phase, normalization, and boundaries','',
           'The two-atom primitive representation is an exact export of the validated hiPhive model. Its vectors, reciprocal rows 2πA^-T, basis coordinates, and oriented-cell folding comparison are saved. Uniform primitive n³ reciprocal meshes use every q point and all six branches; no crystal-symmetry multiplicities are used. Only the three explicitly identified Gamma translations are removed.',
           'Primitive mass-weighted eigenvectors are normalized over the two basis atoms. The displacement at an actual fixed-specimen atom is sqrt[hbar/(2Mi omega)] e_basis(i)(q) exp(iq·Ri)/sqrt(Nq). The 1/sqrt(Nq) factor appears exactly once; summing squared amplitudes is the normalized BZ quadrature. Eigenvectors use the Phonopy atomic gauge. Arbitrary phonon q phases are not inserted into a periodic electron box: each outgoing channel has explicit transverse Bloch shift −q_phonon modulo an electron reciprocal vector, with the corresponding kinetic multiplier and covariant gradient stencil.',
           'The atomic list and z thickness never grow when the phonon mesh is refined. The transverse electron physical cell is also fixed, while electron grid count changes only numerical spacing. This is bulk BZ quadrature with explicit Bloch channel boundaries, not a phonon-supercell enlargement masquerading as a sampling test.',
           'The independently converged Benchmark 5 covariance is held fixed in the electron and phonon-transition quadrature sequences. Primitive-mesh covariance tensors are separately compared with that input, including physical low-q acoustic modes. Thus a covariance error cannot cancel a transition-sum sampling error without being visible.','',
           '## Actual computed configurations','',
           '| Stage | Primitive mesh | Electron grid | Δx (Å) | Propagation step (Å) | Padding (Å) | q completed | Physical modes | Wall time (s) | Status |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in results['cases']:
        settings=row['settings'];length=spec['specimen_thickness_m']*1e10+2*settings['padding_angstrom']
        lines.append(f"| {row['category']} | {settings['mesh']}³ | {settings['grid']}² | {spec['transverse_lengths_m'][0]*1e10/settings['grid']:.6f} | {2*length/settings['samples']:.6f} | {settings['padding_angstrom']} | {row.get('completed_q',0)} | {row.get('physical_mode_count','incomplete')} | {row.get('elapsed_seconds',0):.3f} | {row['status']} |")
    lines.extend(['','## Measured errors and worst observables','',
                  'Spectrum error is integral_W |I_fine−I_coarse| / integral_W I_fine on the fixed signed broadened-energy window [−10,100] meV. Detector/peak-group areas and all fixed 8×8 momentum-bin weights are independently compared. Every reported maximum names its observable. Near-zero quantities retain both absolute and relative changes, with the fixed weak-signal scale; failing observables are never removed.',
                  'Two successive refinement comparisons are required by this configuration. Peak positions are compared in fixed energy groups, with a 0.2 meV target (one tenth FWHM). Missing significant peaks are explicitly flagged. Increasing broadening is not used as a convergence control.','',
                  '| Sequence | Fine settings | Max spectrum L1 | Worst detector | Max observable change | Worst observable | Its absolute change | Max peak shift (meV) | Pass |','|---|---|---:|---|---:|---|---:|---:|---|'])
    for category,rows in results['comparisons'].items():
        for row in rows:
            worst=next(value for value in row['observable_changes']['rows'] if value['observable']==row['observable_changes']['worst_observable'])
            lines.append(f"| {category} | {row['fine_settings']} | {row['spectral_changes']['maximum_relative_L1']:.6e} | {row['spectral_changes']['worst_detector']} | {row['observable_changes']['maximum_acceptance_error']:.6e} | {row['observable_changes']['worst_observable']} | {worst['absolute_change']:.6e} | {row['maximum_peak_shift_mev']:.6f} | {row['passed']} |")
    for name in ('joint_at_fixed_phonon_sampling','joint_electron_and_phonon_comparison'):
        if name in results:
            row=results[name];worst=next(value for value in row['observable_changes']['rows'] if value['observable']==row['observable_changes']['worst_observable'])
            lines.append(f"| {name} | {row['fine_settings']} | {row['spectral_changes']['maximum_relative_L1']:.6e} | {row['spectral_changes']['worst_detector']} | {row['observable_changes']['maximum_acceptance_error']:.6e} | {row['observable_changes']['worst_observable']} | {worst['absolute_change']:.6e} | {row['maximum_peak_shift_mev']:.6f} | {row['passed']} |")
    lines.extend(['','The exact-forward contribution is the discrete quadrature weight at an exact point, not a finite-area detector signal or continuum momentum density. Its relative change can dominate despite a negligible absolute weight. It remains in the predeclared error tables and gates; its absolute scale is reported alongside its relative change. Finite-area central-detector spectral convergence is checked separately.',
                  'Joint refinement changes electron spacing, longitudinal integration, padding, and phonon sampling together relative to the preceding q mesh. A separate comparison at fixed finest phonon mesh checks the electron change alone. Per-observable signed increments and cancellation ratios are retained; an accidentally small joint difference cannot override a failing independent sequence. No unsupported asymptotic extrapolation is used.','',
                  '## Separate statuses and limits','','| Category | Status |','|---|---|'])
    for name,status in results['category_statuses'].items():lines.append(f'| {name} | {status} |')
    lines.extend(['',f"Accepted settings: {results['accepted_settings']}. Best computed settings: {results['best_computed_settings']}. A best computed setting is not labelled accepted when convergence is incomplete.",
                  f"Recommended electron spacings, next sampling test, and timing estimate: {results.get('recommendations','computed in the next runner invocation')}",
                  f"Resource costs/limits: {results['resource_costs']}. Stop reason: {results['resource_stop']}. Each q sector is a small six-channel source problem; memory/runtime are estimated before work and resident memory is checked. A dense millions-channel allocation is never made.",
                  'The prefix-scan source implementation is algebraically the same tested free-propagation/Simpson recurrence as the ordinary longitudinal stepper. It does not replace propagation with analytical Born amplitudes; independent Born values are retained only as numerical-accuracy controls.',
                  'Per-q energies, detector weights, momentum-bin weights, and reference errors are stored in memory-mapped checkpoint arrays. state.json advances atomically only after arrays are flushed. A stopped case resumes from the last recorded q sector, without renormalizing the unfinished quadrature. Cache identities include physical inputs, model checksums, constants, numerical settings, and calculation source versions. Changing a runtime/refinement limit preserves reusable work. Reporting-only manifest corrections reuse the frozen identity only after the complete scientific-input contract is verified; original and current manifests are archived separately in provenance/.',
                  f"Thermal extension: {results['thermal_300K_extension']}. Earlier statuses preserved: {results['prior_benchmark_statuses_preserved']}. Benchmark 2 and Benchmark 3B are not resolved by these calculations.",'',
                  '```bash','cd /home/zuxian/Documents/PhD/Yoshioka','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.si_eels_convergence --config benchmarks/configs/convergence.json',
                  '# Same command resumes compatible cached work; resource limits are configurable:',
                  'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.si_eels_convergence --config benchmarks/configs/convergence.json --runtime-seconds 1800 --memory-gib 2 --max-refinements 30',
                  '.venv/bin/python -m unittest discover -s tests -v','```','',
                  'Numerical convergence within this approximation is separate from physical accuracy of MOD or Lobato and from full material EELS validation. All remaining errors and unfinished categories remain in results.json and manifest.json.',''])
    for name in figures:lines.extend([f'![{name}]({name})',''])
    (output/'report.md').write_text('\n'.join(lines))
    import resource
    costs=results['resource_costs']
    costs['output_generation_seconds']=time.monotonic()-output_started
    costs['total_elapsed_seconds_including_outputs']=costs['elapsed_seconds']+costs['output_generation_seconds']
    costs['peak_process_rss_mib_including_outputs']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    with (output/'report.md').open('a') as stream:
        stream.write(f"\nFigure/report generation: {costs['output_generation_seconds']:.3f} s; total invocation time including outputs: {costs['total_elapsed_seconds_including_outputs']:.3f} s; peak RSS including outputs: {costs['peak_process_rss_mib_including_outputs']:.3f} MiB.\n")
    return figures
