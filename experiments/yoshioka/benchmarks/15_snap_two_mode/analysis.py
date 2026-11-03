"""Auditable numerical comparisons, raw TRPMD reduction, PNGs and pilot report."""
import csv
import html
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy import constants
from scipy.ndimage import gaussian_filter1d
from yoshioka.provenance import sha256,json_write
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.finite_snap_bath import finite_optical_bath
from yoshioka.spectral_data import inspect_numeric_array,conservative_rebin,gaussian_broaden_bin_weights

REFERENCE=Path('/home/zuxian/Documents/PhD/HiPhive_TRPMD')
BASE='p2_N33_F769_S769_D32'
DETECTORS=['0–20','0–5','5–10','10–20']


def compare_phonons(out):
    selected=out/'phonons/fc_4x6x6_d0.0050'
    model=RealSpaceForceConstants.load(selected/'model.npz')
    bath,evidence=finite_optical_bath(model)
    rng=np.random.default_rng(4181)
    q=np.vstack(([0,0,0],rng.uniform(-.5,.5,(30,3)),[.5,0,0],[0,.5,0],[0,0,.5]))
    reference=model.modes(q)[0]*constants.hbar/(1e-3*constants.e)
    rows=[]
    for folder in sorted((out/'phonons').glob('fc_*')):
        if not (folder/'result.json').exists():continue
        data=json.loads((folder/'result.json').read_text())
        candidate=RealSpaceForceConstants.load(folder/'model.npz')
        cb,ce=finite_optical_bath(candidate)
        frequencies=candidate.modes(q)[0]*constants.hbar/(1e-3*constants.e)
        # All non-Gamma sampled modes and Gamma optical modes; rigid translations
        # are identified independently by overlap below, not clipped to zero.
        delta=abs(frequencies-reference);delta[0,:3]=0
        row={'case':folder.name,'maximum_sampled_frequency_change_mev':float(delta.max()),
             'active_quanta_mev':constants.hbar*cb.omega_rad_s/(1e-3*constants.e),
             'active_frequency_change_mev':float(np.max(abs(cb.omega_rad_s-bath.omega_rad_s))*constants.hbar/(1e-3*constants.e)),
             'covariance_relative_change':float(np.linalg.norm(ce['full_covariance_m2']-evidence['full_covariance_m2'])/np.linalg.norm(evidence['full_covariance_m2'])),
             'relative_correction':data['correction_relative_norm'],**data['range']}
        with np.load(folder/'commensurate_modes.npz') as a:
            freq=a['omega_signed_rad_s'].copy();mesh_q=a['q_fractional']
        gw,gv=candidate.modes([0,0,0])
        translation=np.tile(np.eye(3),(4,1));translation/=np.linalg.norm(translation,axis=0)
        rigid=np.argsort(np.sum(abs(gv.conj().T@translation)**2,axis=1))[-3:]
        keep=np.ones(freq.shape,bool);keep[0,rigid]=False
        row['gamma_rigid_indices']=rigid.tolist()
        row['signed_rigid_mev']=(gw[rigid]*constants.hbar/(1e-3*constants.e)).tolist()
        row['minimum_physical_commensurate_mev']=float(freq[keep].min()*constants.hbar/(1e-3*constants.e))
        row['passed']=(row['maximum_sampled_frequency_change_mev']<.05 and row['covariance_relative_change']<.005 and row['relative_correction']<.001 and row['minimum_physical_commensurate_mev']>0)
        rows.append(row)
    np.savez(out/'checks/fc_comparison.npz',q_fractional=q,reference_energies_mev=reference)
    json_write(out/'checks/fc_comparison.json',{'reference':selected.name,'rows':rows,'passed':len(rows)==4 and all(row['passed'] for row in rows),'scope':'Finite-displacement and two-cell comparison, sampled stability; no continuum stability proof'})
    return rows


def inspect_reference(out):
    folder=out/'reference';file=REFERENCE/'Si_5x7x52_10K_beads165_SNAP.npy'
    if (folder/'processing.json').exists():return json.loads((folder/'processing.json').read_text())
    data,info=inspect_numeric_array(file)
    if info['sha256']!='46f97e1d7c651b9367d6e34c34adbf44f8d05dc67a64db3902375e714eea41b0':raise ValueError('reference hash differs from supplied brief')
    if data.shape!=(600,267,266) or data.dtype!=np.float64:raise ValueError('reference shape/type mismatch')
    yy,xx=np.ogrid[:267,:266];mask=(xx-133)**2+(yy-133)**2<=20**2
    raw=np.empty(600);all_momentum=np.empty(600)
    for i in range(600):
        raw[i]=np.sum(data[i][mask]);all_momentum[i]=np.sum(data[i])
    energy=np.fft.fftshift(np.fft.fftfreq(600,d=.0075))*4.13566769692386
    spacing=float(energy[1]-energy[0]);edges=np.r_[energy-spacing/2,energy[-1]+spacing/2]
    axis=np.linspace(-180,180,7201);step=.05;target=np.r_[axis-step/2,axis[-1]+step/2]
    rebinned=conservative_rebin(raw,edges,target)
    broadened,reach=gaussian_broaden_bin_weights(rebinned,step,2.)
    cropped=broadened[reach:reach+len(axis)]
    normalized=raw/all_momentum.sum()
    notebook_smoothed=gaussian_filter1d(normalized,1/spacing,mode='reflect')
    np.savez_compressed(folder/'reduction_and_transformations.npz',native_index=np.arange(600),native_energy_mev=energy,
        native_edges_mev=edges,pixel_mask=mask,raw_pixel_roi=raw,raw_all_momentum=all_momentum,
        notebook_normalization_denominator=all_momentum.sum(),notebook_normalized=normalized,
        notebook_sigma1mev_reflect=notebook_smoothed,target_energy_mev=axis,target_edges_mev=target,
        conservatively_rebinned_weights=rebinned,forward_2mev_full_weights=broadened,
        forward_extended_axis_mev=(np.arange(len(broadened))-reach)*step+axis[0],
        forward_2mev_cropped_weights=cropped,forward_2mev_density=cropped/step)
    np.savetxt(folder/'native_reduction.csv',np.column_stack((np.arange(600),energy,raw,all_momentum,normalized,notebook_smoothed)),delimiter=',',comments='',header='array_index,conditional_energy_mev,raw_pixel_ROI_weight,raw_all_momentum_weight,notebook_normalized,notebook_sigma1mev_reflect')
    np.savetxt(folder/'comparison_curve.csv',np.column_stack((axis,rebinned,cropped,cropped/step)),delimiter=',',comments='',header='conditional_energy_mev,conservatively_rebinned_weight,forward_2mev_weight,forward_2mev_density')
    notebook=REFERENCE/'plot_paper_10K_diffbeads.ipynb'
    nearby=REFERENCE/'plot_paper_TRPMD_MD.ipynb'
    config=REFERENCE/'eels-studio-config.json'
    settings=json.loads(config.read_text())['settings']
    provenance={
        'array_identity':{'status':'VERIFIED','value':info},
        'array_axes':{'status':'CONDITIONAL_ANALYSIS_ASSUMPTION','value':'(energy,y,x), FFT-centered energy','evidence':str(notebook)},
        'sampling':{'status':'CONDITIONAL_ANALYSIS_ASSUMPTION','value':'2.5 fs timestep times stride 3 = 7.5 fs; 600 samples','evidence':'notebook code and studio settings agree; no generation manifest'},
        'energy_sign':{'status':'UNVERIFIED','value':'adopt notebook positive-loss sign for qualitative overlay; exact generation Fourier sign unavailable'},
        'detector':{'status':'VERIFIED_ANALYSIS_CONVENTION','value':'circle (x,y)=(133,133), radius 20 pixels; inclusive pixel boundary','pixels':int(mask.sum())},
        'pixel_to_angle':{'status':'UNVERIFIED','value':'20 pixels is not identified with 20 mrad; no calibrated axis tied to the exact array'},
        'geometry':{'status':'FILENAME_ONLY_FOR_THIS_ARRAY','value':'5x7x52; 165 beads. Nearby Si_110_5x7x52.lmp is not linked by a generation manifest'},
        'incident_normalization':{'status':'UNVERIFIED','value':'notebook normalizes by full energy/momentum sum; not proof of incident-electron normalization'},
        'prior_quantum_correction':{'status':'UNVERIFIED_GENERATION','value':'adjacent analysis notebook states saved spectra already include Kubo correction; no exact generator found. apply_kubo=False is verified in the requested notebook. No correction applied here.'},
        'native_window_and_resolution':{'status':'UNVERIFIED','value':'nearby unrelated filenames mention Tukey windows; not assigned to this exact array'},
        'display_broadening':{'status':'VERIFIED_THIS_ANALYSIS','value':'additional Gaussian 2 meV FWHM after conservative rebinning; unknown native resolution remains. Separate exact notebook reconstruction uses sigma=1 meV, reflect mode.'}}
    # Save source cells instead of copying binary notebook output or changing it.
    excerpts={}
    for p in (notebook,nearby):
        obj=json.loads(p.read_text())
        excerpts[p.name]=[{'cell':i,'type':c['cell_type'],'source':''.join(c.get('source',[]))} for i,c in enumerate(obj['cells']) if any(t in ''.join(c.get('source',[])) for t in ('Kubo','apply_kubo','timestep_fs','fftfreq','aperture_center'))]
    json_write(folder/'analysis_evidence.json',{'hashes':{str(p):sha256(p) for p in (notebook,nearby,config)},'excerpts':excerpts,'studio_selected_settings':{k:settings.get(k) for k in ('spectrum_timestep','spectrum_stride','detector_radius','spectrum_sigma','broaden_spectrum','normalize_probe','angle_source')},'search_scope':['HiPhive_TRPMD notebooks and config','research-script sources','nearby Si_LS and STEM_EELS notebook source cells'],'exact_generation_manifest_found':False})
    candidate=Path('/home/zuxian/Documents/PhD/STEM_EELS/PlotRegions/Si_110_5x7x52.lmp')
    if candidate.exists():
        from ase.io import read
        atoms=read(candidate,format='lammps-data',atom_style='atomic')
        provenance['nearby_geometry_candidate']={'status':'VERIFIED_FILE_NOT_LINKED_TO_ARRAY','path':str(candidate),'sha256':sha256(candidate),'atoms':len(atoms),'cell_angstrom':np.asarray(atoms.cell).tolist()}
    json_write(folder/'provenance.json',provenance)
    record={'array':info,'native_energy_range_mev':[float(energy[0]),float(energy[-1])], 'native_spacing_mev':spacing,
        'native_roi_signed_mass':float(raw.sum()),'native_all_momentum_signed_mass':float(all_momentum.sum()),
        'native_negative_roi_bin_count':int(np.sum(raw<0)),'native_negative_roi_mass':float(raw[raw<0].sum()),
        'rebinned_window_mass':float(rebinned.sum()),'cropped_native_mass':float(raw.sum()-rebinned.sum()),
        'full_forward_broadened_mass':float(broadened.sum()),'display_after_forward_mass':float(cropped.sum()),
        'forward_cropped_mass':float(broadened.sum()-cropped.sum()),
        'gaussian_full_mass_relative_error':float(abs(broadened.sum()-rebinned.sum())/abs(rebinned.sum())),
        'source_bin_semantics':'Discrete native array reductions treated as bin weights; piecewise constant within each assumed energy bin',
        'energy_sign_status':'CONDITIONAL; generation sign unverified','reference_acceptance':'20-pixel ROI, not angular calibrated',
        'transform_sequence':['read-only raw array','sum raw pixels in documented circular ROI','conservative rebin to 0.05 meV bins','forward Gaussian 2 meV FWHM with zero padding','retain full broadened result and cropped display','one optional total-curve area factor on 5–160 meV'],
        'kubo_applied':False,'background_subtracted':False,'energy_shift_mev':0.,'frequency_fitted':False}
    json_write(folder/'processing.json',record)
    return record


def case_comparisons(out):
    folders={p.name:p for p in (out/'cases').iterdir() if (p/'exit.npz').exists()}
    if BASE not in folders:return [],folders
    with np.load(folders[BASE]/'exit.npz') as a:
        base={k:a[k] for k in ('exit_coefficients','basis','qxy_m_inv')}
    with np.load(folders[BASE]/'spectra.npz') as a: base_prob=a['probability'][0]
    rows=[]
    for name,folder in sorted(folders.items()):
        with np.load(folder/'exit.npz') as a:
            wave=a['exit_coefficients'][0];basis=a['basis']
        with np.load(folder/'spectra.npz') as a: prob=a['probability'][0]
        indices=[list(map(tuple,basis)).index(tuple(v)) for v in base['basis']]
        reduced=prob[indices]
        difference=reduced-base_prob
        tolerance=1e-10+.01*abs(base_prob)
        amp=wave[indices];n=amp.shape[-1];b=base['exit_coefficients'][0]
        if n>=b.shape[-1]:
            off=(n-b.shape[-1])//2;amp=amp[:,off:off+b.shape[-1],off:off+b.shape[-1]]
            amplitude=float(np.linalg.norm(amp-b)/np.linalg.norm(b))
        else: amplitude=None
        all_angle=np.sum(abs(wave)**2,axis=(1,2))
        # Executed initial state is the unrenormalized optical vacuum; its tiny
        # omitted mass is retained in the exit file and the thermal manifest.
        with np.load(folder/'exit.npz') as z:
            all_angle*=z['executed_initial_weights'][0]
            masks=z['detector_masks']
        acceptances=np.column_stack((all_angle,prob))
        if not (folder/'channel_probabilities_all_acceptances.csv').exists():
            np.savetxt(folder/'channel_probabilities_all_acceptances.csv',np.column_stack((basis,acceptances)),delimiter=',',comments='',header='final_a,final_b,all_retained_beams,0_20_mrad,0_5_mrad,5_10_mrad,10_20_mrad')
        sectors=np.array([acceptances[basis.sum(axis=1)==i].sum(axis=0) for i in range(int(basis.sum(axis=1).max())+1)])
        if not (folder/'sector_probabilities.npz').exists():
            with np.load(folder/'spectra.npz') as z:
                two_spectrum=z['components_per_mev'][:,basis.sum(axis=1)==2].sum(axis=(0,1))
                axis=z['energy_mev']
            np.savez(folder/'sector_probabilities.npz',sector_occupations=np.arange(len(sectors)),acceptances=['all retained beams',*DETECTORS],sector_absolute_weights=sectors,two_phonon_density_per_mev=two_spectrum,energy_mev=axis)
            np.savetxt(folder/'sector_probabilities.csv',np.column_stack((np.arange(len(sectors)),sectors)),delimiter=',',comments='',header='total_occupation,all_retained_beams,0_20_mrad,0_5_mrad,5_10_mrad,10_20_mrad')
            np.savetxt(folder/'two_phonon_spectrum.csv',np.column_stack((axis,two_spectrum.T)),delimiter=',',comments='',header='energy_mev,0_20_mrad_per_mev,0_5_mrad_per_mev,5_10_mrad_per_mev,10_20_mrad_per_mev')
        quantum=np.array(json.loads((out/'phonons/bath.json').read_text())['quanta_mev'])*1e-3*constants.e
        ratios=np.exp(-quantum/(10*constants.k));cutoff=int(basis.sum(axis=1).max())
        thermal_tail=float(ratios[0]**(cutoff+1)+(1-ratios[0])*sum(ratios[0]**j*ratios[1]**(cutoff-j+1) for j in range(cutoff+1)))
        json_write(folder/'probability_audit.json',{'thermal_probability_outside_target_basis':thermal_tail,
                   'log_exact_vacuum_weight':float(np.log1p(-ratios).sum()),'all_channel_sum':float(all_angle.sum()),'annular_detector_partition_exact':bool(np.array_equal(masks[0],masks[1:].sum(axis=0))),
                   'channel_detector_partition_error':float(np.max(abs(prob[:,0]-prob[:,1:].sum(axis=1)))),
                   'two_phonon_sector_0_20':float(sectors[2,1]),'fractions_denominator':'inclusive probability in the specified detector; all absolute weights retained'})
        record={'case':name,'completed':True,'max_channel_detector_absolute_difference':float(abs(difference).max()),
                'max_channel_detector_error_over_tolerance':float(np.max(abs(difference)/tolerance)),
                'channel_detector_pass':bool(np.all(abs(difference)<=tolerance)),
                'complex_exit_relative_difference_on_common_support':amplitude,
                'complex_exit_pass':amplitude is not None and amplitude<=1e-4,
                'extra_final_sector_probability_all_angles':float(np.sum(abs(wave[[i for i in range(len(basis)) if i not in indices]])**2)),
                'two_phonon_0_20':float(prob[np.sum(basis,axis=1)==2,0].sum()),
                'changed_0_20':float(prob[np.sum(basis,axis=1)>0,0].sum())}
        np.savez(folder/'difference_from_pilot.npz',basis=base['basis'],absolute_channel_detector_difference=difference,tolerance=tolerance)
        rows.append(record)
    support=[]
    for coarse,fine in ((33,65),(65,97)):
        a=f'p2_N{coarse}_F769_S769_D32';b=f'p2_N{fine}_F769_S769_D32'
        if a not in folders or b not in folders:continue
        with np.load(folders[a]/'spectra.npz') as z:pa=z['probability'][0]
        with np.load(folders[b]/'spectra.npz') as z:pb=z['probability'][0]
        with np.load(folders[a]/'exit.npz') as z:wa=z['exit_coefficients'][0]
        with np.load(folders[b]/'exit.npz') as z:wb=z['exit_coefficients'][0]
        offset=(fine-coarse)//2
        amp=float(np.linalg.norm(wb[:,offset:offset+coarse,offset:offset+coarse]-wa)/np.linalg.norm(wa))
        ratio=float(np.max(abs(pb-pa)/(1e-10+.01*abs(pa))))
        support.append({'coarse':a,'fine':b,'max_channel_detector_error_over_tolerance':ratio,
                        'complex_exit_relative_difference':amp,'passed':ratio<=1 and amp<=1e-4})
    json_write(out/'checks/adjacent_support_comparisons.json',support)
    json_write(out/'checks/case_comparisons.json',{'reference':BASE,'rows':rows,'criterion':'each final channel/detector: |delta P| <= 1e-10 + 0.01 P_reference; common complex exit relative error <= 1e-4; no fitted phase'})
    return rows,folders


def figures(out,folders,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    with np.load(folders[BASE]/'spectra.npz') as a:s={k:a[k] for k in a.files}
    with np.load(folders[BASE]/'exit.npz') as a:basis=a['basis']
    energy=s['energy_mev'];components=s['components_per_mev'][0];probs=s['probability'][0]
    labels=[f'({a},{b})' for a,b in basis]
    colors=plt.get_cmap('tab10').colors
    def save(fig,name):
        fig.savefig(out/'figures'/name,dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    for j,label in enumerate(labels):axs[0].plot(energy,components[j,0],label=label,color=colors[j])
    axs[0].plot(energy,s['inclusive_per_mev'][0],color='black',ls='--',label='inclusive total')
    axs[0].set_yscale('symlog',linthresh=1e-12);axs[0].set(xlim=(-5,140),ylabel='Probability / incident electron / meV',title='SNAP two-mode p=2 pilot • 10 K • 1.5483 nm • 0–20 mrad')
    axs[0].legend(ncol=4,fontsize=9)
    for j in range(1,len(labels)):axs[1].plot(energy,components[j,0],label=labels[j],color=colors[j])
    axs[1].plot(energy,s['changed_per_mev'][0],color='black',ls='--',label='changed-state total')
    axs[1].set(xlim=(45,130),xlabel='Energy loss (meV)',ylabel='Absolute probability / meV',title='Changed final states; 2 meV FWHM is imposed display broadening')
    axs[1].legend(ncol=3);save(fig,'channel_spectra.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axs[0].bar(labels,probs[:,0],color=colors[:6]);axs[0].set_yscale('log');axs[0].set(ylabel='Absolute probability / incident electron',title='All six channels • 0–20 mrad')
    two=np.flatnonzero(basis.sum(axis=1)==2);vals=np.r_[probs[two,0],probs[two,0].sum()]
    axs[1].bar([labels[j] for j in two]+['sum n=2'],vals,color=[colors[j] for j in two]+['#333333'])
    axs[1].set(ylabel='Absolute probability / incident electron',title='Two-phonon sector; fixed lab x/y basis')
    axs[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));save(fig,'channel_weights.png')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for d,label in enumerate(DETECTORS):ax.plot(energy,s['changed_per_mev'][d],label=label+' mrad')
    ax.set_yscale('symlog',linthresh=1e-13);ax.set(xlim=(40,135),xlabel='Energy loss (meV)',ylabel='Changed-state probability / meV',title='Absolute detector-resolved spectra; no detector renormalization');ax.legend();save(fig,'detectors.png')
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    shown=[r for r in comparisons if r['case']!=BASE]
    for r in shown:
        with np.load(folders[r['case']]/'difference_from_pilot.npz') as a:delta=a['absolute_channel_detector_difference'][:,0]
        axs[0].plot(labels[1:],delta[1:],'.-',label=r['case'].replace('_D32',''))
        axs[1].barh(r['case'].replace('_D32',''),r['max_channel_detector_error_over_tolerance'])
    axs[0].set_yscale('symlog',linthresh=1e-12);axs[0].set(ylabel='Delta absolute probability (0–20 mrad)',title='Refinements minus p2 N33 F769 S769');axs[0].legend(fontsize=7)
    axs[1].axvline(1,color='red',ls='--');axs[1].set_xscale('symlog',linthresh=.01);axs[1].set(xlabel='Largest channel/detector error ÷ frozen tolerance',title='Values above 1 fail the comparison');save(fig,'refinement_differences.png')
    with np.load(out/'reference/reduction_and_transformations.npz') as a:r={k:a[k] for k in a.files}
    overlap=np.maximum(0,np.minimum(energy+.025,160)-np.maximum(energy-.025,5))
    model_area=float(np.sum(s['inclusive_per_mev'][0]*overlap))
    ref_area=float(np.sum(r['forward_2mev_density']*overlap))
    factors={'positive_interval_mev':[5,160],'model_unscaled_area':model_area,'reference_unscaled_area':ref_area,
             'model_factor':1/model_area,'reference_factor':1/ref_area,
             'one_factor_per_total_curve':True,'model_components_use_total_factor':True,'energy_shift_mev':0.}
    json_write(out/'reference/display_factors.json',factors)
    scaled_model=s['inclusive_per_mev'][0]/model_area;scaled_ref=r['forward_2mev_density']/ref_area
    np.savez(out/'reference/display_overlay.npz',energy_mev=energy,model_total=scaled_model,
             model_components=components[:,0]/model_area,reference_total=scaled_ref,model_factor=1/model_area,reference_factor=1/ref_area)
    np.savetxt(out/'reference/display_overlay.csv',np.column_stack((energy,scaled_model,scaled_ref)),delimiter=',',comments='',header='energy_mev,model_area_normalized,conditional_reference_area_normalized')
    fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    shown_energy=(energy>=5)&(energy<=160)
    for ax in axs:
        ax.plot(energy[shown_energy],scaled_model[shown_energy],color='#163f5f',label='Yoshioka: 4 repeats, two Gamma modes, 0–20 mrad')
        ax.plot(energy[shown_energy],scaled_ref[shown_energy],color='#bf562c',label='TRPMD: filename 52 repeats, 165 beads, radius 20 pixels')
        for j in range(1,6):ax.plot(energy[shown_energy],components[j,0,shown_energy]/model_area,color=colors[j],alpha=.45,lw=.9)
        ax.set(xlim=(5,160),ylabel='Display-only density (area 5–160 meV = 1)')
        ax.set_ylim(bottom=min(0.,float(scaled_model[shown_energy].min()),float(scaled_ref[shown_energy].min()))*1.1)
    axs[0].set(title='Qualitative shape comparison • mismatched model, thickness and acceptance');axs[0].legend(fontsize=9)
    axs[1].set_yscale('symlog',linthresh=1e-7);axs[1].set(xlabel='Energy loss (meV); reference sign/sampling conditional',title='No energy alignment, background subtraction, peak fitting or Kubo re-correction')
    fig.text(.5,-.015,'Model: imposed 2 meV FWHM. Reference: additional 2 meV FWHM; native window/resolution unknown. Pilot numerically unconverged.',ha='center',fontsize=9)
    save(fig,'qualitative_TRPMD_comparison.png')
    fig,axs=plt.subplots(3,1,figsize=(11,9),layout='constrained')
    axs[0].plot(energy,s['inclusive_per_mev'][0]);axs[0].set_yscale('symlog',linthresh=1e-12);axs[0].set(xlim=(-180,180),ylabel='Probability / electron / meV',title='Yoshioka absolute spectrum (unscaled)')
    axs[1].plot(r['native_energy_mev'],r['raw_pixel_roi'],label='raw native bins');axs[1].set_yscale('symlog',linthresh=max(abs(r['raw_pixel_roi']).max()*1e-8,1e-30));axs[1].set(ylabel='Reference raw array units / bin',xlabel='Conditional notebook energy axis (meV)',title='Read-only reference ROI: separate units; no intensity matching')
    axs[2].plot(r['native_index'],r['raw_pixel_roi']);axs[2].set_yscale('symlog',linthresh=max(abs(r['raw_pixel_roi']).max()*1e-8,1e-30));axs[2].set(xlabel='Native energy-array index',ylabel='Reference raw array units / bin',title='Axis-independent reference view: radius 20 pixels at (133,133)')
    save(fig,'unscaled_and_native_reference.png')
    return factors


def summarize_results(out):
    fc=compare_phonons(out)
    reference=inspect_reference(out)
    rows,folders=case_comparisons(out)
    if BASE not in folders:raise ValueError('completed pilot required for spectra report')
    factors=figures(out,folders,rows)
    write_report(out,rows,fc,reference,factors)


def write_report(out,rows,fc,reference,factors):
    bath=json.loads((out/'phonons/bath.json').read_text())
    cfg=json.loads((out/'inputs/config.json').read_text())
    eq=json.loads((out/'phonons/equilibrium.json').read_text())
    with np.load(out/'cases'/BASE/'spectra.npz') as a:prob=a['probability'][0];loss=a['energy_loss_mev'][0]
    with np.load(out/'cases'/BASE/'exit.npz') as a:basis=a['basis']
    manifests={r['case']:json.loads((out/'cases'/r['case']/'manifest.json').read_text()) for r in rows}
    execution={}
    for log in sorted((out/'logs').glob('*_propagate.json')):
        entry=json.loads(log.read_text());a=entry['arguments'];tag=a.get('case_tag','')
        name=f"p{a['p']}_N{a['n']}_F{a['fields']}_S{a['steps']}_D{a['dimension']}"+('_'+tag if tag else '')
        info=execution.setdefault(name,{'attempts':0,'total_invocation_seconds':0.,'maximum_recorded_rss_mib':0.,'logs':[]})
        info['attempts']+=1;info['logs'].append(str(log.relative_to(out)))
        if 'resources' in entry:
            info['total_invocation_seconds']+=entry['resources']['elapsed_seconds']
            info['maximum_recorded_rss_mib']=max(info['maximum_recorded_rss_mib'],entry['resources']['peak_process_rss_mib'])
    json_write(out/'execution_summary.json',{'cases':execution,'completed_cases':len(rows),'production_count':0,'bulk_status':'NOT_ACCEPTED'})
    core=manifests[BASE]['source']
    lines=['# SNAP Si [110], 10 K: finite two-mode p=2 pilot','',
      '**Status: executed pilot, NUMERICALLY_UNCONVERGED.** This is a finite 16-atom optical-bath calculation, not a bulk material acceptance result.',
      '',f"Model: `{cfg['model_id']}`. Exact pilot source revision: `{core['revision']}`; numerical source digest `{manifests[BASE]['contract']['source_sha256']}`. Source snapshots and dependency manifests are retained under `provenance/source/`.",
      '', '## Executed physical model and parameters','',
      'The explicit force interface follows the LAMMPS SNAP coefficient/parameter/element mapping described in https://docs.lammps.org/pair_snap.html; installed LAMMPS version 2025-07-22 is recorded in the execution manifest.', '',
      '| Parameter | Executed value |','|---|---|',
      '| Nuclear force model | LAMMPS `pair_style snap`, metal units, Si mapping; supplied files unchanged |',
      '| Static cell | '+', '.join(f'{v:.10f}' for v in np.diag(eq['cell_angstrom']))+' Å; relaxed basis |',
      '| Nuclear mass/statistics | 28.085 u uniformly; quantum harmonic statistics at 10 K including zero-point motion |',
      '| FC cells/displacements | 4×6×6 (576 atoms), ±0.0025/0.005/0.01 Å; 5×7×7 (980 atoms), ±0.005 Å |',
      '| Specimen/bath | 1×1×4 oriented cells, 16 atoms; periodic nuclear bath with 45 nonrigid modes, 3 rigid translations omitted |',
      f"| Specimen thickness | {bath['specimen_thickness_nm']:.10f} nm; 1.2 nm vacuum on each z side |",
      '| Active modes | Fresh SNAP Gamma optical modes; fixed lab x/y, real, specimen-normalized; no active u_z |',
      '| Active quanta | '+', '.join(f'{v:.9f}' for v in bath['quanta_mev'])+' meV |',
      '| Electron | 60 keV, unit-probability plane wave along [110]; periodic transverse unit-cell rectangle |',
      '| Pilot basis | (0,0), (1,0), (0,1), (2,0), (1,1), (0,2); total occupation p=2 |',
      '| Pilot grids | N33 reciprocal rectangle, M65 auxiliary FFT; field 769, propagation 769 independently controlled |',
      '| Oscillator construction | D32 larger-oscillator exponential projected afterward; independent D48 and GH24/GH32 controls |',
      '| Output | −180 to +180 meV, 0.05 meV bins; Gaussian 2 meV FWHM, imposed display/instrument width |',
      '| Detectors | 0–20, 0–5, 5–10, 10–20 mrad from physical electron momentum axes; absolute discrete beam weights |',
      '| Resources | One scientific worker, one BLAS/FFT/OpenMP thread, 4 GiB RSS, ≤600 s per invocation; checkpoints every 16 steps |','',
      'Static zero-pressure volume is an approximation at 10 K: neither thermal expansion nor zero-point volume relaxation is included. The finite periodic bath is intentionally distinct from a bulk Brillouin-zone thermal covariance. All 45 finite modes supply the local covariance, the two active contributions are subtracted with the same quantum weights, and inactive averaging is applied once. Active coordinates remain explicit; no additional active Debye–Waller factor is applied.',
      '',f"The omitted excited-initial-state probability is **{bath['omitted_initial_probability']:.9e}**. The vacuum harmonic weight rounds to 1 in float64, but the nonzero omitted probability is calculated separately without cancellation. This bounds the omitted contribution to any absolute detector/channel probability. It does not justify ignoring thermal populations at another temperature.",
      '', 'The propagated operator is P H P with full nonlinear atomic displacement phases and repeated scattering. The target cutoff limits states, not scattering events. Coherent amplitudes are retained inside each final channel; orthogonal final states and thermal initial states contribute probabilities. Returning-state elastic transitions are classified by state identity, so a degenerate changed-state transition is not relabeled elastic.',
      '', '## Six-channel pilot probabilities','', '| Final state | Loss (meV) | 0–20 mrad | 0–5 mrad | 5–10 mrad | 10–20 mrad |','|---|---:|---:|---:|---:|---:|']
    for label,e,values in zip(basis,loss,prob):lines.append('| '+str(tuple(map(int,label)))+' | '+f'{e:.8f}'+' | '+' | '.join(f'{v:.10e}' for v in values)+' |')
    two=prob[basis.sum(axis=1)==2].sum(axis=0)
    lines+=['| summed n=2 | — | '+' | '.join(f'{v:.10e}' for v in two)+' |','',
      'Individual (2,0), (1,1), (0,2) populations depend on the basis within an exactly degenerate optical subspace. The fixed x/y convention, actual small finite-displacement splitting, mode phases, eigen residuals and summed sector are retained. Fractions in `channels.csv` use each detector’s inclusive probability as denominator; all original absolute probabilities remain available.',
      '', '## Checks and unresolved convergence','',
      'Criteria were frozen in `inputs/config.json` before running the pilot. Probability comparisons use |ΔP| ≤ 1e−10 + 0.01 P for every final channel and detector; the common-support complex exit comparison uses relative error ≤1e−4 without phase fitting. These finite-observable criteria do not transfer the historical bulk weak-window floor. The absolute term is a pilot probability-error budget per incident electron; it can dominate relative precision in very weak channels, so passing does not mean uniform 1% accuracy there.',
      '', '| Case | Max probability error / tolerance | Complex exit difference | Probability check | All attempt time (s) | Peak RSS (MiB) |','|---|---:|---:|---|---:|---:|']
    for r in rows:
        m=manifests[r['case']];cost=execution[r['case']]
        lines.append(f"| {r['case']} | {r['max_channel_detector_error_over_tolerance']:.6g} | {r['complex_exit_relative_difference_on_common_support']:.6g} | {'PASS' if r['channel_detector_pass'] else 'FAIL'} | {cost['total_invocation_seconds']:.2f} | {cost['maximum_recorded_rss_mib']:.1f} |")
    lines+=['','Runtime totals include every recorded invocation for that case, including failed attempts and repeated setup on a resume; maximum RSS is across those invocations. Raw logs retain each attempt. These comparisons are diagnostic, including their failures. A few passing refinements do not establish continuum electron support, a complete phonon bath, longitudinal active displacements, thermal stability, anharmonic linewidths or external calibration. No pilot is promoted merely because the norm is conserved.',
      '', '| Force-constant case | Max sampled frequency change (meV) | Covariance relative change | Relative symmetry/sum-rule correction | Check |','|---|---:|---:|---:|---|']
    for r in fc:lines.append(f"| {r['case']} | {r['maximum_sampled_frequency_change_mev']:.6g} | {r['covariance_relative_change']:.6g} | {r['relative_correction']:.6g} | {'PASS' if r['passed'] else 'FAIL'} |")
    lines+=['',f"The equilibrium force/energy derivative discrepancy was {eq['derivative_absolute_error']:.3e} eV/Å. Raw and corrected force constants, signed frequencies, symmetry residuals and correction norms are retained. Tiny signed Gamma translation frequencies are identified by translation overlap and omitted explicitly; no physical unstable frequency is clipped. The 576-atom range is 8.6552 Å. Stability checks cover the commensurate grids and the finite bath, not every continuum wavevector.",
      '', 'The first F1537/S769 attempt failed the strict field-relative Hermiticity guard in nearly zero vacuum fields (antisymmetric terms about 3e−32 J). Its failure log and original fields remain. Explicit conjugate momentum-pair construction fixes roundoff cancellation without weakening the guard; a separate paired-fields p2 case checks the exit amplitudes against the original pilot. One refinement was launched before the preceding process exit had been confirmed; all subsequent invocations use a single-worker file lock. This orchestration exception is retained rather than counted as resource-compliant evidence.',
      '', 'The mapped/full-storage action, independent reciprocal action and GH24/GH32 controls agree at approximately 1e−15 on the small complex-amplitude test. The maximum N33 transfer phase blocks agree between D32, D48, GH24 and GH32 below 1e−14. Regression tests include nonvacuum initialization, zero-prefix interruption, exact checkpoint resume, signed and degenerate transitions, detector partitions and spectral area. The initial two test failures were fixture selection and a test rounding tolerance tighter than the frozen area criterion; original failure logs are retained.',
      '', 'The old coordinate-action preflight now counts all atom profiles and construction copies. The new action streams memory-mapped atom fields and drops mapped pages after use; field and propagation grids are independent. Full complex profile storage estimates are 0.775 GiB at N33/F769, 6.10 GiB at N65/F1537 and 27.29 GiB at N97/F3073. These are storage estimates, not measured RSS. Source and per-case manifests record actual RSS/runtime; guards must not be bypassed.',
      '', '## Conditional qualitative TRPMD comparison','',
      '[Qualitative overlay](figures/qualitative_TRPMD_comparison.png) · [Unscaled and native-index views](figures/unscaled_and_native_reference.png)',
      '', 'The reference is the raw read-only `Si_5x7x52_10K_beads165_SNAP.npy`, verified float64 shape (600,267,266), 340905728 bytes, SHA-256 `46f97e1d7c651b9367d6e34c34adbf44f8d05dc67a64db3902375e714eea41b0`. The supplied analysis notebook hash is verified in `reference/analysis_evidence.json`.',
      '', 'The notebook uses (energy,y,x), FFT-centered energy, effective 7.5 fs sampling, and a radius-20-pixel circle at (133,133). The saved Studio settings corroborate timestep/stride but do not establish exact generation metadata. The energy spacing is 0.919037266 meV; the conditional native axis runs −275.711180 to +274.792143 meV. Fourier sign, native window, angular calibration, incident normalization and generation correction history remain unverified for this exact file. An adjacent notebook claims prior Kubo correction; this is analysis evidence, not a generation manifest. No further Kubo correction is applied.',
      '', '**The acceptances and thicknesses differ:** the model uses 0–20 mrad and four repeats, while the reference uses 20 pixels and a filename stating 5×7×52, 165 beads. A nearby 7280-atom structure supports the meaning of that geometry but is not linked to this array by provenance. Do not identify 20 pixels with 20 mrad. The two-mode harmonic model has a few lines and cannot reproduce the dispersive/acoustic bath or intrinsic anharmonic linewidths.',
      '', 'On the conditional energy axis the model concentrates changed-state weight around 58.04 and 116.08 meV. The reference ROI also has broad low-energy weight and a broad high-energy tail. These differences are consistent with the stated model limitations, but this comparison cannot isolate a quantitative cause.',
      '', 'The reference curve is reduced directly from the raw array, conservatively rebinned, and forward-broadened by an additional Gaussian of 2 meV FWHM. That addition does not establish matched native resolution. The notebook’s separate sigma=1 meV (FWHM≈2.355 meV), full-sum normalization and reflect smoothing are reconstructed and saved separately. No background subtraction, energy shift, phonon-frequency adjustment, peak normalization or detector fitting is used.',
      '', f"For the display-only overlay, the interval is 5–160 meV and one factor per total curve is used: model {factors['model_factor']:.12g}, reference {factors['reference_factor']:.12g}. The same model factor multiplies every component. Native reference ROI signed mass is {reference['native_roi_signed_mass']:.12g}; conservative crop removes {reference['cropped_native_mass']:.12g}. Raw negative entries and signed reductions are retained. All transformations, native axis, extended broadening tails and display factors are saved.",
      '', 'No absolute-intensity accuracy, matched-thickness agreement, material accuracy, runtime advantage or bulk acceptance follows from this overlay.',
      '', '## Files and reproducibility','',
      '- `cases/*/exit.npz`: complex coefficients, initial/final labels, physical momentum axes, detector masks and unrenormalized thermal weights.',
      '- `cases/*/spectra.npz`, `spectra.csv`, `channels.csv`: native transitions, inclusive/changed/returning spectra and channel probabilities.',
      '- `cases/*/channel_probabilities_all_acceptances.csv`, `sector_probabilities.npz`/`.csv`, `two_phonon_spectrum.csv`: full retained-beam channel weights, explicit sector sums and detector-resolved two-phonon spectra.',
      '- `phonons/`: relaxed structure, fresh raw/corrected SNAP force constants, signed modes, active vectors and full/inactive covariance.',
      '- `reference/`: raw reduction, every transformation, provenance statuses, display factors and native/cropped masses.',
      '- `checks/`, `logs/`, `provenance/`: checks, failures, resource records and exact source snapshots.',
      '', 'From `/home/zuxian/Documents/research-script/yoshioka-eels`, use the existing scientific interpreter without installing into it:',
      '', '```bash',
      'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
      'flock -n /home/zuxian/Documents/PhD/Yoshioka/benchmark_results/15_snap_two_mode/si110_snap_T10_p2_v1/.scientific-worker.lock timeout 600s /home/zuxian/Documents/PhD/Yoshioka/.venv/bin/python benchmarks/15_snap_two_mode/run.py summarize',
      '/home/zuxian/Documents/PhD/Yoshioka/.venv/bin/python -m unittest discover -s tests -v',
      '```',
      '', 'Completed cases are immutable and the runner refuses to overwrite them. For an incomplete case, rerun its exact `propagate --p … --n … --fields … --steps … --dimension …` command under `timeout 600s` using the matching committed numerical source. Resume verifies source/config/bath/field contracts, initial-wave identity and the checkpoint wave checksum. The exact saved 16-step prefix is authoritative; never alter a digest or reuse historical MOD checkpoints. If numerical source changes, start a versioned new experiment/case rather than bypassing the guard.',
      '', '**Historical status preserved:** the complete-pair four-repeat bulk benchmark remains NOT_ACCEPTED, corrected complete-net production count zero. Rejected Born-tail/control-variate results and separate nonlinear-bath, longitudinal-displacement, thermal-stability, linewidth and calibration milestones remain unchanged. No scheduler job was submitted and no remote push was performed.',
      '', 'All work here is local; a code-backed pinned-workflow handover records the final integrated revision and the next bounded step.']
    adjacent=json.loads((out/'checks/adjacent_support_comparisons.json').read_text())
    lines+=['','## Adjacent support and final software verification','',
            '| Comparison | Probability error / tolerance | Common complex exit error | Check |',
            '|---|---:|---:|---|']
    for row in adjacent:
        lines.append(f"| {row['coarse']} → {row['fine']} | {row['max_channel_detector_error_over_tolerance']:.8g} | {row['complex_exit_relative_difference']:.8g} | {'PASS' if row['passed'] else 'FAIL'} |")
    lines+=['','N65→N97 meets the tested thresholds, narrowly for the complex exit amplitude. This is evidence at fixed F769/S769 and p=2, not a complete certificate for a finer-grid production model. Independent depth and target-cutoff checks at the chosen finer grid remain a future step. The required N33 pilot remains unconverged.',
            '', 'The p=4 control retains native transitions above the ±180 meV display interval. Their cropped display mass is recorded without renormalization; this does not remove them from the saved native channel table.']
    test_log=out/'logs/scientific_suite.log'
    if test_log.exists():
        import re
        match=re.search(r'Ran (\d+) tests in ([\d.]+)s',test_log.read_text())
        if match and re.search(r'^OK\s*$',test_log.read_text(),re.MULTILINE):
            lines+=['',f"The full scientific suite passed {match[1]} tests in {match[2]} seconds with the existing material dependencies. This establishes the tested software contracts, not physical convergence."]
    wheel=out/'provenance/installed_wheel_check.json'
    if wheel.exists():
        check=json.loads(wheel.read_text())
        lines+=['',f"The installed wheel was checked from {check['working_directory']}: {check['module_count']} core modules imported with benchmark imports blocked, and {check['source_files_matched_count']} wheel source files matched the checkout. The original scientific environment lacks setuptools; the retained first build failure was resolved using the existing development build environment. Wheel installation used an isolated target under this result directory."]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # A self-contained browseable report plus direct links to the actual arrays.
    sections=[];in_table=False;in_code=False
    for line in lines:
        if line.startswith('```'):
            sections.append('</pre>' if in_code else '<pre>');in_code=not in_code;continue
        if in_code:sections.append(html.escape(line));continue
        if line.startswith('|'):
            if not in_table:sections.append('<table>');in_table=True
            if set(line.replace('|','').replace(' ','').replace(':',''))=={'-'}:continue
            sections.append('<tr>'+''.join('<td>'+html.escape(v.strip())+'</td>' for v in line.strip('|').split('|'))+'</tr>');continue
        if in_table:sections.append('</table>');in_table=False
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));sections.append(f'<h{level}>'+html.escape(line[level:].strip())+f'</h{level}>')
        elif line:sections.append('<p>'+html.escape(line)+'</p>')
    if in_table:sections.append('</table>')
    gallery=''.join(f'<figure><a href="figures/{p.name}"><img src="figures/{p.name}" loading="lazy"></a><figcaption>{html.escape(p.stem.replace("_"," "))}</figcaption></figure>' for p in sorted((out/'figures').glob('*.png')))
    (out/'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><title>SNAP Si110 10 K pilot</title><style>body{max-width:1150px;margin:35px auto;padding:0 24px;font:16px/1.55 system-ui;color:#172a3a;background:#f8fafb}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td{padding:7px;border-bottom:1px solid #ccd7df}pre{overflow:auto;background:#e7edf2;padding:15px}img{width:100%}figure{margin:30px 0}h1,h2{color:#164768}</style><p><a href="REPORT.md">Plain-text report</a> · <a href="cases/'+BASE+'/channels.csv">Pilot channel CSV</a> · <a href="cases/'+BASE+'/spectra.npz">Pilot spectra NPZ</a></p>'+gallery+'\n'.join(sections))
