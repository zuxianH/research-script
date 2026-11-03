"""Read-only aggregation of historical and newly completed repair evidence."""
import json,mmap
from pathlib import Path
import numpy as np
from scipy import constants
from scipy.integrate import trapezoid
from scipy.optimize import linear_sum_assignment

from yoshioka.integration_measures import uniform_bz_nodes
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from .eels_observables import DETECTORS,PEAK_GROUPS,compare_observables
from .workflow_io import ROOT,json_write

OUTPUT=ROOT/'benchmark_results/repair_20261102_v2'


def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def b7_diagnostics():
    old=read(ROOT/'benchmark_results/detector_cells_20261102_v1/results.json')
    new=read(OUTPUT/'B7/results.json')
    cases=(new or old)['cases'];comparisons=(new or old)['comparisons']
    primitive=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_eels_convergence/primitive_force_constants.npz')
    rows=[]
    for case in cases:
        mesh=case['settings']['mesh'];folder=Path(case['artifact']).parent
        energies=np.load(folder/'energies.npy',mmap_mode='r');weights=np.load(folder/'weights.npy',mmap_mode='r')
        branch=np.zeros(6);q_shell=np.zeros(6);qz_shell=np.zeros(6);hist=np.zeros(180)
        degenerate={};edges=np.array([0,.25,.5,1,2,4,np.inf])
        for lo in range(0,mesh**3,32768):
            hi=min(lo+32768,mesh**3);e=np.asarray(energies[lo:hi]);w=np.asarray(weights[lo:hi,:,1])
            selected=(e>=40)&(e<65);chosen=w*selected
            branch+=np.sum(chosen,axis=0)
            q,_=uniform_bz_nodes(mesh,lo,hi);cart=q@primitive.reciprocal_m_inv*1e-10
            q_shell+=np.bincount(np.clip(np.digitize(np.linalg.norm(cart,axis=1),edges)-1,0,5),weights=np.sum(chosen,axis=1),minlength=6)
            qz_shell+=np.bincount(np.clip(np.digitize(abs(cart[:,2]),edges)-1,0,5),weights=np.sum(chosen,axis=1),minlength=6)
            hist+=np.histogram(e,bins=np.linspace(0,90,181),weights=w)[0]
            # A degeneracy group is identified at each q independently; no
            # eigenvector interpolation or band-index matching across crossings.
            for first in range(6):
                starts=np.ones(len(e),bool) if first==0 else e[:,first]-e[:,first-1]>=.02
                for last in range(first,6):
                    internal=np.all(np.diff(e[:,first:last+1],axis=1)<.02,axis=1)
                    ends=np.ones(len(e),bool) if last==5 else e[:,last+1]-e[:,last]>=.02
                    mask=starts&internal&ends
                    key=f'{first}:{last}'
                    degenerate[key]=degenerate.get(key,0.)+float(np.sum(chosen[mask,first:last+1]))
            energies._mmap.madvise(mmap.MADV_DONTNEED);weights._mmap.madvise(mmap.MADV_DONTNEED)
        state=read(folder/'state.json');area=state['geometric_checks']['cell_area_mrad2']
        forward=case['observables']['exact_forward_q_zero']
        rows.append({'mesh':mesh,'central_middle_weight':float(np.sum(branch)),'energy_sorted_branch_weights':branch.tolist(),
            'branch_note':'local ascending-energy branch labels, not global polarization identities across crossings; degenerate contributions enter the detector trace as a sum',
            'q_norm_shell_edges_angstrom_inv':edges[:-1].tolist()+['infinity'],'q_norm_shell_weights':q_shell.tolist(),
            'absolute_qz_shell_weights':qz_shell.tolist(),'central_spectrum_halfmev_bin_weights':hist.tolist(),
            'energy_bin_edges_mev':np.linspace(0,90,181).tolist(),'local_degenerate_or_singleton_subspace_weights':degenerate,'subspace_definition':'adjacent energy gaps below0.02meV, diagnosed separately at each q; sum all local cluster members, without phase interpolation',
            'exact_forward_fiber_weight':forward,'angular_cell_area_mrad2':area,'forward_fiber_divided_by_cell_area':forward/area,
            'forward_note':'diagnostic estimate, not an independent detector mass or a claim of exact continuum density',
            'geometric_checks':state['geometric_checks']})
    matching=[]
    for a,b in zip(cases[:-1],cases[1:]):
        for key in a['peak_features']:
            ca=a['peak_features'][key];cb=b['peak_features'][key]
            if ca and cb:
                distances=abs(np.array([r['energy_mev'] for r in ca])[:,None]-np.array([r['energy_mev'] for r in cb])[None])
                i,j=linear_sum_assignment(distances)
                matches=[{'coarse_mev':ca[x]['energy_mev'],'fine_mev':cb[y]['energy_mev'],'difference_mev':float(distances[x,y]),'within_one_FWHM':bool(distances[x,y]<=2)} for x,y in zip(i,j)]
            else:matches=[]
            matching.append({'coarse_mesh':a['settings']['mesh'],'fine_mesh':b['settings']['mesh'],'window':key,'matches':matches,
                'procedure':'one-to-one minimum total energy-distance matching of local broadened maxima; distance<=2meV marks candidate feature identity only; original dominant-window-maximum acceptance unchanged'})
    result={'physical_quadrature_status':(new or {}).get('physical_quadrature_status','INCOMPLETE'),'comparisons':comparisons,'regional_rows':rows,
            'peak_feature_matching':matching,'offset64':read(OUTPUT/'B7_offset/results.json'),'offset96':read(OUTPUT/'B7_offset96/results.json'),
            'joint':read(OUTPUT/'B7_joint/results.json'),'FWHM_mev':2,'no_renormalization':True}
    json_write(OUTPUT/'B7_diagnosis_final.json',result)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for case in cases:
        with np.load(case['artifact']) as data:
            axis=data['energy_grid_mev'];s=data['spectrum_central_0_5_mrad']
            select=(axis>=35)&(axis<=65)
            axes[0,0].plot(axis[select],s[select],label=str(case['settings']['mesh'])+'³')
    for a,b in zip(cases[:-1],cases[1:]):
        with np.load(a['artifact']) as da,np.load(b['artifact']) as db:
            axis=db['energy_grid_mev'];select=(axis>=35)&(axis<=65)
            difference=db['spectrum_central_0_5_mrad']-da['spectrum_central_0_5_mrad']
            axes[0,1].plot(axis[select],difference[select],label=f"{a['settings']['mesh']}→{b['settings']['mesh']}")
    for j in (2,3):axes[1,0].plot([r['mesh'] for r in rows],[r['energy_sorted_branch_weights'][j] for r in rows],'.-',label=f'local energy branch {j}')
    axes[1,1].plot([r['mesh'] for r in rows],[r['forward_fiber_divided_by_cell_area'] for r in rows],'.-')
    axes[0,0].set(xlim=(35,65),xlabel='energy loss (meV)',ylabel='weight / meV',title='B7 central detector, 2 meV FWHM')
    axes[0,1].set(xlim=(35,65),xlabel='energy loss (meV)',ylabel='signed weight difference / meV',title='Successive spectral differences')
    axes[1,0].set(xlabel='uniform mesh edge',ylabel='0–5 mrad, 40–65 meV weight',title='Boundary-sensitive local branch contributions')
    axes[1,1].set(xlabel='uniform mesh edge',ylabel='fiber weight / cell area (mrad⁻²)',title='Exact-forward diagnostic; not an extra point mass')
    for ax in axes.ravel():
        if ax is not axes[1,1]:ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'B7_convergence.png',dpi=160);plt.close(fig)
    return result


def material_cases():
    cases=read(ROOT/'benchmark_results/dynamical_si_20261102_v1/results.json')['cases'].copy()
    for name in ('B9A_depth','B9A_batched','B9A_bath','B9A_thick_gemm'):
        data=read(OUTPUT/name/'results.json')
        if data:cases+=data.get('cases',[])
        # Completed case records are atomic even before an invocation exits.
        for p in (OUTPUT/name).glob('cache/*/case.json'):
            cases.append(read(p))
    unique={}
    for row in cases:
        if row.get('status')=='COMPLETE':unique[json.dumps(row['settings'],sort_keys=True)]=row
    return list(unique.values())


def compare_9a(a,b):
    common=[k for k in a['observables'] if k in b['observables']]
    keys=[k for k in common if any(k==d or k.startswith(d+'/') for d in DETECTORS)]
    comparison=compare_observables({k:a['observables'][k] for k in keys},{k:b['observables'][k] for k in keys})
    diagnostics=compare_observables({k:a['observables'][k] for k in common},{k:b['observables'][k] for k in common})
    with np.load(a['artifact']) as da,np.load(b['artifact']) as db:
        axis=db['energy_grid_mev'];spectra={};peaks=[]
        for name in DETECTORS:
            first=da['spectrum_'+name];second=db['spectrum_'+name]
            denominator=max(float(trapezoid(second,axis)),1e-18)
            spectra[name]={'absolute':float(trapezoid(abs(first-second),axis)),'fine_integral':denominator,
                           'relative':float(trapezoid(abs(first-second),axis)/denominator)}
            for key,(lo,hi) in PEAK_GROUPS.items():
                select=(axis>=lo)&(axis<hi)
                if not np.any(select) or max(np.max(abs(first[select])),np.max(abs(second[select])))<1e-18:continue
                x=float(axis[select][np.argmax(first[select])]);y=float(axis[select][np.argmax(second[select])])
                peaks.append({'window':name+'/'+key,'coarse_mev':x,'fine_mev':y,'absolute_change_mev':abs(x-y)})
        # Identical physical Fourier beams inside the detector, at all grids.
        qa=da['qxy_m_inv'];qb=db['qxy_m_inv'];k=2*np.pi/(4.8660604983e-12)
        from yoshioka.free_propagation import electron_wavenumber
        k=electron_wavenumber(60000*constants.elementary_charge)
        qa=qa[np.linalg.norm(qa,axis=1)/k*1000<=20+1e-12];qb=qb[np.linalg.norm(qb,axis=1)/k*1000<=20+1e-12]
        same=len(qa)==len(qb) and np.max(np.min(np.linalg.norm(qa[:,None]-qb[None,:],axis=-1),axis=1))/k*1000<1e-10
    passed=comparison['maximum_acceptance_error']<.01 and max(v['relative'] for v in spectra.values())<.01 and max((p['absolute_change_mev'] for p in peaks),default=0)<.2
    return {'coarse':a['settings'],'fine':b['settings'],'intensities':comparison,'all_reported_diagnostics':diagnostics,'spectra':spectra,'peaks':peaks,
            'same_physical_detector_beams':bool(same),'passed':bool(passed),'definition':'abs(coarse-fine)/max(abs(fine),1e-18); L1 denominator fine detector integral; peak maxima are window features, not identified phonon modes'}


def b9a_diagnostics():
    cases=material_cases();comparisons=[]
    def subset(**settings):return sorted([r for r in cases if all(r['settings'].get(k)==v for k,v in settings.items())],key=lambda r:(r['settings']['grid'],r['settings']['phonon_z']))
    thin=subset(slices=1025,thickness_repeats=2,phonon_z=2,temperature_k=0,padding_angstrom=12,elastic_background=True)
    thick=subset(slices=2049,thickness_repeats=52,phonon_z=2,temperature_k=0,padding_angstrom=12,elastic_background=True)
    bath=subset(grid=65,slices=2049,thickness_repeats=52,temperature_k=0,padding_angstrom=12,elastic_background=True)
    for category,sequence in (('thin_electron_grid',thin),('target_thickness_electron_grid',thick),('longitudinal_phonon_quadrature',bath)):
        for a,b in zip(sequence[:-1],sequence[1:]):comparisons.append({'category':category,**compare_9a(a,b)})
    thin_pairs=[c for c in comparisons if c['category']=='thin_electron_grid']
    result={'status':'INCOMPLETE','cases':cases,'comparisons':comparisons,
            'thin_electron_grid_finest_two_pairs':'PASS' if len(thin_pairs)>=2 and thin_pairs[-1]['fine']['grid']>=257 and all(c['passed'] for c in thin_pairs[-2:]) else 'INCOMPLETE',
            'target_thickness_electron_joint_status':'INCOMPLETE','phonon_status':'INCOMPLETE',
            'bath_definition':'z changes BZ qz sampling, retained mode count and harmonic covariance at fixed atom slab, FC model and T; not a change of force-constant supercell, thermal samples or thickness',
            'missing':'two target-thickness electron refinements and joint source/slice check; qz sampling and transverse Bloch bath refinement; full material remains incomplete',
            'branch_counts':{str(z):12*2*2*z-3 for z in (2,4,8,16)}}
    json_write(OUTPUT/'B9A_diagnosis_final.json',result)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for row in thin:
        with np.load(row['artifact']) as d:
            axis=d['energy_grid_mev'];select=(axis>=0)&(axis<=40)
            axes[0].plot(axis[select],d['spectrum_annulus_5_10_mrad'][select],label=str(row['settings']['grid'])+'²')
    for row in bath:
        with np.load(row['artifact']) as d:axes[1].plot(d['energy_grid_mev'],d['spectrum_central_0_5_mrad'],label=f"bath z={row['settings']['phonon_z']}")
    for row in thick:
        with np.load(row['artifact']) as d:axes[2].plot(d['energy_grid_mev'],d['spectrum_central_0_5_mrad'],label=str(row['settings']['grid'])+'²')
    axes[0].set(title='Thin slab: weak annular acoustic signal',xlim=(0,40))
    axes[1].set(title='Fixed 19.962 nm slab: phonon sampling',xlim=(0,90))
    axes[2].set(title='Fixed thick slab: electron sampling',xlim=(0,90))
    for ax in axes:ax.set(xlabel='electron energy loss (meV)',ylabel='undepleted source weight / meV');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'B9A_electron_and_phonon.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    key='annulus_5_10_mrad/acoustic_0_40_mev'
    axes[0].plot([r['settings']['grid'] for r in thin],[r['observables'][key] for r in thin],'.-')
    axes[0].set(xlabel='electron grid edge',ylabel='raw integrated weight',title='Fixed 5–10 mrad, 0–40 meV window')
    selected=[c for c in comparisons if c['category']=='thin_electron_grid']
    vals=[next(r for r in c['intensities']['rows'] if r['observable']==key)['acceptance_error']*100 for c in selected]
    axes[1].bar([f"{c['coarse']['grid']}→{c['fine']['grid']}" for c in selected],vals)
    axes[1].axhline(1,color='k',ls='--',label='unchanged 1% target');axes[1].legend()
    axes[1].set(ylabel='absolute change / finer weight (%)',title='No weight or spectrum renormalization')
    fig.savefig(OUTPUT/'B9A_weak_window.png',dpi=160);plt.close(fig)
    return result


def thermal_figures():
    data=read(OUTPUT/'B11_checked/results.json');audit=read(OUTPUT/'B11_final_audit/results.json') or read(OUTPUT/'B11_audit/results.json')
    if not data:return None
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for T,ax in zip((300,1000),axes[0]):
        for ti,tr in enumerate(data['trajectories']):
            if tr['settings']['temperature_k']!=T or tr['settings']['seed'] not in (723571,837213) or tr['settings']['repetitions']!=[4,6,6]:continue
            with np.load(OUTPUT/f'B11_checked/mode_T{T}_q0_g3_traj{ti}.npz') as d:
                E=d['E_100ps'];P=d['PSD_100ps']*((1e-3*constants.elementary_charge)/constants.h)
                ax.plot(E,P,label=f"seed {tr['settings']['seed']}")
        row=next(r for r in data['frequency_rows'] if r['temperature_k']==T and r['group']==[9,10,11] and r['q_fractional']==[0,0,0])
        ax.axvline(row['effective_energy_mev'],color='k',ls='--',label='effective harmonic')
        ax.set(xlim=(min(row['effective_energy_mev'],row['MD_center_mev'])-3,row['MD_center_mev']+3),xlabel='mode energy (meV)',ylabel='velocity-variance density (J/meV)',title=f'Γ optical subspace, {T} K target, 100 ps Hann')
        ax.legend(fontsize=8)
    from yoshioka.mode_spectra import hann_energy_kernel
    E=np.linspace(-.8,.8,4001)
    for duration in (20,50,100):axes[1,0].plot(E,hann_energy_kernel(E,duration),label=f'{duration} ps')
    axes[1,0].set(xlabel='energy offset (meV)',ylabel='normalized window density (meV⁻¹)',title='Finite-record response; no Gaussian smoothing');axes[1,0].legend()
    for T in (300,1000):
        rows=[r for r in data['frequency_rows'] if r['temperature_k']==T]
        axes[1,1].errorbar(np.arange(len(rows))+(0 if T==300 else .15),[r['effective_minus_MD_mev'] for r in rows],
            yerr=[r['MD_conservative_spread_mev'] for r in rows],fmt='.',label=f'{T} K')
    axes[1,1].axhline(0,color='k',lw=.7);axes[1,1].set(xlabel='matched mode/subspace index',ylabel='EHM minus MD energy (meV)',title='Conservative replica/segment spread; no lifetimes');axes[1,1].legend()
    fig.savefig(OUTPUT/'B11_modes_and_resolution.png',dpi=160);plt.close(fig)
    return {'original_model_status':'FAIL','largest_new_frequency_difference_mev':max(abs(r['effective_minus_MD_mev']) for r in data['frequency_rows']),
            'new_failing_frequency_rows':[r for r in data['frequency_rows'] if not r['original_frequency_gate_pass']],
            'linewidth_status':data['linewidth_status'],'low_amplitude_control':None if not audit else audit['maximum_low_amplitude_frequency_discrepancy_mev']}


def coordinate_cases():
    data=read(OUTPUT/'B9B_coordinate/results.json') or read(OUTPUT/'B9B_coordinate/checkpoint.json')
    records=[] if not data else data['cases'].copy()
    for path in (OUTPUT/'B9B_coordinate').glob('*.json'):
        row=read(path)
        if row and all(key in row for key in ('settings','artifact','observables')):records.append(row)
    return list({json.dumps(row['settings'],sort_keys=True):row for row in records}.values())


def repeated_figures():
    records=coordinate_cases()
    orders=read(OUTPUT/'B9B_orders/results.json')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    if records:
        all_cases=records.copy();fine=read(OUTPUT/'B9B_fine/results.json')
        if fine and fine['status']=='COMPLETE':all_cases.append(fine)
        cases=sorted([r for r in all_cases if r['settings']['p']==4 and r['settings']['slices']==257 and r['settings']['order']==24],key=lambda r:r['settings']['n'])
        for row in cases:
            with np.load(row['artifact']) as a:axes[0].plot(a['energy_grid_mev'],a['spectrum_full_0_20_mrad'],label=f"{row['settings']['n']}×{row['settings']['n']} beams")
        axes[1].plot([r['settings']['n'] for r in cases],[r['observables']['one_phonon'] for r in cases],'.-',label='all represented electron beams')
        axes[1].plot([r['settings']['n'] for r in cases],[r['observables']['full_0_20_mrad/inelastic'] for r in cases],'.-',label='fixed 20 mrad detector')
    if orders:
        with np.load(OUTPUT/'B9B_orders/orders.npz') as a:
            for name,label in (('full','full repeated'),('one_interaction','one inter-channel interaction'),('orders_through_two','coherent orders 0+1+2')):
                select=(a['energy_grid_mev']>=145)&(a['energy_grid_mev']<=160)
                axes[2].plot(a['energy_grid_mev'][select],a['spectrum_'+name][select],label=label)
    axes[0].set(xlim=(65,175),xlabel='energy loss (meV)',ylabel='reduced-model weight / meV',title='Si transverse-Γ model, 0.384 nm, T=0')
    axes[0].set_yscale('log');axes[0].set_ylim(1e-15,1e-5)
    axes[1].set(xlabel='electron reciprocal basis edge',ylabel='inelastic weight',title='Detector and total-state convergence differ')
    axes[2].set(xlim=(145,160),xlabel='energy loss (meV)',ylabel='reduced-model weight / meV',title='Two-phonon feature: direct and repeated amplitudes')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'B9B_reduced_repeated_EELS.png',dpi=160);plt.close(fig)
    if orders:
        with np.load(OUTPUT/'B9B_orders/orders.npz') as a:
            fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
            axis=a['energy_grid_mev']
            for name,label in (('full','full'),('one_interaction','one interaction'),('orders_through_two','orders0+1+2')):
                axes[0].plot(axis,a['spectrum_'+name],label=label)
            axes[0].set_yscale('log');axes[0].set_ylim(1e-15,1e-4)
            axes[1].plot(axis,a['spectrum_full']-a['spectrum_one_interaction'])
            axes[1].axhline(0,color='k',lw=.7)
            axes[0].set(title='Same nonlinear Hamiltonian and detector trace',ylabel='weight / meV')
            axes[1].set(title='Signed full minus one-interaction spectrum',ylabel='signed weight difference / meV')
            for ax in axes:ax.set(xlim=(60,320),xlabel='electron energy loss (meV)');ax.grid(alpha=.2)
            axes[0].legend(fontsize=8);fig.savefig(OUTPUT/'B9B_orders_difference.png',dpi=160);plt.close(fig)


def repeated_convergence():
    cases=coordinate_cases()
    if not cases:return
    fine=read(OUTPUT/'B9B_fine/results.json')
    if fine and fine['status']=='COMPLETE':cases.append(fine)
    comparisons=[]
    from .repeated_scattering import compare_values
    for parameter,fixed in (('n',{'p':4,'slices':257,'modes':2,'order':24}),('p',{'n':33,'slices':257,'modes':2,'order':24}),
        ('slices',{'n':33,'p':4,'modes':2,'order':24}),('order',{'n':33,'p':4,'slices':257,'modes':2})):
        sequence=sorted([r for r in cases if all(r['settings'][k]==v for k,v in fixed.items())],key=lambda r:r['settings'][parameter])
        for a,b in zip(sequence[:-1],sequence[1:]):
            strict=compare_values(a['observables'],b['observables'])
            material=compare_observables(a['observables'],b['observables'])
            with np.load(a['artifact']) as da,np.load(b['artifact']) as db:
                before=da['wave'];after=db['wave'];embedding=np.zeros_like(after)
                lookup={tuple(v):i for i,v in enumerate(db['basis'])}
                start=(after.shape[1]-before.shape[1])//2
                for i,v in enumerate(da['basis']):
                    if tuple(v) in lookup:embedding[lookup[tuple(v)],start:start+before.shape[1],start:start+before.shape[2]]=before[i]
                complex_error=float(np.linalg.norm(after-embedding))
            comparisons.append({'parameter':parameter,'coarse':a['settings'],'fine':b['settings'],'complex_amplitude_difference_in_finer_space':complex_error,
                'B8_style_observable_checks':strict,'material_observable_checks':material,'material_intensity_pass':material['maximum_acceptance_error']<.01})
    result={'status':'REDUCED_MATERIAL_COMPUTED_FULL_TARGET_INCOMPLETE','cases':cases,'comparisons':comparisons,
            'cutoff_evidence':'empirical total occupation refinement, no infinite-space omitted-coupling bound claimed',
            'matrix_free_validation':read(OUTPUT/'B9B_validation/results.json'),'orders':read(OUTPUT/'B9B_orders/results.json'),
            'full_target_limits':['bulk phonon q bath not represented','longitudinal Gamma coordinates vacuum projected','onlyone0.384nm unitcell, not19.962nm target','all-electron-beam state convergence stricter than fixed detector convergence','no finite-T macroscopic trace'],
            'B8_style_rule':'unchanged abs1e-10 when |fine|<1e-6; otherwise relative1e-4; provided separately from inherited material1% weak-scale1e-18 rule'}
    json_write(OUTPUT/'B9B_convergence_summary.json',result)


def gamma_refinement():
    base=read(OUTPUT/'B9B_material/results.json');extra=read(OUTPUT/'B9B_gamma6/results.json')
    if not base or not extra:return
    cases=sorted([r for r in base['cases']+extra['cases'] if r['settings']['p']==3 and r['settings']['n']==5 and r['settings']['slices']==129],key=lambda r:r['settings']['modes'])
    from .repeated_scattering import compare_values
    comparisons=[]
    for a,b in zip(cases[:-1],cases[1:]):
        common=set(a['observables'])&set(b['observables'])
        comparisons.append({'coarse':a['settings'],'fine':b['settings'],
            'B8_style':compare_values({k:a['observables'][k] for k in common},{k:b['observables'][k] for k in common}),
            'material_style':compare_observables({k:a['observables'][k] for k in common},{k:b['observables'][k] for k in common})})
    json_write(OUTPUT/'B9B_Gamma_refinement.json',{'status':'INCOMPLETE','cases':cases,'comparisons':comparisons,
        'scope':'active transverse Gamma families2/4/6 at fixed p3,n5,slices129; longitudinalGamma coordinates vacuum projected; not fullBZ or joint convergence',
        'discarded_intermediate_states':'inactive modes are a declared projection; no rigorous infinite-bath error bound claimed'})
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for row in cases:
        with np.load(row['artifact']) as a:
            for ax in axes:ax.plot(a['energy_grid_mev'],a['spectrum_full_0_20_mrad'],label=f"{row['settings']['modes']} active transverse Γ modes")
    axes[0].set(xlim=(0,170),ylabel='reduced-model inelastic weight / meV',title='Fixed p3, 5×5 beams, 0.384 nm Si')
    axes[1].set(xlim=(0,250),ylim=(1e-16,1e-5),ylabel='reduced-model inelastic weight / meV',title='Log scale retains weak multiphonon lines');axes[1].set_yscale('log')
    for ax in axes:ax.set_xlabel('electron energy loss (meV)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'B9B_Gamma_modes.png',dpi=160);plt.close(fig)


def thermal_repair_figures():
    old=read(OUTPUT/'B11/results.json');new=read(OUTPUT/'B11_checked/results.json');refit=read(OUTPUT/'B11_refit/results.json')
    if not old or not new or not refit:return
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for data,label in ((old,'before: stale neighbours'),(new,'after: checked restart')):
        tr=next(r for r in data['trajectories'] if r['settings']['temperature_k']==1000 and r['settings']['seed']==723571 and r['settings']['repetitions']==[4,6,6])
        folder=Path(tr['folder']);energy=np.load(folder/'total_energy_j.npy');temperature=np.load(folder/'temperature_k.npy')
        time=(np.arange(len(energy))+1)*tr['state']['sample_interval_s']*1e12
        scale=1.5*(576-1)*constants.Boltzmann*tr['state']['mean_temperature_k']
        axes[0].plot(time,(energy-energy[0])/scale,label=label,lw=.9)
        axes[1].plot(time[:len(time)//200*200].reshape(-1,200).mean(axis=1),temperature[:len(time)//200*200].reshape(-1,200).mean(axis=1),label=label)
    broken=read(OUTPUT/'MD_checkpoint_force_audit.json')['rows'];fixed=read(OUTPUT/'MD_checked_pilot.json')['fresh_force_checks']
    axes[2].semilogy([r['time_ps'] for r in broken if r['trajectory']==2],[r['relative_force_error'] for r in broken if r['trajectory']==2],'.-',label='before')
    subset=[r for r in fixed if 12398<=r['frame']<=12402]
    axes[2].semilogy([(r['frame']+1)*.005 for r in subset],[r['relative_force_error'] for r in subset],'.-',label='after')
    axes[0].set(xlabel='NVE time (ps)',ylabel='[E(t)−E(0)] / thermal KE',title='Checkpoint error changed the trajectory')
    axes[1].set(xlabel='NVE time (ps)',ylabel='1 ps block mean temperature (K)',title='Same starting state and nominal 1000 K')
    axes[2].set(xlabel='NVE time (ps)',ylabel='relative force error against fresh MOD',title='Independent static-force check near write')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'MD_checkpoint_repair.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for T,ax in zip((300,1000),axes):
        rows=[r for r in refit['fits'] if r['temperature_k']==T]
        ax.plot([r['count'] for r in rows],[r['training_force_relative_RMS_error'] for r in rows],'.-',label='training')
        ax.plot([r['count'] for r in rows],[r['force_relative_RMS_error'] for r in rows],'.-',label='independent held-out trajectory')
        ax.axhline(.15,color='k',ls='--',label='unchanged target0.15')
        ax.set(xlabel='training configurations',ylabel='relative force RMS error',title=f'{T} K target: corrected 100 ps data')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(OUTPUT/'B11_corrected_fit_learning.png',dpi=160);plt.close(fig)


def main():
    import matplotlib;matplotlib.use('Agg')
    b7=b7_diagnostics();b9=b9a_diagnostics();thermal=thermal_figures();repeated_figures();repeated_convergence();gamma_refinement();thermal_repair_figures()
    json_write(OUTPUT/'analysis_status.json',{'B7_phonon':b7['physical_quadrature_status'],'B7_joint':'INCOMPLETE',
        'B9A_thin_electron':b9['thin_electron_grid_finest_two_pairs'],'B9A_full':'INCOMPLETE','B11':thermal})


if __name__=='__main__':main()
