"""Distinct convergence gates for the symmetry-preserving source calculations."""
import json
from pathlib import Path
import numpy as np
from scipy import constants
from scipy.integrate import trapezoid
from .eels_observables import DETECTORS,PEAK_GROUPS,compare_observables
from .workflow_io import ROOT,json_write

OUT=ROOT/'benchmark_results/repair_20261102_v2'


def collect():
    rows={}
    for name in ('B9A_bloch','B9A_bloch_electron','B9A_bloch_bath','B9A_bloch_extended','B9A_bloch_transverse','B9A_bloch_fullmesh','B9A_fourier_grid','B9A_fourier_joint','B9A_fft_grid','B9A_fft_joint'):
        for path in (OUT/name).glob('cache/*/case.json'):
            row=json.loads(path.read_text());s={**row['settings'],'phonon_xy':row['settings'].get('phonon_xy',2)}
            row['physical_settings']=s;rows[json.dumps(s,sort_keys=True)]=row
    return list(rows.values())


def compare(a,b):
    # The historical acceptance code evaluates these original four detector
    # totals/windows/gain weights. All additional forward/off-axis diagnostics
    # are retained and compared separately, never silently removed.
    keys=[key for key in a['observables'] if any(key==d or key.startswith(d+'/') for d in DETECTORS)]
    primary=compare_observables({k:a['observables'][k] for k in keys},{k:b['observables'][k] for k in keys})
    all_values=compare_observables(a['observables'],b['observables']);l1={};peaks=[]
    with np.load(a['artifact']) as da,np.load(b['artifact']) as db:
        axis=db['energy_grid_mev']
        for name in DETECTORS:
            first=da['spectrum_'+name];second=db['spectrum_'+name]
            l1[name]=float(trapezoid(abs(first-second),axis)/max(trapezoid(second,axis),1e-18))
            for group,(lo,hi) in PEAK_GROUPS.items():
                select=(axis>=lo)&(axis<hi)
                if not np.any(select) or max(np.max(abs(first[select])),np.max(abs(second[select])))<1e-18:continue
                x=float(axis[select][np.argmax(first[select])]);y=float(axis[select][np.argmax(second[select])])
                peaks.append({'window':name+'/'+group,'coarse_mev':x,'fine_mev':y,'absolute_change_mev':abs(x-y)})
    common=max(l1.values())<.01 and max((r['absolute_change_mev'] for r in peaks),default=0)<.2
    return {'coarse':a['physical_settings'],'fine':b['physical_settings'],'original_detector_checks':primary,
        'all_reported_observable_checks':all_values,'spectral_L1':l1,'peaks':peaks,
        'original_gate_passed':bool(common and primary['maximum_acceptance_error']<.01),
        'all_diagnostics_passed':bool(common and all_values['maximum_acceptance_error']<.01)}


def main():
    rows=collect();comparisons=[]
    def sequence(**fixed):return [r for r in rows if all(r['physical_settings'].get(k)==v for k,v in fixed.items())]
    fixed={'thickness_repeats':52,'temperature_k':0,'padding_angstrom':12,'elastic_background':True}
    electron=sorted(sequence(**fixed,phonon_xy=2,phonon_z=2,slices=2049),key=lambda r:r['physical_settings']['grid_unit'])
    slices=sorted(sequence(**fixed,phonon_xy=2,phonon_z=2,grid_unit=65),key=lambda r:r['physical_settings']['slices'])
    bathz=sorted(sequence(**fixed,phonon_xy=2,grid_unit=33,slices=2049),key=lambda r:r['physical_settings']['phonon_z'])
    bathxy=sorted(sequence(**fixed,phonon_z=2,grid_unit=33,slices=2049),key=lambda r:r['physical_settings']['phonon_xy'])
    for label,group in (('electron_grid',electron),('slices',slices),('phonon_z',bathz),('phonon_xy',bathxy)):
        for a,b in zip(group[:-1],group[1:]):comparisons.append({'changed':label,**compare(a,b)})
    joint=[]
    finest_n=max((r['physical_settings']['grid_unit'] for r in electron),default=129)
    targets=[r for r in sequence(**fixed,phonon_xy=2,phonon_z=2,grid_unit=finest_n) if r['physical_settings']['slices']>2049]
    target=sorted(targets,key=lambda r:r['physical_settings']['slices'])[-1:]
    if target:
        previous_n=electron[-2]['physical_settings']['grid_unit'] if len(electron)>1 else finest_n
        for coarse in (sequence(**fixed,phonon_xy=2,phonon_z=2,grid_unit=finest_n,slices=2049),sequence(**fixed,phonon_xy=2,phonon_z=2,grid_unit=previous_n,slices=2049)):
            if coarse:joint.append(compare(coarse[0],target[0]))
    def status(category,rule):
        selected=[r for r in comparisons if r['changed']==category]
        return 'PASS' if len(selected)>=2 and all(r[rule] for r in selected[-2:]) else 'INCOMPLETE'
    result={'status':'INCOMPLETE','cases':rows,'comparisons':comparisons,'electron_joint':joint,
        'electron_original_four_detector_gate':status('electron_grid','original_gate_passed'),
        'electron_all_reported_diagnostics':status('electron_grid','all_diagnostics_passed'),
        'slice_original_gate':status('slices','original_gate_passed'),
        'electron_joint_original_gate':'PASS' if len(joint)==2 and all(r['original_gate_passed'] for r in joint) else 'INCOMPLETE',
        'phonon_full_3D_status':'INCOMPLETE','full_electron_phonon_joint_status':'INCOMPLETE',
        'acceptance_audit':'original4detector/windows/gain gates:abs delta/max(abs finer,1e-18)<1%,relativeL1<1%,dominant-window maximum shift<0.2meV, two successive refinements; additional offaxis/forward diagnostics retained separately',
        'representation':'all physical modes of the same periodic phonon bath, electron unit-cell Bloch sectors with physical Qperp=G-qperp; no intensity projection/renormalization',
        'numerical_vs_model':'electron grids and slices refine numerical sampling; phonon xy/z refine distinct BZ quadratures and mode counts with corresponding covariance; force constants, slab, temperature and2meV FWHM stay fixed',
        'validation':json.loads((OUT/'B9A_bloch_validation/results.json').read_text())}
    # Measure a known-zero elastic sector, not just total norm.
    model_path=ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz'
    from yoshioka.lattice_dynamics import RealSpaceForceConstants
    model=RealSpaceForceConstants.load(model_path);leak=[]
    legacy=json.loads((ROOT/'benchmark_results/dynamical_si_20261102_v1/results.json').read_text())['cases']
    for name in ('B9A_depth','B9A_batched','B9A_thick_gemm'):
        for p in (OUT/name).glob('cache/*/case.json'):legacy.append(json.loads(p.read_text()))
    for r in legacy:
        s=r['settings']
        if s['temperature_k'] or not s['elastic_background'] or s['thickness_repeats']!=52 or s['phonon_z']!=2:continue
        folder=Path(r['checkpoint']).parent;p=folder/'incident_exit.npy'
        if not p.exists():
            state=json.loads(Path(r['checkpoint']).read_text());p=Path(state['adopted_incident']).parent/'incident_exit.npy'
        wave=np.load(p);n=wave.shape[0];h=np.rint(np.fft.fftfreq(n)*n).astype(int)
        forbidden=(h[:,None]%2!=0)|(h[None,:]%2!=0)
        coefficients=np.fft.fft2(wave)*np.sqrt(4*model.cell_m[0,0]*model.cell_m[1,1])/(n*n)
        leak.append({'route':'odd2x2cell','grid':n,'dx_angstrom':2*model.cell_m[0,0]*1e10/n,'forbidden_sector_weight':float(np.sum(abs(coefficients[forbidden])**2)),'norm':float(np.sum(abs(coefficients)**2))})
    for r in electron:
        folder=Path(r['checkpoint']).parent;wave=np.load(folder/'incident_exit.npy');n=wave.shape[0]
        repeated=np.tile(wave,(2,2))/2;coeff=np.fft.fft2(repeated)*np.sqrt(4*model.cell_m[0,0]*model.cell_m[1,1])/(4*n*n)
        h=np.rint(np.fft.fftfreq(2*n)*2*n).astype(int);forbidden=(h[:,None]%2!=0)|(h[None,:]%2!=0)
        leak.append({'route':'unit_cell_Bloch','grid':n,'dx_angstrom':model.cell_m[0,0]*1e10/n,'forbidden_sector_weight':float(np.sum(abs(coeff[forbidden])**2)),'norm':float(np.sum(abs(coeff)**2))})
    result['elastic_forbidden_sector_diagnostic']=leak
    json_write(OUT/'B9A_Bloch_convergence.json',result)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for r in electron:
        with np.load(r['artifact']) as a:axes[0].plot(a['energy_grid_mev'],a['spectrum_full_0_20_mrad'],label=str(r['physical_settings']['grid_unit'])+'²/unit')
    for r in rows:
        s=r['physical_settings']
        if s['grid_unit']!=33 or s['slices']!=2049 or s['thickness_repeats']!=52 or not s['elastic_background']:continue
        with np.load(r['artifact']) as a:axes[1].plot(a['energy_grid_mev'],a['spectrum_central_0_5_mrad'],label=f"{s['phonon_xy']}×{s['phonon_xy']}×{s['phonon_z']};M={r['mode_count']}")
    for name in ('odd2x2cell','unit_cell_Bloch'):
        points=[r for r in leak if r['route']==name];points.sort(key=lambda r:r['dx_angstrom'])
        axes[2].semilogy([r['dx_angstrom'] for r in points],[max(r['forbidden_sector_weight'],1e-36) for r in points],'.-',label=name)
    axes[0].set(xlim=(0,90),xlabel='electron energy loss (meV)',ylabel='source weight / meV',title='Fixed 19.962 nm slab: electron refinements')
    axes[1].set(xlim=(0,90),xlabel='electron energy loss (meV)',ylabel='source weight / meV',title='Phonon directions differ at equal mode count')
    axes[2].set(xlabel='physical electron dx (Å)',ylabel='translation-forbidden elastic weight',title='Norm alone did not detect aliasing')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(OUT/'B9A_Bloch_repairs.png',dpi=160);plt.close(fig)
    print({k:v for k,v in result.items() if k.endswith('gate') or k.endswith('status')},flush=True)


if __name__=='__main__':main()
