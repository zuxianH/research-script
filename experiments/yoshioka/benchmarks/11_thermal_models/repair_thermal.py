"""Longer independent NVE references; original effective-model failure retained.

Direct mode-resolved dynamics is an explicitly separate richer reference, not
a fitted effective-harmonic correction and not a TACAW peak fit.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import constants
from scipy.optimize import linear_sum_assignment

from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.molecular_dynamics import checkpointed_md
from yoshioka.mode_spectra import project_normal_coordinates,periodogram_modes,fit_windowed_mode,hann_fwhm_mev
from .anharmonic_validation import groups_at_q
from .workflow_io import ROOT,ResourceLimits,json_write,sha256


def envelope_effective_samples(values):
    """Conservative initial-positive-sequence ESS for an energy-like envelope.

    This measures serial correlation, not a rigorous uncertainty for all fitted
    parameters. Slow envelope correlations must not be replaced by the first
    zero of the oscillating displacement autocorrelation.
    """
    x=np.asarray(values,dtype=float);x=x-np.mean(x);n=len(x)
    if not np.any(x):return {'effective_samples':float(n),'integrated_correlation_samples':1.}
    f=np.fft.rfft(x,n=2*n);c=np.fft.irfft(abs(f)**2,n=2*n)[:n]/np.arange(n,0,-1);c/=c[0]
    pair=c[1:-1:2]+c[2::2];positive=np.flatnonzero(pair<=0)
    stop=int(positive[0]) if len(positive) else len(pair)
    tau=max(1.,1+2*float(np.sum(pair[:stop])))
    return {'effective_samples':float(min(n,n/tau)),'integrated_correlation_samples':tau,
            'raw_samples':n,'assumption':'initial-positive-pair autocorrelation sum of squared modal velocity envelope; diagnostic, not independent replicas'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='benchmarks/configs/repair_thermal_20261102_v2.json')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    output=ROOT/config['output'];output.mkdir(parents=True,exist_ok=True)
    from ase.io import read
    unit=read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
    limits=ResourceLimits(config['runtime_limit_seconds'],config['memory_limit_gib'],100)
    old_path=ROOT/'benchmark_results/anharmonic_20261102_v1/results.json'
    old=json.loads(old_path.read_text())
    result={'status':'INCOMPLETE','original_effective_model_status':'FAIL','richer_extension':'direct classical mode-resolved NVE dynamics',
            'parameters':config,'trajectories':[],'mode_fits':[],'frequency_rows':[],
            'old_models_sha256':sha256(old_path),'line_width_convention':'energy FWHM=2 hbar/τ_amplitude; τ_population=hbar/FWHM; no lifetime for unresolved width'}
    for item in config['trajectories']:
        settings={**config['default_settings'],**item}
        folder,state=checkpointed_md(unit,ROOT/'Si.tersoff.mod',settings,output/'trajectories',limits)
        row={'settings':settings,'folder':str(folder),'state':state};result['trajectories'].append(row)
        json_write(output/'checkpoint.json',result)
        print('thermal MD',settings,state['status'],state.get('mean_temperature_k'),state['cumulative_seconds'],flush=True)
        if state['status']!='COMPLETE':
            result['resource_costs']=limits.costs();json_write(output/'results.json',result);return
    reference=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    for T in (300,1000):
        selected=next(r for r in old['fits'] if r['temperature_k']==T and r['supercell']==[4,6,6] and r['count']==64 and r['cutoff_angstrom']==5 and r['ridge']==0)
        model=RealSpaceForceConstants.load(selected['model_file'])
        for qi,q in enumerate((np.array([0.,0.,0.]),np.array([0.,.5,0.]))):
            ref_e,ref_v,groups=groups_at_q(reference,q)
            omega,v=model.modes(q);e=omega*constants.hbar/(1e-3*constants.elementary_charge)
            _,assignment=linear_sum_assignment(-abs(ref_v.conj().T@v)**2)
            for gi,group in enumerate(groups):
                matched=assignment[group];predicted=float(np.mean(e[matched]));fits=[]
                for ti,tr in enumerate(result['trajectories']):
                    if tr['settings']['temperature_k']!=T:continue
                    atoms=unit.repeat(tuple(tr['settings']['repetitions']));folder=Path(tr['folder'])
                    velocity=np.load(folder/'velocities_m_s.npy',mmap_mode='r')
                    cadence=tr['state']['sample_interval_s'];count=len(velocity)
                    projected=project_normal_coordinates(velocity,atoms.positions*1e-10,atoms.get_masses()*constants.atomic_mass,model,q,v[:,matched])
                    ess=envelope_effective_samples(np.sum(abs(projected)**2,axis=1))
                    saved={}
                    for duration in config['segments_ps']:
                        frames=int(round(duration*1e-12/cadence))
                        for bi,lo in enumerate(range(0,count-frames+1,frames)):
                            energy,power=periodogram_modes(projected[lo:lo+frames],cadence)
                            fit=fit_windowed_mode(energy,power,predicted,duration)
                            fit.update(T=T,q=q.tolist(),group=group,trajectory=ti,segment_index=bi,segment_ps=duration,
                                actual_temperature_k=float(np.mean(np.load(folder/'temperature_k.npy',mmap_mode='r')[lo:lo+frames])),
                                repetitions=tr['settings']['repetitions'],dt_fs=tr['settings']['timestep_fs'],seed=tr['settings']['seed'],
                                Nyquist_mev=constants.h/(2*cadence)/(1e-3*constants.elementary_charge),envelope_ESS=ess)
                            fits.append(fit);result['mode_fits'].append(fit)
                            if bi==0:saved[f'E_{duration}ps']=energy;saved[f'PSD_{duration}ps']=power
                    np.savez_compressed(output/f'mode_T{T}_q{qi}_g{gi}_traj{ti}.npz',**saved)
                main=[r for r in fits if r['repetitions']==[4,6,6] and r['dt_fs']==1 and r['segment_ps']==100 and r['seed'] in config['primary_seeds']]
                centers=np.array([r['center_mev'] for r in main]);center=float(np.mean(centers))
                blocks=[r for r in fits if r['repetitions']==[4,6,6] and r['dt_fs']==1 and r['seed'] in config['primary_seeds']]
                spread=max(max(abs(r['center_mev']-center) for r in blocks),max(r['energy_bin_mev']/2 for r in main))
                width=float(np.mean([r['fitted_intrinsic_FWHM_mev'] for r in main]))
                width_spread=max(abs(r['fitted_intrinsic_FWHM_mev']-width) for r in blocks)
                width_window=max(r['Hann_power_window_FWHM_mev'] for r in main)
                # All three original gates retained. Setting sensitivity is an
                # additional gate; temperature differences are recorded.
                controls=[r for r in fits if r['segment_ps']==100]
                setting_spread=max(abs(r['fitted_intrinsic_FWHM_mev']-width) for r in controls)
                resolved=width>2*width_window and max(width_spread,setting_spread)<.5*width and all(r['fit_relative_RMS_residual']<.1 for r in main)
                result['frequency_rows'].append({'temperature_k':T,'q_fractional':q.tolist(),'group':group,'matched_group':matched.tolist(),
                    'volume_angstrom3_per_unit':float(unit.get_volume()),'reference_0K_mev':float(np.mean(ref_e[group])),
                    'effective_energy_mev':predicted,'MD_center_mev':center,'MD_conservative_spread_mev':spread,
                    'effective_minus_MD_mev':predicted-center,'original_frequency_gate_pass':abs(predicted-center)<max(.5,2*spread),
                    'subspace_min_singular_overlap':float(np.min(np.linalg.svd(ref_v[:,group].conj().T@v[:,matched],compute_uv=False))),
                    'mode_polarization_weights_xyz':np.sum(abs(v[:,matched].reshape(-1,3,len(group)))**2,axis=(0,2)).tolist(),
                    'FWHM_mev':width,'FWHM_block_spread_mev':width_spread,'FWHM_setting_spread_mev':setting_spread,
                    'Hann_window_FWHM_mev':width_window,'linewidth_status':'RESOLVED_IN_TESTED_CLASSICAL_MODEL' if resolved else 'NOT_RESOLVED',
                    'lifetime_population_ps':float(constants.hbar/(width*1e-3*constants.elementary_charge)*1e12) if resolved else None,
                    'upper_bound_mev':None,'uncertainty_note':'conservative replica/segment/setting spread, not a confidence interval; controls at differing NVE temperature are not pure dt/size convergence'})
                json_write(output/'checkpoint.json',result)
                print('thermal mode',T,q.tolist(),group,'difference',predicted-center,'spread',spread,'width',width,'resolved',resolved,flush=True)
    result['status']='COMPLETED_DIAGNOSTICS'
    result['richer_model_extension_status']='DIRECT_MODE_DYNAMICS_REFERENCE_COMPUTED'
    result['effective_harmonic_status']='FAIL' # historical target failure cannot be erased by a wider uncertainty band
    result['linewidth_status']='NOT_RESOLVED' if any(r['linewidth_status']=='NOT_RESOLVED' for r in result['frequency_rows']) else 'RESOLVED_IN_TESTED_CLASSICAL_MODEL'
    result['resource_costs']=limits.costs();json_write(output/'results.json',result)


if __name__=='__main__':main()
