"""Separate checkpoint corruption, clean MD reference, and EHM inadequacy."""
import json
from pathlib import Path
import numpy as np
from scipy import constants
from ase.io import read
from .workflow_io import ROOT,json_write,sha256
from .repair_thermal import envelope_effective_samples
from yoshioka.mode_spectra import project_normal_coordinates
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from .anharmonic_validation import groups_at_q


def main():
    output=ROOT/'benchmark_results/repair_20261102_v2';checked=json.loads((output/'B11_checked/results.json').read_text())
    evidence=output/'reused_evidence' if (output/'reused_evidence').exists() else output
    prior=json.loads((evidence/'B11_audit/results.json').read_text());old=json.loads((ROOT/'benchmark_results/anharmonic_20261102_v1/results.json').read_text())
    initial=json.loads((evidence/'B11/results.json').read_text());unit=read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz');atoms=unit.repeat((4,6,6))
    result={**prior,'mode_covariance':[],'reused_clean_control_sha256':sha256(evidence/'B11_audit/results.json'),
        'reused_control_scope':'low-amplitude MD and paired fresh MOD force evaluations; paired data came from clean original20ps trajectories, not the corrupted new100ps case',
        'corrected_reference_sha256':sha256(output/'B11_checked/results.json')}
    for T in (300,1000):
        selected=next(r for r in old['fits'] if r['temperature_k']==T and r['supercell']==[4,6,6] and r['count']==64 and r['cutoff_angstrom']==5 and r['ridge']==0)
        model=RealSpaceForceConstants.load(selected['model_file']);tr=next(r for r in checked['trajectories'] if r['settings']['temperature_k']==T and r['settings']['seed']==723571 and r['settings']['repetitions']==[4,6,6])
        folder=Path(tr['folder']);u=np.load(folder/'displacements_m.npy',mmap_mode='r');v=np.load(folder/'velocities_m_s.npy',mmap_mode='r')
        E,eigen,groups=groups_at_q(model,np.zeros(3));omega=E*(1e-3*constants.elementary_charge)/constants.hbar
        for group in groups:
            normal_u=project_normal_coordinates(u,atoms.positions*1e-10,atoms.get_masses()*constants.atomic_mass,model,[0,0,0],eigen[:,group])
            normal_v=project_normal_coordinates(v,atoms.positions*1e-10,atoms.get_masses()*constants.atomic_mass,model,[0,0,0],eigen[:,group])
            var=np.var(normal_u,axis=0);kin=np.var(normal_v,axis=0);kBT=constants.Boltzmann*tr['state']['mean_temperature_k']
            result['mode_covariance'].append({'temperature_k':T,'actual_temperature_k':tr['state']['mean_temperature_k'],'group':group,
                'displacement_variance_over_EHM_prediction':(var/(kBT/omega[group]**2)).tolist(),'velocity_variance_over_equipartition':(kin/kBT).tolist(),
                'envelope_sampling':envelope_effective_samples(np.sum(abs(normal_v)**2,axis=1)),
                'interpretation':'corrected100ps replica, separate covariance diagnostic, not a confidence interval'})
    folder=output/'B11_final_audit';folder.mkdir(exist_ok=True);json_write(folder/'results.json',result)
    def rows(data,key):
        return [{'settings':r['settings'],'mean_temperature_k':r['state'].get('mean_temperature_k'),
            'energy_drift_over_thermal_KE':r['state'].get('max_energy_drift_relative_to_thermal_kinetic_energy'),
            'quality_passed':r['state'].get('max_energy_drift_relative_to_thermal_kinetic_energy',1)<1e-3} for r in data[key]]
    quality={'status':'REPAIRED_CHECKPOINT_DRIVER','old20ps':rows(old,'dynamics'),'initial100ps':rows(initial,'trajectories'),'corrected100ps':rows(checked,'trajectories'),
        'mechanism':'write_restart remaps atoms; continuing run pre no without force/neighbor reinitialization can use stale neighbors',
        'old_bad_force_audit':str(output/'MD_checkpoint_force_audit.json'),'corrected_pilot':str(output/'MD_checked_pilot.json'),
        'original20ps_status':'all stored energy checks pass; original EHM failure is retained',
        'initial100ps_reference_status':'superseded; three trajectories fail new numerical energy gate; do not use initial2.6095meV as final reference',
        'corrected100ps_reference_status':'energy quality passed; individual shape/width/statistical gates still apply',
        'selected_refit':str(output/'B11_refit/results.json')}
    json_write(output/'MD_quality_status.json',quality)
    text=['# Corrected trajectory controls','','Energy drift is max|E(t)−E(0)|/[1.5(N−1)kB mean(T)], not divided by the large negative binding energy. The numerical reference gate is10⁻³. All production is NVE after NVT equilibration at fixed volume. Actual temperatures, not target labels, determine physical comparability.','',
        '| Target K | Atoms | dt(fs) | Seed | NVT damping(fs) | Actual mean K | Std K | Max energy drift / thermal KE |','|---|---|---|---|---|---|---|---|']
    for r in checked['trajectories']:
        s=r['settings'];v=r['state'];text.append(f"| {s['temperature_k']} | {4*np.prod(s['repetitions'])} | {s['timestep_fs']} | {s['seed']} | {s['thermostat_time_fs']} | {v['mean_temperature_k']:.8g} | {v['temperature_std_k']:.8g} | {v['max_energy_drift_relative_to_thermal_kinetic_energy']:.8g} |")
    text+=['','The two corrected primary1000K replicas are closely temperature matched. The other controls still have temperature differences; they do not provide a pure equal-temperature size/timestep extrapolation. Low-amplitude and mirror-force controls are reused with their original creation records. The old20ps energy checks pass; the corrupted100ps extension is preserved but superseded.']
    (output/'trajectory_controls.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
