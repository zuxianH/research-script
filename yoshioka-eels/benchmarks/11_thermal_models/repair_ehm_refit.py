"""Independent EHM learning curves using repaired, energy-checked MD data."""
import contextlib,json,time
from pathlib import Path
import numpy as np
from scipy import constants
from scipy.optimize import linear_sum_assignment
from ase.io import read
from hiphive import ClusterSpace
from .anharmonic_modes import fit_effective,validate_forces
from .repair_thermal import envelope_effective_samples
from .workflow_io import ROOT,ResourceLimits,json_write,sha256
from yoshioka.lattice_dynamics import RealSpaceForceConstants,real_space_from_supercell
from yoshioka.mode_spectra import project_normal_coordinates


def main():
    output=ROOT/'benchmark_results/repair_20261102_v2/B11_refit';output.mkdir(parents=True,exist_ok=True)
    checked=ROOT/'benchmark_results/repair_20261102_v2/B11_checked/results.json';data=json.loads(checked.read_text())
    if not data.get('trajectory_energy_quality_gate',{}).get('passed'):raise ValueError('repaired MD did not pass energy quality gate')
    limits=ResourceLimits(600,2);unit=read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
    reference=unit.repeat((4,6,6));harmonic=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    result={'status':'INCOMPLETE','scope':'same static quadratic model class, independent corrected trajectories, fixed volume; no fitting to EELS/TACAW',
        'source_data_sha256':sha256(checked),'potential_sha256':sha256(ROOT/'Si.tersoff.mod'),'fits':[],'sampling':[],
        'original_status':'FAIL','old_learning_grid':'benchmark_results/repair_20261102_v2/B11_learning/results.json',
        'force_acceptance_relative_RMS':.15,'source_sha256':{p:sha256(ROOT/p) for p in ('benchmarks/repair_ehm_refit.py','benchmarks/anharmonic_modes.py','yoshioka/molecular_dynamics_checked.py','yoshioka/lattice_dynamics.py')}}
    with (output/'hiphive.log').open('w') as log,contextlib.redirect_stdout(log):cs=ClusterSpace(unit,[5.])
    for T in (300,1000):
        training=next(r for r in data['trajectories'] if r['settings']['temperature_k']==T and r['settings']['seed']==723571 and r['settings']['repetitions']==[4,6,6])
        validation=next(r for r in data['trajectories'] if r['settings']['temperature_k']==T and r['settings']['seed']==837213 and r['settings']['repetitions']==[4,6,6])
        folder=Path(training['folder']);u=np.load(folder/'displacements_m.npy',mmap_mode='r');v=np.load(folder/'velocities_m_s.npy',mmap_mode='r')
        omega,eigen=harmonic.modes([0,0,0]);chosen=np.array([9,10,11]);mass=reference.get_masses()*constants.atomic_mass
        normal_v=project_normal_coordinates(v,reference.positions*1e-10,mass,harmonic,[0,0,0],eigen[:,chosen])
        normal_u=project_normal_coordinates(u,reference.positions*1e-10,mass,harmonic,[0,0,0],eigen[:,chosen])
        proxy=np.sum(abs(normal_v)**2+omega[chosen]**2*abs(normal_u)**2,axis=1)
        cadence=training['state']['sample_interval_s'];block=int(round(.2e-12/cadence));blocks=proxy[:len(proxy)//block*block].reshape(-1,block).mean(axis=1)
        ess=envelope_effective_samples(blocks);result['sampling'].append({'temperature_k':T,'modal_energy_proxy_ESS':ess,'block_ps':.2,
            'note':'initial-positive-pair autocorrelation of blocked |v|²+omega²|u|²; diagnostic effective-configuration proxy, not a confidence interval'})
        for count in (32,64,128):
            limits.check();started=time.monotonic();path=output/f'T{T}_n{count}.json'
            if path.exists():row=json.loads(path.read_text())
            else:
                with (output/'hiphive.log').open('a') as log,contextlib.redirect_stdout(log):
                    fcp,fitted=fit_effective(cs,reference,folder,count,0)
                    heldout=validate_forces(fcp,reference,Path(validation['folder']),64)
                    real,metadata=real_space_from_supercell(unit,(4,6,6),fcp.get_force_constants(reference).get_fc_array(2))
                model_path=output/f'T{T}_n{count}.npz';real.save(model_path);fcp.write(str(output/f'T{T}_n{count}.fcp'))
                rows=[]
                for target in [r for r in data['frequency_rows'] if r['temperature_k']==T]:
                    q=target['q_fractional'];_,old_eigen=harmonic.modes(q);w,new_eigen=real.modes(q)
                    _,assignment=linear_sum_assignment(-abs(old_eigen.conj().T@new_eigen)**2)
                    group=np.asarray(target['group']);matched=assignment[group]
                    overlap=float(np.min(np.linalg.svd(old_eigen[:,group].conj().T@new_eigen[:,matched],compute_uv=False)))
                    energy=float(np.mean(w[matched])*constants.hbar/(1e-3*constants.elementary_charge));difference=energy-target['MD_center_mev']
                    rows.append({**target,'effective_energy_mev':energy,'effective_minus_MD_mev':difference,
                        'original_frequency_gate_pass':bool(abs(difference)<max(.5,2*target['MD_conservative_spread_mev'])),
                        'refitted_subspace_min_overlap':overlap,'reference_basis_note':'MD subspace trace from independently checked reference; refitted subspace overlap recorded'})
                row={'temperature_k':T,'count':count,'cutoff_angstrom':5,'ridge':0,**fitted,**heldout,'model_metadata':metadata,
                    'model_file':str(model_path),'frequency_rows':rows,'force_gate_passed':heldout['force_relative_RMS_error']<.15,
                    'actual_training_temperature_k':training['state']['mean_temperature_k'],'actual_validation_temperature_k':validation['state']['mean_temperature_k'],
                    'effective_configuration_proxy':min(count,ess['effective_samples']),'elapsed_seconds':time.monotonic()-started}
                json_write(path,row)
            result['fits'].append(row);json_write(output/'checkpoint.json',result)
    selected=[r for r in result['fits'] if r['count']==128]
    accepted=all(r['force_gate_passed'] and all(s['original_frequency_gate_pass'] and s['refitted_subspace_min_overlap']>.9 and s.get('reference_shape_valid',False) for s in r['frequency_rows']) for r in selected)
    result['status']='PASS_TESTED_STATIC_MODEL' if accepted else 'FAIL';result['resource_costs']=limits.costs()
    json_write(output/'results.json',result)
    print(json.dumps({'status':result['status'],'heldout_force_errors':[(r['temperature_k'],r['count'],r['force_relative_RMS_error']) for r in result['fits']]},indent=2))


if __name__=='__main__':main()
