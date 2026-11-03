"""Implementation, force-parity and sampling audits independent of EELS peaks."""
import json
from pathlib import Path
import numpy as np
from scipy import constants

from yoshioka.molecular_dynamics import checkpointed_md
from yoshioka.mod_forces import ModifiedTersoffForces
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.mode_spectra import project_normal_coordinates,periodogram_modes,fit_windowed_mode
from .anharmonic_validation import groups_at_q
from .repair_thermal import envelope_effective_samples
from .workflow_io import ROOT,ResourceLimits,json_write,sha256


def main():
    output=ROOT/'benchmark_results/repair_20261102_v2/B11_audit';output.mkdir(parents=True,exist_ok=True)
    from ase.io import read
    unit=read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
    old=json.loads((ROOT/'benchmark_results/anharmonic_20261102_v1/results.json').read_text())
    long=json.loads((ROOT/'benchmark_results/repair_20261102_v2/B11/results.json').read_text())
    limits=ResourceLimits(600,2,100)
    reference=RealSpaceForceConstants.load(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz')
    result={'status':'INCOMPLETE','parity_force_tests':[],'mode_covariance':[],'implementation_controls':[],
        'audit':{'units':'stored d m, F N, v m/s; eigen omega rad/s, E=hbar omega; PSD E=h f. Mass28.085u converted by SciPy atomic_mass; no extra2pi',
                 'atom_order':'fixed LAMMPS atom IDs, sort0,0; primitive mapping checked to1e-7 fractional tolerance by projection routine',
                 'wrapping':'minimum-image about same equilibrium atoms, COM removed; compare displacement bound to half cell',
                 'volume':'identical fixed 0K volume for all same-volume branches; no thermal-expansion shift included',
                 'mode_tracking':'eigenvector Hungarian assignment and full degenerate subspaces; overlap recorded per mode row',
                 'training_validation':'whole independent seed trajectories; no neighboring-frame train/test split; envelope ESS reported separately',
                 'range_regularization':'previous36fit variants and blocked fits retained; compare their frequency variation with MD discrepancy, not only heldout forces',
                 'MD_bias':'new100ps NVE; previous1/.5fs and192/576atoms plus new controls. Different actual temperatures confound a pure dt/size extrapolation',
                 'model':'static quadratic force law is odd about the reference; actual force parity test isolates a component it cannot reproduce on a paired set'},
        'potential_sha256':sha256(ROOT/'Si.tersoff.mod'),'original_model_status':'FAIL'}
    # Low-amplitude *actual MOD dynamics*, not another copy of the FC formula.
    settings={'repetitions':[3,4,4],'temperature_k':.1,'seed':542917,'timestep_fs':.5,'sample_stride':10,
              'equilibration_steps':10000,'production_steps':40000,'thermostat_time_fs':100}
    folder,state=checkpointed_md(unit,ROOT/'Si.tersoff.mod',settings,output/'low_temperature',limits)
    result['low_amplitude_MD']={'folder':str(folder),'state':state,'settings':settings}
    atoms=unit.repeat(tuple(settings['repetitions']))
    velocity=np.load(folder/'velocities_m_s.npy',mmap_mode='r')
    for q in (np.array([0.,0.,0.]),np.array([0.,.5,0.])):
        energies,vectors,groups=groups_at_q(reference,q)
        for group in groups:
            projected=project_normal_coordinates(velocity,atoms.positions*1e-10,atoms.get_masses()*constants.atomic_mass,reference,q,vectors[:,group])
            E,power=periodogram_modes(projected,state['sample_interval_s'])
            predicted=float(np.mean(energies[group]));fit=fit_windowed_mode(E,power,predicted,20)
            result['implementation_controls'].append({'q':q.tolist(),'group':group,'FC_energy_mev':predicted,'MD_energy_mev':fit['center_mev'],
                'difference_mev':fit['center_mev']-predicted,'fit':fit})
    # A paired mirror sample exposes even forces, independent of any fit.
    for T in (300,1000):
        trajectory=next(r for r in old['dynamics'] if r['settings']['temperature_k']==T and r['settings']['repetitions']==[4,6,6] and r['settings']['seed']==481723)
        folder=Path(trajectory['folder']);atoms=unit.repeat((4,6,6));d=np.load(folder/'displacements_m.npy',mmap_mode='r')
        frames=np.linspace(0,len(d)-1,8,dtype=int);even2=total2=0.;perframe=[]
        with ModifiedTersoffForces(atoms,ROOT/'Si.tersoff.mod') as calc:
            for index in frames:
                displacement=np.asarray(d[index])*1e10
                plus=calc.evaluate(atoms.positions+displacement)['forces_ev_per_angstrom']
                minus=calc.evaluate(atoms.positions-displacement)['forces_ev_per_angstrom']
                even=.5*(plus+minus);a=float(2*np.sum(even*even));b=float(np.sum(plus*plus+minus*minus))
                even2+=a;total2+=b;perframe.append({'frame':int(index),'paired_odd_force_model_error_lower_bound':float(np.sqrt(a/b))})
        result['parity_force_tests'].append({'temperature_k':T,'frames':perframe,'paired_odd_force_model_error_lower_bound':float(np.sqrt(even2/total2)),
            'meaning':'minimum relative RMS error for any odd force model on this explicitly paired +u/-u set; not asserted as a bound on the unpaired equilibrium distribution'})
        selected=next(r for r in old['fits'] if r['temperature_k']==T and r['supercell']==[4,6,6] and r['count']==64 and r['cutoff_angstrom']==5 and r['ridge']==0)
        model=RealSpaceForceConstants.load(selected['model_file'])
        trajectory=next(r for r in long['trajectories'] if r['settings']['temperature_k']==T and r['settings']['seed']==723571 and r['settings']['repetitions']==[4,6,6])
        folder=Path(trajectory['folder']);positions=atoms.positions*1e-10;masses=atoms.get_masses()*constants.atomic_mass
        displacements=np.load(folder/'displacements_m.npy',mmap_mode='r');velocities=np.load(folder/'velocities_m_s.npy',mmap_mode='r')
        E,v,groups=groups_at_q(model,np.zeros(3));omega=E*(1e-3*constants.elementary_charge)/constants.hbar
        for group in groups:
            u=project_normal_coordinates(displacements,positions,masses,model,np.zeros(3),v[:,group])
            vel=project_normal_coordinates(velocities,positions,masses,model,np.zeros(3),v[:,group])
            var=np.mean(abs(u-np.mean(u,axis=0))**2,axis=0);kin=np.mean(abs(vel-np.mean(vel,axis=0))**2,axis=0)
            kBT=constants.Boltzmann*trajectory['state']['mean_temperature_k']
            ess=envelope_effective_samples(np.sum(abs(vel)**2,axis=1))
            result['mode_covariance'].append({'temperature_k':T,'actual_temperature_k':trajectory['state']['mean_temperature_k'],'group':group,
                'displacement_variance_over_EHM_prediction':(var/(kBT/omega[group]**2)).tolist(),
                'velocity_variance_over_equipartition':(kin/kBT).tolist(),'envelope_sampling':ess,
                'interpretation':'covariance prediction is independent of spectral center; one100ps replica, not a confidence interval'})
        result['audit'][f'wrapping_bound_T{T}']={'maximum_displacement_angstrom':trajectory['state']['max_displacement_angstrom'],
            'minimum_half_cell_angstrom':float(np.min(atoms.cell.lengths())/2)}
    result['maximum_low_amplitude_frequency_discrepancy_mev']=max(abs(r['difference_mev']) for r in result['implementation_controls'])
    result['low_amplitude_units_mapping_control_passed']=result['maximum_low_amplitude_frequency_discrepancy_mev']<.1
    result['status']='COMPLETED_DIAGNOSTICS';result['resource_costs']=limits.costs();json_write(output/'results.json',result)
    print('thermal audit',result['maximum_low_amplitude_frequency_discrepancy_mev'],result['parity_force_tests'],flush=True)


if __name__=='__main__':main()
