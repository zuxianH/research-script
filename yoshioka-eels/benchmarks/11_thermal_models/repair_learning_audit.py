"""Learning curves against a declared modal-energy independence proxy."""
import json
from pathlib import Path
import numpy as np
from scipy import constants
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.mode_spectra import project_normal_coordinates
from .repair_thermal import envelope_effective_samples
from .workflow_io import ROOT,json_write


def main():
    output=ROOT/'benchmark_results/repair_20261102_v2/B11_learning';output.mkdir(parents=True,exist_ok=True)
    old=json.loads((ROOT/'benchmark_results/anharmonic_20261102_v1/results.json').read_text())
    from ase.io import read
    unit=read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
    results={'status':'COMPLETED_DIAGNOSTICS','learning_curves':[],'all_fit_variations':old['fits'],
        'sampling_definition':'squared mass-weighted velocity PLUS omega_eff^2 squared displacement, summed over theGamma optical triplet;0.2ps block averaging removes fast phase oscillations. ESS is an energy-envelope proxy for serial dependence, not a fit-parameter confidence interval or a count of independent replicas.'}
    for T in (300,1000):
        for sc in ([3,4,4],[4,6,6]):
            trajectory=next(r for r in old['dynamics'] if r['settings']['temperature_k']==T and r['settings']['repetitions']==sc and r['settings']['seed']==932731)
            selected=next(r for r in old['fits'] if r['temperature_k']==T and r['supercell']==sc and r['count']==64 and r['cutoff_angstrom']==5 and r['ridge']==0)
            model=RealSpaceForceConstants.load(selected['model_file']);omega,v=model.modes(np.zeros(3))
            atoms=unit.repeat(tuple(sc));folder=Path(trajectory['folder']);masses=atoms.get_masses()*constants.atomic_mass
            d=np.load(folder/'displacements_m.npy',mmap_mode='r');vel=np.load(folder/'velocities_m_s.npy',mmap_mode='r')
            u=project_normal_coordinates(d,atoms.positions*1e-10,masses,model,np.zeros(3),v[:,9:12])
            qdot=project_normal_coordinates(vel,atoms.positions*1e-10,masses,model,np.zeros(3),v[:,9:12])
            energy=np.sum(abs(qdot)**2+omega[None,9:12]**2*abs(u)**2,axis=1)
            block=max(1,int(round(.2e-12/trajectory['state']['sample_interval_s'])))
            blocked=energy[:len(energy)//block*block].reshape(-1,block).mean(axis=1)
            ess=envelope_effective_samples(blocked)
            curves=[]
            for row in old['fits']:
                if row['temperature_k']!=T or row['supercell']!=sc or row['cutoff_angstrom']!=5 or row['ridge']!=0:continue
                sampled=energy[row['training_frames']]
                curves.append({'training_frames':row['count'],'independent_configuration_proxy':min(float(row['count']),ess['effective_samples']),
                    'subsample_envelope_ESS':envelope_effective_samples(sampled),'training_force_relative_RMS_error':row['training_force_relative_RMS_error'],
                    'heldout_force_relative_RMS_error':row['force_relative_RMS_error']})
            results['learning_curves'].append({'temperature_k':T,'supercell':sc,'actual_training_temperature_k':trajectory['state']['mean_temperature_k'],
                'duration_ps':20,'envelope_block_ps':block*trajectory['state']['sample_interval_s']*1e12,'energy_envelope_ESS':ess,'rows':curves})
    json_write(output/'results.json',results)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for row in results['learning_curves']:
        label=f"{row['temperature_k']}K, {row['supercell']}"
        x=[r['training_frames'] for r in row['rows']];y=[r['heldout_force_relative_RMS_error'] for r in row['rows']]
        axes[0].plot(x,y,'.-',label=label)
        axes[1].plot([r['independent_configuration_proxy'] for r in row['rows']],y,'.-',label=label)
    axes[0].set(xlabel='training configurations',ylabel='independent-seed relative force RMS error')
    axes[1].set(xlabel='modal-energy independence proxy',ylabel='independent-seed relative force RMS error')
    for ax in axes:ax.axhline(.15,color='k',ls='--');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(output/'force_learning.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
