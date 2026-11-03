"""Actual-MOD checkpoint pilot and fresh-force checks at known failure times."""
import json
from pathlib import Path
import numpy as np
from scipy.constants import elementary_charge
from ase.io import read
from .workflow_io import ROOT,ResourceLimits,json_write
from yoshioka.molecular_dynamics_checked import checkpointed_md
from yoshioka.mod_forces import ModifiedTersoffForces


def main():
    output=ROOT/'benchmark_results/repair_20261102_v2'
    old=json.loads((output/'B11/results.json').read_text())
    settings=old['trajectories'][2]['settings']
    unit=read(ROOT/'benchmark_results/si_lattice_dynamics/equilibrium.extxyz')
    limits=ResourceLimits(300,2)
    folder,state=checkpointed_md(unit,ROOT/'Si.tersoff.mod',settings,output/'B11_checked/trajectories',limits)
    rows=[]
    if state['status']=='COMPLETE':
        atoms=unit.repeat(tuple(settings['repetitions']));u=np.load(folder/'displacements_m.npy',mmap_mode='r');f=np.load(folder/'forces_n.npy',mmap_mode='r')
        with ModifiedTersoffForces(atoms,ROOT/'Si.tersoff.mod') as calc:
            for index in (199,200,201,12399,12400,12401,17599,17600,19999):
                fresh=calc.evaluate(atoms.positions+u[index]*1e10)['forces_ev_per_angstrom']*elementary_charge/1e-10
                rows.append({'frame':index,'relative_force_error':float(np.linalg.norm(fresh-f[index])/max(np.linalg.norm(f[index]),1e-300))})
    passed=state['status']=='COMPLETE' and state['max_energy_drift_relative_to_thermal_kinetic_energy']<1e-3 and max((r['relative_force_error'] for r in rows),default=1)<1e-10
    result={'status':'PASS' if passed else 'FAIL','settings':settings,'checkpoint':str(folder/'state.json'),'trajectory_state':state,
        'fresh_force_checks':rows,'resource_costs':limits.costs(),'quality_criteria':{'max_energy_drift_over_thermal_KE':1e-3,'max_relative_fresh_force_error':1e-10},
        'original_failure':str(output/'MD_checkpoint_force_audit.json'),'scope':'checkpoint handling, not an anharmonic-model acceptance'}
    json_write(output/'MD_checked_pilot.json',result);print(json.dumps({k:v for k,v in result.items() if k!='trajectory_state'},indent=2))


if __name__=='__main__':main()
