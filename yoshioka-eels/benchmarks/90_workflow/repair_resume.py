"""Resume a selected repair task under the audited current provenance profile.

The profile is adopted after this task's explicit audit. It does not replace or
backdate any per-calculation checkpoint contract. No calculation runs by default.
"""
import argparse,json,runpy,sys
from pathlib import Path
from .workflow_io import ROOT,fingerprint,json_write,sha256
from .guarded_run import profile

OUTPUT=ROOT/'benchmark_results/repair_20261102_v2'
CONFIG='benchmarks/configs/'
TASKS={
    'b7':('benchmarks.repair_b7',['--config',CONFIG+'repair_b7_20261102_v2.json']),
    'b7-joint':('benchmarks.repair_b7_joint',['--config',CONFIG+'repair_b7_joint_20261102_v2.json']),
    'b7-joint-evaluate':('benchmarks.repair_b7_joint_finalize',[]),
    '9a-electron':('benchmarks.repair_9a_fourier_depth',['--config',CONFIG+'repair_9a_fft_grid_20261102_v2.json']),
    '9a-joint':('benchmarks.repair_9a_fourier_depth',['--config',CONFIG+'repair_9a_fft_joint_20261102_v2.json']),
    '9a-phonon-z':('benchmarks.repair_9a_bloch',['--config',CONFIG+'repair_9a_bloch_bath_20261102_v2.json']),
    '9a-phonon-xy':('benchmarks.repair_9a_bloch_mesh',['--config',CONFIG+'repair_9a_bloch_transverse_20261102_v2.json']),
    '9a-phonon-xyz':('benchmarks.repair_9a_bloch_direct',['--config',CONFIG+'repair_9a_bloch_fullmesh_20261102_v2.json']),
    '9a-validate':('benchmarks.repair_bloch_validation',[]),
    '9a-source-audit':('benchmarks.repair_source_local_audit',[]),
    '9a-evaluate':('benchmarks.repair_bloch_analysis',[]),
    '9b-validate':('benchmarks.repair_matrix_free',[]),
    '9b-material':('benchmarks.repair_material_repeated',['--config',CONFIG+'repair_material_20261102_v2.json']),
    '9b-gamma6':('benchmarks.repair_material_repeated',['--config',CONFIG+'repair_material_gamma6_20261102_v2.json']),
    '9b-electron':('benchmarks.repair_material_coordinate',['--config',CONFIG+'repair_material_coordinate_electron_20261102_v2.json']),
    '9b-cutoff':('benchmarks.repair_material_coordinate',['--config',CONFIG+'repair_material_coordinate_cutoff_slice_quadrature_20261102_v2.json']),
    '9b-electron49':('benchmarks.repair_material_coordinate',['--config',CONFIG+'repair_material_coordinate_electron49_20261102_v2.json']),
    '9b-fine-init':('benchmarks.repair_material_checkpoint_init',['--config',CONFIG+'repair_material_checkpoint_20261102_v2.json']),
    '9b-fine':('benchmarks.repair_material_checkpoint',['--config',CONFIG+'repair_material_checkpoint_20261102_v2.json']),
    '9b-orders':('benchmarks.repair_material_orders',[]),
    '9b-order24-audit':('benchmarks.repair_coordinate_order_audit',['--order','24']),
    '9b-order32-audit':('benchmarks.repair_coordinate_order_audit',['--order','32']),
    '10-metadata':('benchmarks.repair_metadata',[]),
    '11-dynamics':('benchmarks.repair_thermal_checked',['--config',CONFIG+'repair_thermal_checked_20261102_v2.json']),
    '11-audit':('benchmarks.repair_thermal_final_audit',[]),
    '11-refit':('benchmarks.repair_ehm_refit',[]),
    '11-learning':('benchmarks.repair_learning_audit',[]),
    'figures':('benchmarks.repair_analysis',[]),
}


def current_profile(module):
    result=profile(module)
    # A relocated virtualenv path is provenance, not a physical constant. Guard
    # the actual table bytes and stable package-relative name, not its mount.
    table=result['Lobato_Si_actual_table'];location=Path(table['source_path'])
    result['Lobato_Si_table_sha256']=sha256(location)
    table['source_path']='abtem/parametrizations/data/'+location.name
    result['resume_guard_sha256']=sha256(__file__)
    files=sorted({arguments[arguments.index('--config')+1] for target,arguments in TASKS.values() if target==module and '--config' in arguments})
    result['configurations']={name:{k:v for k,v in json.loads((ROOT/name).read_text()).items() if k not in ('runtime_limit_seconds','memory_limit_gib')} for name in files}
    result['physical_input_sha256']={name:sha256(ROOT/name) for name in ('Si.tersoff.mod','benchmark_results/si_lattice_dynamics/hiphive_model.npz','benchmark_results/si_lattice_dynamics/equilibrium.extxyz')}
    if module in ('benchmarks.repair_ehm_refit','benchmarks.repair_thermal_final_audit'):
        source=OUTPUT/'B11_checked/results.json'
        if not source.exists():raise ValueError('run the corrected11-dynamics task before this data-dependent task')
        data=json.loads(source.read_text())
        result['corrected_MD_input_contracts']=[row['state']['digest'] for row in data['trajectories']]
        result['corrected_MD_energy_quality']=data['trajectory_energy_quality_gate']
    return result


def establish():
    # Store current transitive source/actual constants and table fingerprints.
    # Calculations already have their own older, authoritative inner contracts.
    modules=sorted({entry[0] for entry in TASKS.values()})
    records={module:current_profile(module) for module in modules}
    row={'profiles':records,'digests':{k:fingerprint(v) for k,v in records.items()},
        'scope':'prospective resume guard after explicit current audit; not a retroactive creation record',
        'FFT_workers_exception':'four workers in depth-major Fourier source routines; BLAS/OMP remain one',
        'inner_contracts':'unchanged original creation/source contracts remain authoritative; compatible prefix adoption recorded separately'}
    json_write(OUTPUT/'resume_profiles.json',row)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',nargs='?',default='status',choices=['status','audit-current',*TASKS])
    args=parser.parse_args()
    if args.task=='audit-current':establish();return
    if args.task=='status':
        path=OUTPUT/'results.json'
        print(json.dumps(json.loads(path.read_text()).get('statuses',{}) if path.exists() else {'status':'report not generated'},indent=2));return
    module,arguments=TASKS[args.task];path=OUTPUT/'resume_profiles.json'
    if not path.exists():raise ValueError('current audit profile missing; establish only after checking physical/cache provenance')
    audit=json.loads(path.read_text());current=current_profile(module)
    if fingerprint(current)!=audit['digests'][module]:raise ValueError('source/constants/table/environment changed; preserve these results and use a new audited output directory')
    sys.argv=[module,*arguments];runpy.run_module(module,run_name='__main__')


if __name__=='__main__':main()
