"""Read-only physical-input and committed-result audit before resumption.

The immutable runner contracts retain their creation hashes. This additional
entry point checks live physical inputs rather than trusting recorded hashes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .workflow_io import ROOT,sha256,environment,json_write
from .approximate_eels import PRIMITIVE,ORIENTED
from .complete_leading_eels import DEFAULT,KERNELS,nodes,run
from yoshioka.electron_phonon import LobatoSilicon


def audit_electron_potential():
    evidence_path=ROOT/'benchmark_results/14_two_phonon_eels/complete_leading_20261102_v1/thermal_comparison/root_lobato_retrospective.json'
    evidence=json.loads(evidence_path.read_text())
    # This is an explicit retrospective audit: new pair creation records did
    # omit a byte hash. The prior protected resume hash and 56 creation-time
    # Si parameter records supply independent, preserved physical evidence.
    for key in ('historical_resume_record','prior_protection_anchor'):
        row=evidence[key]
        if sha256(ROOT/row['path'])!=row['sha256']:raise ValueError('Changed preserved electron-potential anchor')
    for row in evidence['creation_contracts']:
        path=ROOT/row['path']
        if sha256(path)!=row['sha256']:raise ValueError('Changed preserved potential-parameter contract')
        params=json.loads(path.read_text())['contract']['potential_parameters']
        if params!=evidence['current_Si_parameters']:raise ValueError('Historical Si parameters disagree')
    potential=LobatoSilicon.from_abtem_table()
    table_hash=sha256(potential.source_path)
    if table_hash!=evidence['historical_resume_record']['table_sha256']:
        raise ValueError('Changed Lobato electron-potential table')
    if not (np.array_equal(potential.a_angstrom,evidence['current_Si_parameters']['a']) and
            np.array_equal(potential.b_angstrom2,evidence['current_Si_parameters']['b'])):
        raise ValueError('Changed Si electron-potential parameters')
    return {'electron_Lobato_table_sha256':table_hash,'electron_potential_anchor_sha256':sha256(evidence_path),
        'new_pair_creation_table_byte_hash':'MISSING; retrospective protected prior-resume bytes and56creation-parameter contracts explicitly audited',
        'electron_potential_creation_parameter_contracts':len(evidence['creation_contracts'])}


def audit_inputs(output):
    output=Path(output);record=json.loads((output/'inputs.json').read_text())
    actual={'primitive_sha256':sha256(PRIMITIVE),'oriented_sha256':sha256(ORIENTED),
            'force_potential_sha256':sha256(ROOT/'Si.tersoff.mod'),
            'proposal_mass_sha256':sha256(output/'proposal_mass.npy')}
    for key,value in actual.items():
        if record[key]!=value:raise ValueError('Changed physical/proposal input: '+key)
    if environment()!=json.loads((output/'environment.json').read_text()):raise ValueError('Changed numerical environment')
    config=json.loads((output/'config.json').read_text());t=config['temperature_k']
    path=output/'covariance'/f'T{t:g}_mesh32.json';cov=json.loads(path.read_text())
    physical=Path(cov['physical_covariance_source'])
    if sha256(physical)!=cov['physical_covariance_sha256']:raise ValueError('Changed source physical covariance')
    source=json.loads(physical.read_text())
    if source['force_constants_sha256']!=actual['primitive_sha256']:raise ValueError('Physical covariance belongs to different force constants')
    declared=np.array(cov['covariance_m2'])
    required=np.zeros((3,3)) if config['reference_kind']=='static' else np.array(source['covariance_m2'])
    if not np.array_equal(declared,required):raise ValueError('Reference covariance inconsistent with declared elastic convention')
    alias=json.loads((output/'covariance/T300_mesh32.json').read_text())
    if not np.array_equal(np.array(alias['covariance_m2']),required):raise ValueError('Background reference covariance alias differs')
    actual.update(audit_electron_potential())
    actual['covariance_record_sha256']=sha256(path)
    return actual


def audit_results(output,scheme):
    actual=audit_inputs(output);rows=[]
    for folder in sorted((Path(output)/'pairs'/scheme).glob('batch*')):
        state=json.loads((folder/'state.json').read_text());c=state['contract'];count=state['completed_pairs']
        if c['inputs']!=json.loads((Path(output)/'inputs.json').read_text()):raise ValueError('Result recorded different inputs')
        if c['config']!=json.loads((Path(output)/'config.json').read_text()):raise ValueError('Changed live calculation configuration')
        if scheme!='radial':
            mass=Path(output)/('adapted_mass.npy' if scheme=='adapted' else 'proposal_mass.npy')
            if sha256(mass)!=c['sampling_mass_sha256']:raise ValueError('Changed full sampling-table hash')
        for name,expected in c['kernels'].items():
            if sha256(ROOT/name)!=expected:raise ValueError('Changed result source: '+name)
        unit=36 if scheme=='stratified' else 1
        power=int(np.ceil(np.log2(max(count/unit,1))))
        q,b,iw=nodes(Path(output),scheme,power,c['seed'])
        for i in range(count):
            path=folder/f'pair{i:06d}.npz'
            if sha256(path)!=state['files'][path.name]:raise ValueError('Changed committed result: '+str(path))
            with np.load(path) as d:
                if not (np.array_equal(q[i],d['q_fractional']) and np.array_equal(b[i],d['branches']) and np.array_equal(iw[i],d['inverse_marginals'])):
                    raise ValueError('Sample/checkpoint mismatch')
                coherent=d['direct_weights']+d['sequential_weights']+d['interference_weights']
                scale=abs(d['direct_weights'])+abs(d['sequential_weights'])+abs(d['interference_weights'])
                bound=64*np.finfo(float).eps*scale+2e-11*abs(d['weights'])
                if np.any(abs(d['weights']-coherent)>bound):raise ValueError('Direct/sequential interference identity failed')
                if not np.all(np.isfinite(d['weights'])):raise ValueError('Nonfinite calculated weight')
        rows.append({'batch':folder.name,'completed_pairs':count})
    return {'physical_inputs':actual,'scheme':scheme,'batches':rows,'all_checks_passed':True,'read_only':True,'guard_sha256':sha256(__file__)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT/'bulk_static_v2');p.add_argument('--scheme',default='born',choices=('born','adapted','stratified','radial'));p.add_argument('--resume-power',type=int);p.add_argument('--workers',type=int,default=2);p.add_argument('--write-audit',action='store_true');a=p.parse_args()
    result=audit_results(a.output,a.scheme)
    if a.write_audit:json_write(a.output/f'{a.scheme}_integrity_audit.json',result)
    print(result,flush=True)
    if a.resume_power is not None:
        run(a.output,a.scheme,a.resume_power,batches=8,workers=a.workers)
        result=audit_results(a.output,a.scheme)
        json_write(a.output/f'{a.scheme}_integrity_audit.json',result)

if __name__=='__main__':main()
