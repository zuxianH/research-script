#!/usr/bin/env python3
"""Independent read-only output audit; write a small verification record."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time
import resource
import subprocess

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import constants
from scipy.special import ndtr
from yoshioka.provenance import sha256,json_write,fingerprint


def verify(out):
    started=time.monotonic();rows=[]
    cfg=json.loads((out/'inputs/config.json').read_text())
    expected={'SNAPotential.snapcoeff':cfg['coefficients_sha256'],'SNAPotential.snapparam':cfg['parameters_sha256']}
    for name,digest in expected.items():
        assert sha256(out/'inputs'/name)==digest,(name,'input changed')
        assert sha256(Path('/home/zuxian/Documents/PhD/Yoshioka')/name)==digest,(name,'original changed')
    for folder in sorted((out/'cases').iterdir()):
        if not (folder/'exit.npz').exists():continue
        with np.load(folder/'exit.npz') as a, np.load(folder/'spectra.npz') as s:
            wave=a['exit_coefficients'];basis=a['basis'];weights=a['executed_initial_weights'];mask=a['detector_masks']
            norm=np.sum(abs(wave)**2,axis=(1,2,3))
            expected_probability=np.array([[[(abs(wave[i,f][m])**2).sum()*weights[i] for m in mask] for f in range(len(basis))] for i in range(len(wave))])
            energies=basis@a['mode_quanta_j']/(1e-3*constants.e)
            losses=energies[None,:]-energies[a['initial_indices'],None]
            axis=s['energy_mev'];spacing=(axis[-1]-axis[0])/(len(axis)-1)
            sigma=2/(2*np.sqrt(2*np.log(2)))
            fractions=ndtr((axis[-1]+spacing/2-losses)/sigma)-ndtr((axis[0]-spacing/2-losses)/sigma)
            expected_mass=np.einsum('ifd,if->d',expected_probability,fractions)
            actual_mass=s['inclusive_per_mev'].sum(axis=-1)*spacing
            mass_error=float(np.max(abs(actual_mass-expected_mass)))
            difference=float(np.max(abs(s['probability']-expected_probability)))
            assert np.max(abs(norm-1))<1e-9
            assert difference<1e-14
            assert mass_error<1e-12
            assert np.array_equal(mask[0],mask[1:].sum(axis=0))
            assert np.max(abs(losses-s['energy_loss_mev']))<1e-12
            assert np.array_equal(s['returning_state'],a['initial_indices'][:,None]==np.arange(len(basis))[None,:])
            p=int(basis.sum(axis=1).max());assert len(basis)==(p+1)*(p+2)//2
            if p==2:assert basis.tolist()==[[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]
            rows.append({'case':folder.name,'channels':len(basis),'initial_states':len(wave),
                'final_probability_norm_error':float(np.max(abs(norm-1))),
                'detector_channel_sum_error':difference,'spectral_area_absolute_error':mass_error,
                'native_total_detector_mass':s['native_mass'].tolist(),'display_window_mass':actual_mass.tolist()})
        state=json.loads((folder/'checkpoint/state.json').read_text())
        assert state['status']=='COMPLETE'
        assert sha256(folder/'checkpoint'/state['wave_file'])==state['wave_sha256']
        manifest=json.loads((folder/'manifest.json').read_text())
        source=manifest['source'];snapshot=out/'provenance/source'/source['source_sha256']
        assert all(sha256(snapshot/name)==digest for name,digest in source['files'].items())
        numerical={k:v for k,v in source['files'].items() if k.startswith('yoshioka/') or k.endswith('/run.py') or k.endswith('/config.json')}
        assert fingerprint(numerical)==manifest['contract']['source_sha256']
    preservation=json.loads((out/'provenance/preservation_before_navigation.json').read_text())
    old_results={p:d for p,d in preservation.items() if '/14_two_phonon_eels/' in p}
    assert all(sha256(p)==digest for p,digest in old_results.items())
    logs=(out/'logs/scientific_suite.log').read_text()
    count=re.search(r'Ran (\d+) tests in ([\d.]+)s',logs)
    assert count and re.search(r'^OK\s*$',logs,re.MULTILINE),'scientific suite did not pass'
    installed=json.loads((out/'provenance/installed_wheel_check.json').read_text())
    assert installed['passed']
    deps={}
    for name in ('numpy','scipy','matplotlib','ase','lammps','mpich','phonopy','hiphive','abtem','psutil'):
        dist=importlib.metadata.distribution(name)
        records=[p for p in dist.files or [] if str(p).endswith('.dist-info/RECORD')]
        deps[name]={'version':dist.version,'installation_record_sha256':sha256(dist.locate_file(records[0])) if records else None}
    dependencies={'python_executable':sys.executable,'python_executable_sha256':sha256(sys.executable),'python_version':sys.version,'distributions':deps,
        'meaning':'RECORD hashes pin installed file-digest lists; Lobato and potential data hashes are also recorded in the field/input manifests. No environment installation occurred.'}
    json_write(out/'provenance/dependencies.json',dependencies)
    reference=json.loads((out/'reference/processing.json').read_text())
    assert sha256(reference['array']['path'])==reference['array']['sha256']
    comparisons=json.loads((out/'checks/case_comparisons.json').read_text())
    result={'status':'SOFTWARE_AND_OUTPUT_AUDIT_PASS','finite_pilot_status':'NUMERICALLY_UNCONVERGED',
        'reason':'N33 versus N65 fails frozen support probability and complex-amplitude criteria',
        'cases':rows,'completed_case_count':len(rows),'scientific_suite':{'tests':int(count[1]),'seconds':float(count[2]),'passed':True},
        'wheel_check':installed,'original_historical_decision_hashes_preserved':old_results,
        'bulk_status':'NOT_ACCEPTED','corrected_complete_net_production_count':0,
        'source_revision':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'runtime_seconds':time.monotonic()-started,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    json_write(out/'verification.json',result)
    print(json.dumps({'status':result['status'],'cases':len(rows),'tests':int(count[1]),'runtime_seconds':result['runtime_seconds']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    verify(parser.parse_args().output)
