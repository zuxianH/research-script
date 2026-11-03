"""Initialize an immutable zero-slice checkpoint before bounded invocations.

This also supports budgets shorter than the first16-slice commit interval. An
existing committed state is never replaced. Run this once before the fine
material runner; it is cheap and idempotent.
"""
import argparse,json
from math import comb
from pathlib import Path
import numpy as np
from .workflow_io import ROOT,fingerprint,json_write,sha256


def initialize(config):
    output=ROOT/config['output'];folder=output/'checkpoint';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'state.json';s=config['settings']
    if path.exists():
        state=json.loads(path.read_text())
        if state.get('completed_slices',0)>0 or state.get('wave_file'):return state
    count=comb(s['modes']+s['p'],s['p']);bytes_state=16*count*s['n']**2
    if bytes_state+350*1024**2>config['memory_limit_gib']*1024**3:raise MemoryError('initial channel state exceeds memory budget')
    contract={'settings':s,'force_constants_sha256':sha256(ROOT/'benchmark_results/si_lattice_dynamics/hiphive_model.npz'),
        'sources':{f:sha256(ROOT/f) for f in ('benchmarks/repair_material_checkpoint.py','yoshioka/checkpointed_channel_action.py','yoshioka/coordinate_channel_action.py','yoshioka/reciprocal_material.py')}}
    wave=np.zeros((count,s['n'],s['n']),complex);wave[0,s['n']//2,s['n']//2]=1
    file=folder/'wave_0.npy';np.save(file,wave,allow_pickle=False)
    state={'digest':fingerprint(contract),'contract':contract,'completed_slices':0,'total_slices':s['slices'],
        'cumulative_seconds':0.,'max_norm_drift':0.,'status':'INCOMPLETE','wave_file':file.name,'wave_sha256':sha256(file)}
    json_write(path,state);return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='benchmarks/configs/repair_material_checkpoint_20261102_v2.json')
    args=parser.parse_args();state=initialize(json.loads(Path(args.config).read_text()));print('initialized/retained',state['completed_slices'],'slices')


if __name__=='__main__':main()
