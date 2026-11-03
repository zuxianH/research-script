"""Atomic slice checkpoints for a bounded matrix-free propagation invocation."""
from pathlib import Path
import json,time
import numpy as np


def checkpointed_action(hamiltonian,folder,contract,limits,checkpoint_slices=16,initial_wave=None):
    from .provenance import fingerprint,json_write,sha256
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    h=hamiltonian
    initial=np.zeros(h.shape,complex) if initial_wave is None else np.array(initial_wave,dtype=complex,copy=True)
    if initial_wave is None:initial[0,h.n//2,h.n//2]=1
    if initial.shape!=h.shape or not np.all(np.isfinite(initial)) or abs(np.sum(abs(initial)**2)-1)>1e-12:
        raise ValueError('initial wave must have the Hamiltonian shape and unit probability')
    if initial_wave is not None:
        import hashlib
        contract=dict(contract,initial_wave_sha256=hashlib.sha256(initial.tobytes()).hexdigest())
    digest=fingerprint(contract);path=folder/'state.json'
    state=json.loads(path.read_text()) if path.exists() else {'digest':digest,'contract':contract,'completed_slices':0,'total_slices':h.steps,'cumulative_seconds':0.,'max_norm_drift':0.,'status':'INCOMPLETE'}
    if state['digest']!=digest or state['total_slices']!=h.steps:raise ValueError('channel checkpoint contract mismatch')
    if state['completed_slices']:
        if sha256(folder/state['wave_file'])!=state['wave_sha256']:raise ValueError('channel checkpoint wave checksum mismatch')
        wave=np.load(folder/state['wave_file'],allow_pickle=False)
        if wave.shape!=h.shape or wave.dtype!=np.complex128:raise ValueError('channel checkpoint shape/dtype mismatch')
    else:
        wave=initial.copy()
    if state['status']=='COMPLETE':return wave,state
    resumed=state['completed_slices'];started=time.monotonic();committed=started
    try:
        for i in range(resumed,h.steps):
            limits.check()
            wave=h.exponential_action(wave,i)
            state['max_norm_drift']=max(state['max_norm_drift'],abs(float(np.sum(abs(wave)**2))-1))
            if (i+1)%checkpoint_slices==0 or i+1==h.steps:
                name=f'wave_{i+1}.npy';temporary=folder/(name+'.tmp')
                with temporary.open('wb') as stream:np.save(stream,wave,allow_pickle=False)
                temporary.replace(folder/name);previous=state.get('wave_file')
                state.update(completed_slices=i+1,wave_file=name,wave_sha256=sha256(folder/name))
                state['cumulative_seconds']+=time.monotonic()-committed;json_write(path,state);committed=time.monotonic()
                if previous and previous!=name:(folder/previous).unlink(missing_ok=True)
    except (TimeoutError,MemoryError) as exc:
        # The exact committed wave/prefix pair is authoritative; at most one
        # bounded slice block is discarded. Never resume from an inexact state.
        if path.exists():
            state=json.loads(path.read_text())
            wave=np.load(folder/state['wave_file']) if state['completed_slices'] else initial.copy()
        else:
            wave=initial.copy()
            state['max_norm_drift']=0.
        state['stop_reason']=str(exc)
    state['latest_invocation_seconds']=time.monotonic()-started;state['resumed_from_slices']=resumed
    state['status']='COMPLETE' if state['completed_slices']==h.steps else 'INCOMPLETE'
    json_write(path,state);return wave,state
