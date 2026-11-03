"""Stream only the static background needed by the complete pair hierarchy.

The hierarchy evolves its own elastic wave. The older general-purpose setup
also saved an incident trajectory that this workflow never used. This helper
retains the identical depth_field constructor without that extra allocation.
"""
from pathlib import Path
import json,time,resource
import numpy as np
from .workflow_io import ROOT,ResourceLimits,json_write,sha256
from .approximate_eels import ORIENTED,geometry
from yoshioka.lattice_dynamics import RealSpaceForceConstants
from yoshioka.free_propagation import TransverseGrid
from yoshioka.fourier_depth_fields import depth_field
from yoshioka.electron_phonon import LobatoSilicon


def background_only_setup(output,grid_size,steps,analytic=False):
    config=json.loads((output/'config.json').read_text())
    if config['reference_kind']!='static':raise ValueError('explicit static Taylor setup required')
    model=RealSpaceForceConstants.load(ORIENTED)
    positions,reference,length=geometry(model,config['selected_repeats'],config['padding_angstrom'])
    grid=TransverseGrid(model.cell_m[0,0],model.cell_m[1,1],grid_size,grid_size)
    cov=np.zeros((3,3));settings=dict(temperature_k=300,grid_unit=grid_size,slices=steps)
    if analytic:
        return config,settings,(positions,reference,length,cov,grid,None,None,
            {'representation':'analytic fields on demand; no sampled background or unused incident cache'})
    path=output/'background_only.npy';sidecar=output/'background_only.json'
    potential=LobatoSilicon.from_abtem_table()
    contract=dict(grid=grid.__dict__,steps=steps,length=length,covariance=cov.tolist(),
        geometry_sha256=sha256(ORIENTED),lobato_sha256=sha256(potential.source_path),
        kernels={p:sha256(ROOT/p) for p in ('yoshioka/fourier_depth_fields.py','yoshioka/electron_phonon.py','benchmarks/approximate_eels.py')})
    if sidecar.exists():
        record=json.loads(sidecar.read_text())
        if record['contract']!=contract or sha256(path)!=record['background_sha256']:
            raise ValueError('changed background-only cache')
        bg=np.load(path,mmap_mode='r')
    else:
        if path.exists():raise ValueError('uncommitted background exists; preserve and use another case identifier')
        limits=ResourceLimits(600,2);started=time.monotonic()
        # The Fourier constructor streams 256-beam slabs and evicts mapped
        # pages. The saved real background is a mapping, not an in-RAM volume.
        estimated=128*1024**2+16*256*steps*10+16*grid_size**2*12
        limits.check(estimated_bytes=estimated)
        bg,diagnostic=depth_field(positions,cov,grid,length,steps,path)
        limits.check()
        record=dict(contract=contract,background_sha256=sha256(path),reality=diagnostic,
            seconds=time.monotonic()-started,estimated_live_bytes=estimated,
            memory_limit_gib=2,peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            removed='unused incident-wave trajectory; elastic propagation remains in the main hierarchy')
        json_write(sidecar,record)
    return config,settings,(positions,reference,length,cov,grid,bg,None,record)
