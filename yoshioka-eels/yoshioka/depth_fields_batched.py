"""Larger transverse construction batches for the same checked source field.

Only an ephemeral working-field batching parameter changes. A field is still
complete and checked before a final channel is committed. No propagation,
quadrature, filtering, detector, or phase convention is changed.
"""
from pathlib import Path
from .dynamical_source import plane_field
from .depth_fields import convert_depth_field,depth_provider,incident_depth


def depth_field(positions,covariance,grid,length,steps,target,displacements=None):
    target=Path(target);temporary=target.with_suffix('.fourier.npy')
    fourier=plane_field(positions,covariance,grid,length,steps,temporary,displacements,chunk_size=512)
    result,diagnostic=convert_depth_field(fourier,grid,target)
    del fourier
    temporary.unlink();Path(str(temporary)+'.reality.json').unlink(missing_ok=True)
    return result,diagnostic
