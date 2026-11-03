"""Explicit probability measures for bulk quadrature and detector integration.

The BZ measure is d^3q / V_BZ (dimensionless). A quadrature weight is not
a momentum density. Physical point masses, when present, are added once and
are never multiplied by detector area.
"""

import numpy as np


def uniform_bz_nodes(mesh, start=0, stop=None, shift=(0., 0., 0.)):
    """Deterministic chunks of a periodic midpoint/vertex cubature, no extras."""
    if not isinstance(mesh, int) or mesh < 1:
        raise ValueError("mesh must be a positive integer")
    shift = np.asarray(shift, dtype=float)
    if shift.shape != (3,) or not np.all(np.isfinite(shift)):
        raise ValueError("shift must have three finite components in mesh cells")
    stop = mesh**3 if stop is None else stop
    if not 0 <= start <= stop <= mesh**3:
        raise ValueError("invalid quadrature index interval")
    indices = np.array(np.unravel_index(np.arange(start, stop), (mesh,)*3)).T
    nodes = (indices + shift) / mesh
    nodes %= 1.
    nodes[nodes > .5] -= 1.
    return nodes, np.full(stop-start, 1./mesh**3)


def detector_measure(density, area_weights, inside, point_weights=(), point_inside=()):
    """Integrate a density and, separately, genuine discrete probability masses."""
    density, area_weights, inside = np.broadcast_arrays(density, area_weights, inside)
    points, selected = np.broadcast_arrays(point_weights, point_inside)
    if np.any(area_weights < 0):
        raise ValueError("negative area weights")
    return float(np.sum(density * area_weights * inside) + np.sum(points * selected))


def physical_observables(values):
    """B7's exact-forward cubature sample is diagnostic, already inside bins."""
    return {key: value for key, value in values.items() if key != 'exact_forward_q_zero'}
