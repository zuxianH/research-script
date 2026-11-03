"""Finite-volume angular detector weights, independent of scattering intensity.

The source amplitudes already contain BZ weights. Replace a nodal detector
indicator by the fraction of its angular cubature cell lying in that detector.
No fitted scale, bin-area normalization of spectra or physical-mode deletion.
"""

import numpy as np


def positive_circle_rectangle(x0, x1, y0, y1, radius):
    x0, x1, y0, y1 = np.broadcast_arrays(x0, x1, y0, y1)
    x0, y0 = np.maximum(x0, 0), np.maximum(y0, 0)
    x1, y1 = np.maximum(x1, 0), np.maximum(y1, 0)
    limit = np.sqrt(np.maximum(0, radius**2-y0**2))
    right = np.minimum(x1, limit)
    full_end = np.sqrt(np.maximum(0, radius**2-y1**2))
    full_right = np.maximum(x0, np.minimum(right, full_end))
    full = np.maximum(0, full_right-x0)*np.maximum(0, y1-y0)
    left = np.maximum(x0, full_end)
    right = np.maximum(left, right)
    def primitive(x):
        x = np.clip(x, 0, radius)
        return .5*(x*np.sqrt(np.maximum(0, radius**2-x**2))+radius**2*np.arcsin(x/radius))
    partial = primitive(right)-primitive(left)-y0*(right-left)
    valid = (x1 > x0) & (y1 > y0) & (y0 < radius) & (x0 < radius)
    return np.where(valid, np.maximum(0, full+partial), 0)


def circle_rectangle_area(x0, x1, y0, y1, radius):
    """Exact area of a disk intersected with axis-aligned rectangles."""
    if radius <= 0:
        raise ValueError('positive circle radius required')
    total = 0.
    for lo_x, hi_x in ((np.maximum(x0, 0), np.maximum(x1, 0)),
                       (np.maximum(-np.asarray(x1), 0), np.maximum(-np.asarray(x0), 0))):
        for lo_y, hi_y in ((np.maximum(y0, 0), np.maximum(y1, 0)),
                           (np.maximum(-np.asarray(y1), 0), np.maximum(-np.asarray(y0), 0))):
            total = total+positive_circle_rectangle(lo_x, hi_x, lo_y, hi_y, radius)
    return total


def cell_fractions(centers_mrad, spacing_mrad):
    """All original B7 detectors, 8x8 pixels and half planes, with shared cells."""
    centers = np.asarray(centers_mrad)
    spacing = np.asarray(spacing_mrad)
    lower = centers-spacing/2
    upper = centers+spacing/2
    area = np.prod(spacing)
    disks = [circle_rectangle_area(lower[..., 0], upper[..., 0], lower[..., 1], upper[..., 1], radius)/area for radius in (5., 10., 20.)]
    detectors = np.stack((disks[2], disks[0], disks[1]-disks[0], disks[2]-disks[1]), axis=-1)
    pixels = np.zeros((*centers.shape[:-1], 64))
    edges = np.linspace(-20, 20, 9)
    for ix, iy in np.ndindex(8, 8):
        x0, x1 = np.maximum(lower[..., 0], edges[ix]), np.minimum(upper[..., 0], edges[ix+1])
        y0, y1 = np.maximum(lower[..., 1], edges[iy]), np.minimum(upper[..., 1], edges[iy+1])
        valid = (x1 > x0) & (y1 > y0)
        if np.any(valid):
            pixels[..., ix*8+iy][valid] = circle_rectangle_area(x0[valid], x1[valid], y0[valid], y1[valid], 20.)/area
    half_positive = circle_rectangle_area(np.maximum(lower[..., 0], 0), upper[..., 0], lower[..., 1], upper[..., 1], 20.)/area
    halves = np.stack((half_positive, disks[2]-half_positive), axis=-1)
    return detectors, pixels, halves


def angular_cell_spacing(primitive_reciprocal, electron_lengths_m, electron_k, mesh):
    """Spacing of the exact image of the integer BZ mesh modulo electron beams.

    For this Si primitive/electron-cell pair both integer projection gcds are 2
    for even meshes. Hence the angular grid spacing is 2*beam_spacing/mesh.
    The rectangular angular cells plus staggered longitudinal fibers partition
    the same normalized BZ measure. q weights are not multiplied by area again.
    """
    beam = 2*np.pi/np.asarray(electron_lengths_m)
    integer = np.asarray(primitive_reciprocal)[:, :2]/beam
    if not np.allclose(integer, np.rint(integer), atol=1e-8, rtol=0):
        raise ValueError('incommensurate primitive/electron reciprocal relation')
    integer = np.rint(integer).astype(int)
    divisors = np.array([np.gcd.reduce(np.r_[mesh, abs(integer[:, axis])]) for axis in range(2)])
    # Marginal gcds alone are insufficient if the image is not a product lattice.
    residues = np.arange(mesh)
    first, second = np.meshgrid(residues, residues)
    pairs = (first[..., None]*integer[0]+second[..., None]*integer[1]) % mesh
    if len(np.unique(pairs.reshape(-1, 2), axis=0)) != np.prod(mesh//divisors):
        raise ValueError('this mesh image is not the required rectangular angular lattice')
    return beam/electron_k*1000*divisors/mesh
