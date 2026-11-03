"""Read-only numeric reference arrays and conservative spectral processing."""

from pathlib import Path
import hashlib

import numpy as np
from scipy.constants import Boltzmann, elementary_charge


def inspect_numeric_array(path, chunk_size=16):
    """Inspect bounded read-only chunks without interpreting axes or normalizing."""
    path = Path(path)
    data = np.load(path, mmap_mode='r', allow_pickle=False)
    if not np.issubdtype(data.dtype, np.number) or data.ndim < 1:
        raise ValueError('expected a numeric, non-scalar .npy array')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    counts = {'finite': 0, 'nonfinite': 0, 'negative_real': 0, 'nonzero_imaginary': 0}
    minimum, maximum = np.inf, -np.inf
    for start in range(0, len(data), chunk_size):
        values = data[start:start+chunk_size]
        finite = np.isfinite(values)
        counts['finite'] += int(np.sum(finite))
        counts['nonfinite'] += int(np.sum(~finite))
        counts['negative_real'] += int(np.sum(values.real < 0))
        counts['nonzero_imaginary'] += int(np.sum(values.imag != 0))
        if np.any(finite):
            minimum = min(minimum, float(np.min(values.real[finite])))
            maximum = max(maximum, float(np.max(values.real[finite])))
    return data, {'path': str(path.resolve()), 'sha256': digest.hexdigest(),
                  'shape': list(data.shape), 'dtype': str(data.dtype), 'bytes': path.stat().st_size,
                  'minimum_real': None if not np.isfinite(minimum) else minimum,
                  'maximum_real': None if not np.isfinite(maximum) else maximum, **counts}


def kubo_factor(energy_mev, temperature_k):
    """beta E/(1-exp(-beta E)), without applying it implicitly to any dataset."""
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError('positive finite temperature is required')
    x = np.asarray(energy_mev, dtype=float)*(1e-3*elementary_charge)/(Boltzmann*temperature_k)
    result = np.ones_like(x)
    positive, negative = x > 1e-7, x < -1e-7
    result[positive] = x[positive]/(-np.expm1(-x[positive]))
    result[negative] = x[negative]*np.exp(x[negative])/np.expm1(x[negative])
    small = ~(positive | negative)
    result[small] = 1+x[small]/2+x[small]**2/12
    return result


def conservative_rebin(weights, source_edges, target_edges, axis=-1, source_density=False):
    """Map bin weights by fractional overlap; cropped weight is never restored.

    A piecewise constant density within each source bin is the declared
    interpolation. Output is bin-integrated weight, irrespective of input units.
    Apply successively to rebin multiple momentum/energy axes.
    """
    source, target = np.asarray(source_edges), np.asarray(target_edges)
    for edges in (source, target):
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError('strictly increasing finite bin edges are required')
    values = np.moveaxis(np.asarray(weights), axis, -1)
    if values.shape[-1] != len(source)-1:
        raise ValueError('source axis length does not match bin edges')
    overlap = np.maximum(0., np.minimum(target[1:, None], source[None, 1:])-
                         np.maximum(target[:-1, None], source[None, :-1]))
    matrix = overlap if source_density else overlap/np.diff(source)[None]
    return np.moveaxis(values @ matrix.T, -1, axis)


def require_verified_metadata(metadata, keys):
    """Quantitative primary comparison requires traceable physical metadata."""
    missing = [key for key in keys if metadata.get(key, {}).get('status') not in ('VERIFIED', 'DERIVED_FROM_VERIFIED')]
    if missing:
        raise ValueError('comparison blocked by unverified metadata: '+', '.join(missing))


def gaussian_broaden_bin_weights(weights, bin_step, fwhm):
    """Forward Gaussian convolution with exact CDF bin weights and zero padding.

    Returns the full extended array and the number of extra bins on each side.
    Input bin weights are placed at bin centers. Cropped weight is never put
    back into the recorded window. Gaussian FWHM/bin step use the same units.
    """
    from scipy.signal import fftconvolve
    from scipy.special import ndtr
    if not np.isfinite(bin_step) or bin_step <= 0 or not np.isfinite(fwhm) or fwhm <= 0:
        raise ValueError('positive finite bin spacing and FWHM required')
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    reach = int(np.ceil(12*sigma/bin_step))+1
    offsets = np.arange(-reach, reach+1)*bin_step
    kernel = ndtr((offsets+bin_step/2)/sigma)-ndtr((offsets-bin_step/2)/sigma)
    return fftconvolve(np.asarray(weights), kernel, mode='full'), reach
