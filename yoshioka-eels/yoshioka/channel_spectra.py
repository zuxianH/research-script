"""Absolute channel-resolved thermal transition lines and Gaussian display bins."""
import numpy as np
from scipy.constants import elementary_charge, hbar
from scipy.special import ndtr


def transition_lines(exit_waves, initial_indices, basis, mode_quanta_j, initial_weights, detector_masks):
    """Orthogonal final states and thermal initial states add as probabilities.

    Each complex exit coefficient already contains coherent path interference.
    Returning-state classification uses state identity, never energy alone.
    """
    waves=np.asarray(exit_waves);basis=np.asarray(basis)
    initial_indices=np.asarray(initial_indices,dtype=int)
    weights=np.asarray(initial_weights)
    masks=np.asarray(detector_masks,dtype=bool)
    if waves.shape[0]!=len(initial_indices) or waves.shape[1]!=len(basis) or len(weights)!=len(initial_indices):
        raise ValueError('inconsistent initial/final channel counts')
    if masks.shape[1:]!=waves.shape[2:] or np.any(weights<0) or weights.sum()>1+1e-12:
        raise ValueError('invalid detector masks or unnormalized thermal probabilities')
    energy=basis@np.asarray(mode_quanta_j)/(1e-3*elementary_charge)
    probability=np.einsum('ifyx,dyx->ifd',abs(waves)**2,masks)*weights[:,None,None]
    loss=energy[None,:]-energy[initial_indices,None]
    returning=initial_indices[:,None]==np.arange(len(basis))[None,:]
    return {'energy_loss_mev':loss,'probability':probability,'returning_state':returning,
            'initial_labels':basis[initial_indices],'final_labels':basis,'thermal_weights':weights}


def broaden_transition_lines(lines, energy_mev, fwhm_mev=2.):
    """Bin-integrated Gaussian weights; divide by bin width for density.

    The requested output interval crops tails without renormalizing them.
    One externally declared display factor may multiply a whole curve/components.
    """
    axis=np.asarray(energy_mev);step=axis[1]-axis[0]
    if step<=0 or not np.allclose(np.diff(axis),step): raise ValueError('uniform increasing energy axis required')
    sigma=fwhm_mev/(2*np.sqrt(2*np.log(2)))
    if sigma<=0: raise ValueError('positive Gaussian FWHM required')
    loss=lines['energy_loss_mev']
    kernel=ndtr((axis[None,None,:]+step/2-loss[:,:,None])/sigma)-ndtr((axis[None,None,:]-step/2-loss[:,:,None])/sigma)
    component=np.einsum('ifd,ife->ifde',lines['probability'],kernel)/step
    changed=~lines['returning_state']
    return {'components_per_mev':component,'inclusive_per_mev':component.sum(axis=(0,1)),
            'changed_per_mev':(component*changed[:,:,None,None]).sum(axis=(0,1)),
            'returning_per_mev':(component*lines['returning_state'][:,:,None,None]).sum(axis=(0,1)),
            'native_mass':lines['probability'].sum(axis=(0,1)),
            'display_window_mass':component.sum(axis=(0,1,3))*step,
            'energy_mev':axis,'bin_width_mev':step,'gaussian_fwhm_mev':fwhm_mev}


def angular_detector_masks(qxy, electron_wavenumber, bounds=((0,20),(0,5),(5,10),(10,20))):
    # Discrete plane-wave probability weights are already bin weights: no
    # momentum-area factor and no detector-by-detector renormalization.
    angle=np.arctan(np.linalg.norm(qxy,axis=-1)/electron_wavenumber)*1000
    masks=np.array([(angle>=lo)&(angle<hi) for lo,hi in bounds])
    return angle,masks
