"""Predeclared absolute one-phonon observables shared by Benchmarks 6 and 7."""

import numpy as np
from scipy.constants import elementary_charge,hbar
from scipy.integrate import trapezoid


DETECTORS = {"full_0_20_mrad": (0,20),"central_0_5_mrad": (0,5),"annulus_5_10_mrad": (5,10),"annulus_10_20_mrad": (10,20)}
PEAK_GROUPS = {"acoustic_0_40_mev": (0,40),"middle_40_65_mev": (40,65),"optical_65_90_mev": (65,90),"above_90_mev": (90,np.inf)}


def detector_masks(angles_mrad):
    angles=np.asarray(angles_mrad)
    return {name:(angles>=lower)&(angles<=upper+1e-12 if upper==20 else angles<upper) for name,(lower,upper) in DETECTORS.items()}


def absolute_spectrum(energies_j,weights,energy_grid_mev,fwhm_mev=2):
    sigma=fwhm_mev/(2*np.sqrt(2*np.log(2)))
    energies=np.asarray(energies_j)/(1e-3*elementary_charge)
    output=np.zeros(len(energy_grid_mev))
    for energy,weight in zip(energies,weights):
        output+=weight*np.exp(-.5*((energy_grid_mev-energy)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    return output


def summarize(amplitudes,omega_rad_s,angles_mrad,energy_grid_mev=None,fwhm_mev=2,qxy_m_inv=None):
    energies=omega_rad_s*hbar
    masks=detector_masks(angles_mrad)
    probabilities=abs(amplitudes)**2
    values={}
    spectra={}
    mode_weights={}
    for name,mask in masks.items():
        mode=np.sum(probabilities[:,mask],axis=1)
        mode_weights[name]=mode
        values[name]=float(np.sum(mode))
        energies_mev=energies/(1e-3*elementary_charge)
        for group,(low,high) in PEAK_GROUPS.items():
            selected=(energies_mev>=low)&(energies_mev<high)
            values[f"{name}/{group}"]=float(np.sum(mode[selected]))
        if energy_grid_mev is not None:spectra[name]=absolute_spectrum(energies,mode,energy_grid_mev,fwhm_mev)
    # Fixed complementary momentum regions, without using crystal multiplicities.
    if qxy_m_inv is not None:
        values["positive_qx_half"] = float(np.sum(probabilities[:,qxy_m_inv[:,0]>=0]))
        values["negative_qx_half"] = float(np.sum(probabilities[:,qxy_m_inv[:,0]<0]))
        values["exact_forward_q_zero"] = float(np.sum(probabilities[:,np.linalg.norm(qxy_m_inv,axis=1)<1e-5]))
    return values,spectra,mode_weights


def compare_observables(coarse,fine,weak_scale=1e-18):
    rows=[]
    for name,value in fine.items():
        if value is None:continue
        absolute=abs(coarse[name]-value)
        rows.append({"observable":name,"coarse":coarse[name],"fine":value,"absolute_change":absolute,
                     "relative_change":absolute/abs(value) if abs(value)>0 else None,
                     "weak_signal":abs(value)<weak_scale,"acceptance_error":absolute/weak_scale if abs(value)<weak_scale else absolute/abs(value)})
    worst=max(rows,key=lambda row:row['acceptance_error'])
    return {"rows":rows,"maximum_acceptance_error":worst['acceptance_error'],"worst_observable":worst['observable'],
            "maximum_absolute_change":max(row['absolute_change'] for row in rows)}


def spectrum_change(coarse,fine,energy_grid_mev):
    rows=[]
    for name,spectrum in fine.items():
        denominator=float(trapezoid(spectrum,energy_grid_mev))
        absolute=float(trapezoid(abs(spectrum-coarse[name]),energy_grid_mev))
        rows.append({"detector":name,"L1_absolute":absolute,"fine_integrated_inelastic_weight":denominator,
                     "L1_relative":absolute/denominator if denominator>0 else None})
    worst=max(rows,key=lambda row:row['L1_relative'] or 0)
    return {"rows":rows,"maximum_relative_L1":worst['L1_relative'] or 0,"worst_detector":worst['detector']}
