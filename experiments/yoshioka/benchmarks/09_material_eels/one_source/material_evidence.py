"""B9 absolute observables, separate refinement gates and representation costs."""

import json
from math import comb
from pathlib import Path

import numpy as np
from scipy import constants
from scipy.integrate import trapezoid

from yoshioka.free_propagation import electron_wavenumber
from .eels_observables import DETECTORS, PEAK_GROUPS, absolute_spectrum, detector_masks
from .workflow_io import ROOT, json_write, sha256


def relative_difference(a, b):
    return {'coarse': float(a), 'fine': float(b), 'absolute_change': float(abs(a-b)),
            'relative_change': float(abs(a-b)/abs(b)) if abs(b) > 1e-18 else None,
            'acceptance_error':float(abs(a-b)/max(abs(b),1e-18))}


def main():
    output = ROOT/'benchmark_results/material_evidence_20261102_v1'
    output.mkdir(parents=True, exist_ok=True)
    path = ROOT/'benchmark_results/dynamical_si_20261102_v1/results.json'
    source = json.loads(path.read_text())
    cases = [c for c in source['cases'] if c['status'] == 'COMPLETE']
    padding_path = ROOT/'benchmark_results/dynamical_si_padding_20261102_v1/results.json'
    if padding_path.exists():
        padding = json.loads(padding_path.read_text())
        cases += [c for c in padding['cases'] if c['status'] == 'COMPLETE']
    result = {'benchmark': '9A material observables and 9B feasibility audit', 'status': 'INCOMPLETE',
        '9A_status': 'INCOMPLETE', '9B_status': 'INCOMPLETE', 'source_results_sha256': sha256(path),
        'supplemental_padding_results_sha256':sha256(padding_path) if padding_path.exists() else None,
        'cases': [], 'refinements': [], 'thermal_absolute_cutoff_diagnostics': [],
        '9B_feasibility': source['9B_feasibility'], 'normalization': 'Fourier-mode weights, dimensionless; no angular-area multiplier. Elastic exit norm and undepleted inelastic source weights are reported separately.',
        'fixed_specimen_policy': 'phonon_z changes only the periodic bath embedded in the same atom slab; actual atom thickness is fixed when bath is refined',
        'transverse_extent': 'The transverse cell is a periodic compatibility cell, not a vacuum extent. Replicating the identical potential/displacement pattern into a larger electron cell preserves the old Fourier subspace and its probabilities: atomic sums and incident area both grow by the same factor, while new forbidden beams cancel. A regression checks this normalization. Adding independent transverse phonon momenta is a separate, still-unconverged physical bath refinement.',
        'boundaries': 'finite z slab plus vacuum; transverse 2x2 periodic electron cell. Finite bath momenta are compatible with this cell. Not a converged bulk-BZ model.',
        'detailed_balance': {
            'identity': 'S_F(-Q,-E)=exp(-beta E) S_F(Q,E) for F(Q)^dagger=F(-Q), from interchanging initial/final thermal states',
            'geometry': 'The distorted-wave transition operator includes incoming/outgoing elastic propagation. Its adjoint reverses those paths. A same-pixel gain/loss relation is not imposed on the forward illumination calculation; a reciprocal entrance/exit geometry is required.',
            'tested': 'The harmonic linear-source factors nB/(nB+1)=exp(-beta hbar omega) are tested separately; this does not establish same-pixel reciprocity.'},
        'missing_convergence': ['two successive accepted material grid/slice refinements and joint check',
            'padding/tail refinement', 'fixed-specimen phonon-bath refinement in all directions',
            'bulk phonon/detector convergence (Benchmark 7 remains incomplete)',
            '9B full material channel propagation and controlled thermal trace'],
        'channel_cutoff': 'not a numerical parameter in 9A, which is one linear source per mode; full-channel truncation belongs to 9B',
        'energy_crop': 'signed raw lines stored, crop is a detector/energy selection and is never interpreted as lost closed-system probability'}
    result['inherited_acceptance'] = {'relative_intensity':.01, 'relative_spectrum_L1':.01,
        'peak_shift_mev':.2, 'FWHM_mev':2, 'weak_probability_scale':1e-18,
        'weak_rule':'abs(change)/max(abs(fine),1e-18) < .01, inherited from B7; finite-to-zero changes are not ignored'}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seen_keys = set()
    for case in cases:
        setting = case['settings']
        with np.load(case['artifact']) as data:
            qxy = data['qxy_m_inv']
            theta = qxy/electron_wavenumber(60000*constants.elementary_charge)*1000
            radius = np.linalg.norm(theta, axis=1)
            weights = data['mode_momentum_weights']
            born = data['born_mode_momentum_weights']
            transfer = data['transfers_j']/(1e-3*constants.elementary_charge)
            axis = data['energy_grid_mev']
            masks = detector_masks(radius)
            # Inherited B7 8x8 pixel geometry, including off-axis windows, specified before inspecting values.
            edges = np.linspace(-20, 20, 9)
            maps = {}
            for name, (lo, hi) in PEAK_GROUPS.items():
                chosen = (transfer >= lo)&(transfer < hi)
                maps[name] = np.histogram2d(theta[radius <= 20, 1], theta[radius <= 20, 0], bins=(edges, edges),
                    weights=np.sum(weights[chosen][:, radius <= 20], axis=0))[0]
            masks['offaxis_pixel_5_4'] = (theta[:, 0] >= 5)&(theta[:, 0] < 10)&(theta[:, 1] >= 0)&(theta[:, 1] < 5)&(radius <= 20)
            masks['exact_forward_mode'] = np.linalg.norm(qxy, axis=1) == 0
            spectra = {name: absolute_spectrum(data['transfers_j'], np.sum(weights[:, mask], axis=1), axis)
                       for name, mask in masks.items()}
            values = {name: float(np.sum(weights[:, mask])) for name, mask in masks.items()}
            area_errors = {name: float(abs(trapezoid(spectra[name], axis)-values[name])) for name in masks}
            fine_axis = np.arange(-120, 120.0125, .025)
            fine_spectrum = absolute_spectrum(data['transfers_j'], np.sum(weights[:, masks['central_0_5_mrad']], axis=1), fine_axis)
            sampled_coarse = np.interp(fine_axis, axis, spectra['central_0_5_mrad'])
            energy_sampling = {'coarse_step_mev': .05, 'fine_step_mev': .025,
                'relative_L1_interpolation_change': float(trapezoid(abs(sampled_coarse-fine_spectrum), fine_axis)/max(trapezoid(fine_spectrum, fine_axis),1e-18)),
                'integrated_weight_absolute_change': float(abs(trapezoid(fine_spectrum, fine_axis)-trapezoid(spectra['central_0_5_mrad'],axis))),
                'FWHM_mev': 2, 'raw_line_energies_unchanged': True}
            thermal = transfer[:case['mode_count']]*1e-3*constants.elementary_charge
            temperature = setting['temperature_k']
            if temperature:
                ratio = np.exp(-thermal/(constants.Boltzmann*temperature))
                nbar = 1/np.expm1(thermal/(constants.Boltzmann*temperature))
                log_vacuum = float(np.sum(np.log1p(-ratio)))
                p = 2
                distribution = np.zeros(p+1)
                distribution[0] = np.exp(log_vacuum)
                for r in ratio:
                    distribution = np.convolve(distribution, r**np.arange(p+1))[:p+1]
                result['thermal_absolute_cutoff_diagnostics'].append({'temperature_k': temperature,
                    'mode_count': len(ratio), 'absolute_p': p, 'retained_initial_weight': float(np.sum(distribution)),
                    'log_vacuum_weight': log_vacuum, 'mean_total_thermal_occupation': float(np.sum(nbar)),
                    'Bose_detailed_balance_factor_error': float(np.max(abs(nbar/(nbar+1)-ratio))),
                    'interpretation': 'This is why a small total absolute-occupation cutoff is not used for material thermal states.'})
            key = f"n{setting['grid']}_z{setting['slices']}_L{setting['thickness_repeats']}_T{temperature}_bath{setting['phonon_z']}_pad{setting['padding_angstrom']}_elastic{int(setting['elastic_background'])}"
            if key in seen_keys:
                raise ValueError('material postprocessing artifact identity collision')
            seen_keys.add(key)
            artifact = output/(key+'.npz')
            np.savez_compressed(artifact, energy_grid_mev=axis, theta_mrad=theta,
                **{'spectrum_'+k:v for k,v in spectra.items()}, **{'window_map_'+k:v for k,v in maps.items()},
                full_energy_momentum_weights=weights, signed_line_energies_mev=transfer)
            row = {'settings': setting, 'thickness_nm': case['thickness_nm'], 'mode_count': case['mode_count'],
                'artifact': str(artifact), 'observables': case['observables'], 'additional_observables': values,
                'maximum_absolute_spectrum_area_error': max(area_errors.values()),
                'energy_sampling': energy_sampling,
                'gain_weight': float(np.sum(weights[transfer < 0])),
                'zero_T_gain_check': float(np.sum(weights[transfer < 0])) == 0 if temperature == 0 else None,
                'elastic_exit_norm_drift': case['max_incident_norm_drift'],
                'total_inelastic_all_momenta': case['total_inelastic_weight_all_represented_momenta'],
                'full_20mrad_absolute_difference_from_Born': values['full_0_20_mrad']-case['Born_weight_within_20_mrad'],
                'source_cumulative_seconds': json.loads(Path(case['checkpoint']).read_text())['cumulative_seconds']}
            result['cases'].append(row)
            if setting['grid'] == 65 and setting['thickness_repeats'] == 52 and setting['phonon_z'] == 2:
                fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
                for ax, (name, value) in zip(axes[:2], list(maps.items())[:2]):
                    image = ax.imshow(value, origin='lower', extent=(-20, 20, -20, 20))
                    ax.set(title=name, xlabel='theta x (mrad)', ylabel='theta y (mrad)')
                    fig.colorbar(image, ax=ax, label='absolute bin probability')
                # Sum over y beams at fixed x; probabilities remain discrete weights.
                xvalues, inverse = np.unique(theta[:, 0], return_inverse=True)
                energy_edges = np.arange(-100, 100.5, .5)
                image = np.zeros((len(energy_edges)-1, len(xvalues)))
                for mode in range(len(transfer)):
                    index = np.searchsorted(energy_edges, transfer[mode], side='right')-1
                    if 0 <= index < len(image):
                        image[index] += np.bincount(inverse, weights=weights[mode]*(radius <= 20), minlength=len(xvalues))
                show = (xvalues >= -20)&(xvalues <= 20)
                im = axes[2].imshow(image[:, show], aspect='auto', origin='lower', extent=(xvalues[show][0], xvalues[show][-1], -100, 100))
                axes[2].set(xlabel='theta x (mrad); y summed', ylabel='energy loss (meV)', title=f'9A, T={temperature} K')
                fig.colorbar(im, ax=axes[2], label='probability in 0.5 meV × discrete x channel')
                fig.savefig(output/(key+'_maps.png'), dpi=150)
                plt.close(fig)
    # One-parameter changes at fixed specimen and temperature. Preserve weak errors explicitly.
    for i, a in enumerate(result['cases']):
        for b in result['cases'][i+1:]:
            changed = [key for key in a['settings'] if a['settings'][key] != b['settings'][key]]
            if len(changed) != 1 or changed[0] not in ('grid', 'slices', 'phonon_z', 'padding_angstrom'):
                continue
            rows = {name: relative_difference(value, b['observables'][name]) for name, value in a['observables'].items()}
            with np.load(a['artifact']) as da, np.load(b['artifact']) as db:
                axis = db['energy_grid_mev']
                l1 = {name: float(trapezoid(abs(da['spectrum_'+name]-db['spectrum_'+name]), axis)/
                    max(trapezoid(db['spectrum_'+name], axis), 1e-18)) for name in DETECTORS}
                peaks = []
                for name in DETECTORS:
                    for group, (lo, hi) in PEAK_GROUPS.items():
                        mask = (axis >= lo)&(axis < hi)
                        ca, cb = da['spectrum_'+name][mask], db['spectrum_'+name][mask]
                        if not len(ca) or max(np.max(abs(ca)),np.max(abs(cb))) < 1e-18:
                            continue
                        x, y = float(axis[mask][np.argmax(ca)]), float(axis[mask][np.argmax(cb)])
                        peaks.append({'detector':name, 'group':group, 'coarse_mev':x, 'fine_mev':y, 'shift_mev':abs(x-y),
                            'identity_note':'dominant window maximum; boundary or competing maxima do not by themselves identify motion of one mode'})
            result['refinements'].append({'changed_parameter': changed[0], 'coarse': a['settings'], 'fine': b['settings'],
                'observables': rows, 'spectrum_L1_relative': l1,
                'peak_comparisons':peaks, 'peak_target_pass':all(r['shift_mev'] < .2 for r in peaks),
                'physical_intensity_pass': all(r['acceptance_error'] < .01 for r in rows.values()),
                'spectral_pass': max(l1.values()) < .01})
    json_write(output/'results.json', result)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for row in result['cases']:
        s = row['settings']
        if s['grid'] != 65 or s['phonon_z'] != 2 or not s['elastic_background']:
            continue
        with np.load(row['artifact']) as d:
            label = f"{row['thickness_nm']:.3g} nm, {s['temperature_k']} K"
            axes[0].plot(d['energy_grid_mev'], d['spectrum_central_0_5_mrad'], label=label)
            axes[1].plot(d['energy_grid_mev'], d['spectrum_offaxis_pixel_5_4'], label=label)
    for ax, title in zip(axes, ('central 0–5 mrad', 'off-axis 5–10 × 0–5 mrad')):
        ax.set(xlim=(-100, 100), xlabel='energy loss (meV)', ylabel='absolute probability / meV', title=title)
        ax.legend(fontsize=7)
    fig.savefig(output/'detector_thickness_temperature.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
