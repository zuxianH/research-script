"""Paired four-model spectra, raw weights, and separately labelled sensitivities."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import t as student_t

from .approximate_eels import AXIS, DETECTORS
from .approximate_eels_analysis import (available_born, selected_born, features,
    shape, compare)
from .approximate_eels_compare import match_features
from .born_quadrature import line_spectra
from .controlled_eels import DEFAULT, PREVIOUS, JOBS, check_source_contract
from .workflow_io import ROOT, json_write

WINDOWS = {'loss_0_40': (0, 40), 'loss_40_65': (40, 65), 'loss_65_90': (65, 90),
    'gain_0_40': (-40, 0), 'gain_40_65': (-65, -40), 'gain_65_90': (-90, -65),
    'loss_above90': (90, np.inf), 'all_signed': (-np.inf, np.inf)}


def paired_estimate(first_source, second_source, first_gamma, second_gamma):
    """Common-random-number source and Gamma errors, as independent batch pairs.

    Each input is an already integrated batch estimate with its 1/N factor.
    Gamma and source scrambles are independent; their covariance contributions
    add. We keep model-to-model covariance by differencing before variance.
    """
    a, b, ga, gb = map(np.asarray, (first_source, second_source, first_gamma, second_gamma))
    if a.shape != b.shape or ga.shape != gb.shape or a.shape[1:] != ga.shape[1:] or min(len(a), len(ga)) < 2:
        raise ValueError('Paired batches must have matching shapes and at least two independent scrambles')
    source_difference = b-a; gamma_difference = gb-ga
    mean = np.mean(source_difference, axis=0)+np.mean(gamma_difference, axis=0)
    variance = np.var(source_difference, axis=0, ddof=1)/len(a)+np.var(gamma_difference, axis=0, ddof=1)/len(ga)
    sem = np.sqrt(variance)
    return mean, sem, student_t.ppf(.975, min(len(a), len(ga))-1)*sem


def read_raw(folder, n=None):
    e = np.load(folder/'energies_mev.npy', mmap_mode='r')
    d = np.load(folder/'weights.npy', mmap_mode='r')
    b = np.load(folder/'born_weights.npy', mmap_mode='r') if (folder/'born_weights.npy').exists() else None
    return e[:n] if n is not None else e, d[:n] if n is not None else d, b[:n] if b is not None and n is not None else b


def window_sum(energy, weights, lo, hi):
    return np.sum(weights*((energy >= lo)&(energy < hi))[..., None], axis=tuple(range(energy.ndim)))


def born_window_weights(folder, temperature, mesh):
    e, w, _ = read_raw(folder/'born'/f'T{temperature}_mesh{mesh}')
    with np.load(folder/f'Gamma_cell_T{temperature}_mesh{mesh}.npz') as g:
        ge = g['energies_mev']; gw = g['weights']
    result = {}
    for name, (lo, hi) in WINDOWS.items():
        value = np.zeros(4)
        for start in range(0, len(e), 4096): value += window_sum(e[start:start+4096], w[start:start+4096], lo, hi)
        value -= window_sum(e[0], w[0], lo, hi)
        value += np.mean([window_sum(x, y, lo, hi) for x, y in zip(ge, gw)], axis=0)
        result[name] = value
    return result


def reconstruct(config, folder):
    temperature = config['temperatures_k'][0]
    mesh, axis, baseline, _, gamma_sem = selected_born(folder, temperature)
    corrections = []; direct = []; raw_delta = {name: [] for name in WINDOWS}; cases = config['source_cases']
    coordinates = []; densities = []
    for case in cases:
        if check_source_contract(config, case, folder) != 2**case['power']: raise ValueError('Incomplete production prefix')
        path = folder/'source'/case['name']; state = json.loads((path/'state.json').read_text())
        if state['status'] != 'COMPLETE': raise ValueError('Partial source cannot be summarized as complete')
        with np.load(path/'spectrum.npz') as d:
            corrections.append(d['correction']); direct.append(d['direct'])
            coordinates.append(d['q_fractional']); densities.append(d['inverse_sampling_density'])
        e, w, b = read_raw(path)
        if e.shape[1] != 12 or np.any(e[:, :6] <= 0) or not np.array_equal(e[:, 6:], -e[:, :6]): raise ValueError('Missing or unsigned physical branches')
        if np.any(w < 0) or np.any(b < 0) or not np.isfinite(w).all(): raise ValueError('Invalid raw probabilities')
        for name, (lo, hi) in WINDOWS.items(): raw_delta[name].append(window_sum(e, w-b, lo, hi)/len(e))
    corrections = np.array(corrections); direct = np.array(direct)
    with np.load(folder/f'Gamma_cell_T{temperature}_mesh{mesh}.npz') as g:
        gamma = g['independent_corrections'].copy(); gamma_q = g['q_fractional'].copy()
    mean = baseline+np.mean(corrections, axis=0)
    sem = np.sqrt(np.var(corrections, axis=0, ddof=1)/len(corrections)+gamma_sem**2)
    halfwidth = student_t.ppf(.975, min(len(corrections), len(gamma))-1)*sem
    born_raw = born_window_weights(folder, temperature, mesh)
    raw = {name: born_raw[name]+np.mean(raw_delta[name], axis=0) for name in WINDOWS}
    return dict(axis=axis, born=baseline, mean=mean, corrections=corrections, direct=direct,
        gamma=gamma, gamma_q=gamma_q, gamma_sem=gamma_sem, sem=sem, halfwidth=halfwidth,
        born_raw=born_raw, raw=raw, q=np.array(coordinates), inverse=np.array(densities), config=config, folder=folder,
        mesh=mesh, raw_delta={key: np.array(value) for key, value in raw_delta.items()})


def covariance_diagnostics(folder, temperature):
    rows = []
    previous = None
    for mesh in (16, 24, 32, 64):
        path = folder/'covariance'/f'T{temperature}_mesh{mesh}.json'
        if not path.exists(): continue
        d = json.loads(path.read_text()); a = np.array(d['covariance_m2'])
        rows.append({'mesh': mesh, 'covariance_m2': a, 'trace_m2': np.trace(a),
            'relative_matrix_change_from_previous': np.linalg.norm(a-previous)/np.linalg.norm(a) if previous is not None else None,
            'force_sha256': d['force_constants_sha256']})
        previous = a
    return rows


def numerical(data):
    config = data['config']; folder = data['folder']; temperature = config['temperatures_k'][0]; rows = []
    born = available_born(folder, temperature)
    for a, b in zip(born[:-1], born[1:]):
        coarse, fine = a[2].copy(), b[2].copy()
        for m, spectrum in ((a[0], coarse), (b[0], fine)):
            p = folder/f'Gamma_cell_T{temperature}_mesh{m}.npz'
            if p.exists():
                with np.load(p) as g: spectrum += g['correction']
        rows.append({'kind': 'Born q quadrature with local Gamma replacement', 'coarse_mesh': a[0], 'fine_mesh': b[0],
            **compare(data['axis'], coarse, fine)})
    prefixes = []
    final_power = min(c['power'] for c in config['source_cases'])
    for power in range(max(4, final_power-2), final_power+1):
        n = 2**power; delta = []
        for c in config['source_cases']:
            e, w, b = read_raw(folder/'source'/c['name'], n)
            delta.append(line_spectra(e, (w-b)/n, data['axis']))
        value = data['born']+np.mean(delta, axis=0)
        selected = (data['axis'] >= 5)&(data['axis'] < 90)
        integral = trapezoid(np.array(delta)[:, selected], data['axis'][selected], axis=1)
        gamma_integral = trapezoid(data['gamma'][:, selected], data['axis'][selected], axis=1)
        half = student_t.ppf(.975, 7)*np.sqrt(np.var(integral, axis=0, ddof=1)/8+np.var(gamma_integral, axis=0, ddof=1)/8)
        prefixes.append((n, value, {'q_per_batch': n, 'independent_batches': 8,
            'loss_weight_5_90': trapezoid(value[selected], data['axis'][selected], axis=0),
            'loss_weight_5_90_student95_halfwidth': half,
            'scope': 'Confidence estimate of integrated weight across independent batch integrals, not an integral of pointwise confidence bands.'}))
    for a, b in zip(prefixes[:-1], prefixes[1:]):
        rows.append({'kind': 'independent-batch nested q counts', 'coarse_q_per_batch': a[0], 'fine_q_per_batch': b[0],
            'coarse_statistics': a[2], 'fine_statistics': b[2], **compare(data['axis'], a[1], b[1])})
    for c in config.get('refinement_cases', [])+[config['covariance_case']] if 'covariance_case' in config else []:
        p = folder/'source'/c['name']/'state.json'
        if not p.exists() or json.loads(p.read_text()).get('status') != 'COMPLETE': continue
        n = 2**c['power']; fine_e, fine_w, _ = read_raw(p.parent, n)
        e, w, _ = read_raw(folder/'source'/config['source_cases'][0]['name'], n)
        np.testing.assert_array_equal(e, fine_e)
        # Full production mean plus a paired direct-DW difference, using the
        # unchanged dense control variate. This remains valid for covariance
        # changes; omitting their Born change from a D-B difference would not.
        refined = data['mean']+line_spectra(e, (fine_w-w)/n, data['axis'])
        rows.append({'kind': c['name'], 'matched_q': n, 'settings': c,
            'scope': 'Matched production-q sensitivity estimate; not a full fine-grid production calculation or statistical confidence bound.',
            **compare(data['axis'], data['mean'], refined)})
    return rows


def curve_metrics(axis, spectrum, raw, tacaw):
    selected = (axis >= 5)&(axis < 90); gain = (axis <= -5)&(axis > -90)
    rows = []
    for j, detector in enumerate(DETECTORS):
        conditional, area = shape(axis, spectrum[:, j]); f = features(axis, spectrum[:, j])
        normalized, _ = shape(axis, tacaw)
        rows.append({'detector': detector, 'features': f, 'weight_5_90': area,
            'raw_unbroadened_window_weights': {name: value[j] for name, value in raw.items()},
            'conditional_TACAW_shape_L1': float(trapezoid(abs(conditional[selected]-normalized[selected]), axis[selected])),
            'conditional_TACAW_shape_L1_region_contributions': {name: float(trapezoid(
                abs(conditional[selected]-normalized[selected])*((axis[selected] >= lo)&(axis[selected] < hi)), axis[selected]))
                for name, lo, hi in (('acoustic_5_40', 5, 40), ('middle_40_65', 40, 65), ('optical_65_90', 65, 90))},
            'conditional_feature_matching': match_features(features(axis, tacaw), f),
            'gain_over_loss_5_90': float(trapezoid(spectrum[gain, j], axis[gain])/area),
            'broadened_signed_weight': float(trapezoid(spectrum[:, j], axis)),
            'raw_signed_weight': raw['all_signed'][j]})
    return rows


def load_tacaw(temperature, axis):
    # Reuse the immutable disclosed reduction; do not repeat corrections or
    # assign a new energy axis. The all-momentum sum and disk remain separate.
    path = PREVIOUS/f'TACAW_T{temperature}_raw_reductions.npz'
    with np.load(path) as d:
        x = d['broadened_axis_mev']; step = float(d['analysis_energy_mev'][1]-d['analysis_energy_mev'][0])
        return {name: np.interp(axis, x, d[name+'_broadened_weights']/step)
            for name in ('all_recorded_momenta', 'pixel_disk_radius20')}


def summarize(output):
    collections = {}
    previous_config = json.loads((ROOT/'benchmarks/configs/approximate_eels_production_20261102_v1.json').read_text())
    for temperature in (300, 1000):
        c = {**previous_config, 'temperatures_k': [temperature], 'model': 'fixed', 'job': f'fixed_r4_T{temperature}',
            'source_cases': [s for s in previous_config['source_cases'] if s['temperature_k'] == temperature]}
        collections[c['job']] = reconstruct(c, PREVIOUS)
        with np.load(PREVIOUS/f'final_T{temperature}_spectra.npz') as old:
            np.testing.assert_allclose(collections[c['job']]['mean'], old['mean'], rtol=1e-14, atol=1e-22)
    for name in JOBS:
        config = json.loads((output/'configs'/f'{name}.json').read_text()); folder = ROOT/config['output']
        if not all((folder/'source'/c['name']/'spectrum.npz').exists() for c in config['source_cases']): continue
        collections[name] = reconstruct(config, folder)
    if len(collections) < 3: raise ValueError('Complete fitted300 production before four-model analysis')
    rows = []; checks = {}; areas = []; uncertainty = {}; paired = []
    for name, data in collections.items():
        c = data['config']; temperature = c['temperatures_k'][0]; axis = data['axis']; selected = (axis >= 5)&(axis < 90)
        tacaw = load_tacaw(temperature, axis)
        for method in ('born', 'mean'):
            spectrum = data[method]; raw = data['born_raw'] if method == 'born' else data['raw']
            metrics = curve_metrics(axis, spectrum, raw, tacaw['all_recorded_momenta'])
            disk_metrics = curve_metrics(axis, spectrum, raw, tacaw['pixel_disk_radius20'])
            rows.append({'id': name+'_'+('Born' if method == 'born' else 'DW'), 'job': name,
                'method': 'Born' if method == 'born' else 'distorted_wave', 'phonon_model': c['model'],
                'temperature_k': temperature, 'oriented_repeats': c['selected_repeats'], 'detectors': metrics,
                'conditional_TACAW_disk_sensitivity': disk_metrics,
                'density_units': 'model inelastic weight per meV; raw weights are not normalized detector probabilities'})
            error = max(abs(r['broadened_signed_weight']-r['raw_signed_weight'])/max(abs(r['raw_signed_weight']), 1e-100) for r in metrics)
            areas.append({'curve': rows[-1]['id'], 'maximum_relative_area_error': error, 'passed': error < 1e-9, 'renormalized': False})
        checks[name] = numerical(data) if name in JOBS else []
        variance = trapezoid(data['halfwidth'][selected], axis[selected], axis=0)/trapezoid(data['mean'][selected], axis[selected], axis=0)
        rng = np.random.default_rng(749031); shape_samples = []; gain_ratios = []; loss_weights = []
        peak_samples = {k: [] for k in features(axis, data['mean'][:, 0])}
        reference_shape = data['mean']/trapezoid(data['mean'][selected], axis[selected], axis=0)[None]
        for _ in range(500):
            value = data['born']+np.mean(data['corrections'][rng.integers(8, size=8)], axis=0)
            value += np.mean(data['gamma'][rng.integers(8, size=8)], axis=0)-np.mean(data['gamma'], axis=0)
            norm = trapezoid(value[selected], axis[selected], axis=0)
            gain = (axis <= -5)&(axis > -90)
            gain_ratios.append(trapezoid(value[gain], axis[gain], axis=0)/norm)
            loss_weights.append(norm)
            shape_samples.append(trapezoid(abs(value[selected]/norm[None]-reference_shape[selected]), axis[selected], axis=0))
            f = features(axis, value[:, 0])
            for key in peak_samples: peak_samples[key].append(f[key]['peak_mev'])
        correction_weights = trapezoid(data['corrections'][:, selected], axis[selected], axis=1)
        correction_mean = np.mean(correction_weights, axis=0)
        correction_halfwidth = student_t.ppf(.975, 7)*np.std(correction_weights, axis=0, ddof=1)/np.sqrt(8)
        uncertainty[name] = {'batches': 8, 'q_per_batch': 2**min(case['power'] for case in c['source_cases']),
            'integrated_pointwise95_halfwidth_over_weight_by_detector': variance,
            'bootstrap_seed': 749031, 'bootstrap_replicates': 500,
            'bootstrap_shape_L1_95_percentile_by_detector': np.percentile(shape_samples, 95, axis=0),
            'bootstrap_primary_feature_2_5_97_5_percentiles_mev': {k: np.percentile(v, [2.5, 97.5]) for k, v in peak_samples.items()},
            'bootstrap_loss_weight_2_5_97_5_percentiles_by_detector': np.percentile(loss_weights, [2.5, 97.5], axis=0),
            'bootstrap_gain_loss_ratio_2_5_97_5_percentiles_by_detector': np.percentile(gain_ratios, [2.5, 97.5], axis=0),
            'DW_minus_Born_weight_5_90': correction_mean,
            'DW_minus_Born_weight_5_90_pointwise_student95_halfwidth': correction_halfwidth,
            'DW_minus_Born_uncertainty_note': 'The shared deterministic baseline and shared Gamma correction cancel; independent source batch-integral variance only.',
            'scope': 'Pointwise finite-eight-batch sensitivity; excludes deterministic quadrature bias and physical-model error.'}
        np.savez_compressed(output/(name+'_spectra.npz'), energy_mev=axis, Born=data['born'], DW=data['mean'],
            DW_standard_error=data['sem'], DW_pointwise95_halfwidth=data['halfwidth'], source_corrections=data['corrections'],
            Gamma_corrections=data['gamma'], q_fractional=data['q'], inverse_sampling_density=data['inverse'],
            raw_Born_signed_weight=data['born_raw']['all_signed'], raw_DW_signed_weight=data['raw']['all_signed'])
    pairs = [(f'fixed_r4_T{t}', f'fitted_r4_T{t}', 'phonon model at fixed thickness') for t in (300, 1000)]
    pairs += [('fixed_r4_T300', 'fixed_r8_T300', 'physical thickness, fixed harmonic'),
              ('fitted_r4_T300', 'fitted_r8_T300', 'physical thickness, temperature-fitted')]
    for first, second, kind in pairs:
        if second not in collections: continue
        a, b = collections[first], collections[second]
        shared = min(a['q'].shape[1], b['q'].shape[1])
        np.testing.assert_array_equal(a['q'][:, :shared], b['q'][:, :shared])
        np.testing.assert_array_equal(a['inverse'][:, :shared], b['inverse'][:, :shared])
        np.testing.assert_array_equal(a['gamma_q'], b['gamma_q'])
        _, sem, half = paired_estimate(a['corrections'], b['corrections'], a['gamma'], b['gamma'])
        np.savez_compressed(output/(first+'_to_'+second+'_paired.npz'), energy_mev=a['axis'],
            signed_DW_difference=b['mean']-a['mean'], paired_standard_error=sem, pointwise95_halfwidth=half)
        for method in ('born', 'mean'):
            paired.append({'first': first, 'second': second, 'kind': kind, 'method': method,
                'shared_nested_q_per_batch': shared, 'first_q_per_batch': a['q'].shape[1], 'second_q_per_batch': b['q'].shape[1],
                'pairing_note': 'Uncertainty uses differences between8independent batch integrals; nested prefixes preserve covariance even when final q counts differ.',
                'not_a_numerical_convergence_gate': True, **compare(a['axis'], a[method], b[method])})
    rows.sort(key=lambda r: (r['temperature_k'], r['oriented_repeats'], r['phonon_model'] == 'fitted', r['method'] != 'Born'))
    inherited = {name: json.loads((PREVIOUS/name).read_text()) for name in
        ('numerical_confirmation.json', 'Born_convergence.json')}
    prior_statistics = json.loads((PREVIOUS/'source_statistics.json').read_text())
    inherited['source_count_checks'] = {t: prior_statistics['temperatures'][t]['numerical_sample_count_comparisons'] for t in ('300', '1000')}
    result = {'status': 'COMPUTED' if len(collections) == 6 else 'PARTIAL', 'curves': rows,
        'numerical_checks': checks, 'area_audits': areas, 'uncertainty': uncertainty, 'physical_comparisons': paired,
        'covariance_checks': {name: covariance_diagnostics(d['folder'], d['config']['temperatures_k'][0]) for name, d in collections.items()},
        'inherited_fixed_r4_numerical_evidence': inherited,
        'reference_scope': 'TACAW7.5fs axis conditional; Yoshioka0–20mrad versus unmatched recorded TACAW momenta. Fixed volume. No fitted shifts, scales, extra Kubo correction or intrinsic linewidths.',
        'original_B11A_status': 'FAIL preserved', 'intrinsic_linewidths': 'NOT_RESOLVED preserved'}
    json_write(output/'comparison.json', result)
    return result, collections


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', default=DEFAULT)
    args = p.parse_args(); output = ROOT/args.output
    result, _ = summarize(output)
    print(result['status'], 'curves:', len(result['curves']), flush=True)


if __name__ == '__main__': main()
