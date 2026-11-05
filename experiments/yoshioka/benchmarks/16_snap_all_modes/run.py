#!/usr/bin/env python3
"""Si [110] finite-cell SNAP/hiPhive harmonic, full 3D nonlinear channel run.

Keep inputs and generated arrays in the external --output directory. Each
potential matrix element uses the requested working-Fock exponential before
projection into the total-occupation basis; longitudinal phases are retained.
"""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy import constants
from scipy.linalg import null_space
from scipy.sparse.linalg import LinearOperator, expm_multiply

from yoshioka.channel_spectra import angular_detector_masks, broaden_transition_lines, transition_lines
from yoshioka.coupled_channels import channel_basis, thermal_channel_weights
from yoshioka.electron_phonon import LobatoSilicon
from yoshioka.elastic_propagation import electron_speed
from yoshioka.free_propagation import electron_wavenumber
from yoshioka.projected_material_action import phase_blocks
from yoshioka.provenance import fingerprint, json_write, sha256
from yoshioka.snap_forces import SNAPForces


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def prepare(out, cfg):
    """Fit harmonic SNAP forces in hiPhive, retaining independent validation."""
    from ase.io import read, write
    from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
    from yoshioka.lattice_dynamics import finite_periodic_modes, real_space_from_supercell
    from yoshioka.mod_forces import complete_force_constants, correct_sum_rules
    from yoshioka.snap_forces import central_snap_force_constants

    folder = out / 'phonons'
    folder.mkdir(exist_ok=True)
    if (folder / 'preparation.json').exists():
        saved = json.loads((folder / 'preparation.json').read_text())
        if saved['config_digest'] != fingerprint(cfg) or not saved['passed']:
            raise ValueError('existing preparation has a different configuration or failed')
        log('Using completed phonon preparation')
        return
    unit = read(out / 'inputs/Si_110_temp300.lmp', format='lammps-data', atom_style='atomic')
    unit.set_masses(np.full(len(unit), cfg['mass_u']))
    coeff, param = out / 'models/SNAPotential.snapcoeff', out / 'models/SNAPotential.snapparam'
    for file in (coeff, param, out / 'inputs/Si_110_temp300.lmp'):
        if sha256(file) != cfg['input_sha256'][file.name]:
            raise ValueError('input identity changed: ' + str(file))
    with SNAPForces(unit, coeff, param) as calc:
        unit, equilibrium = calc.relax_zero_pressure()
    write(folder / 'equilibrium.extxyz', unit)
    log('Static zero-pressure SNAP cell relaxed')
    reps = tuple(cfg['fit_supercell'])
    reference = unit.repeat(reps)
    cs = ClusterSpace(unit, [cfg['harmonic_cutoff_angstrom']])
    container = StructureContainer(cs)
    rng = np.random.default_rng(cfg['fit_seed'])
    displacements, exact, odd = [], [], []
    with SNAPForces(reference, coeff, param) as calc:
        for sample in range(cfg['training_pairs'] + cfg['validation_pairs']):
            u = rng.normal(scale=cfg['fit_sigma_angstrom'], size=(len(reference), 3))
            u -= u.mean(axis=0)
            plus = calc.evaluate(reference.positions + u)['forces_ev_per_angstrom']
            minus = calc.evaluate(reference.positions - u)['forces_ev_per_angstrom']
            displacements.append(u)
            exact.append(plus)
            odd.append((plus - minus) / 2)
            if sample < cfg['training_pairs']:
                structure = reference.copy()
                structure.new_array('displacements', u)
                structure.new_array('forces', odd[-1])
                container.add_structure(structure)
            log(f'SNAP training/validation pair {sample + 1}')
    matrix, target = container.get_fit_data()
    scale = np.linalg.norm(matrix, axis=0)
    if np.any(scale == 0):
        raise ValueError('unidentified hiPhive parameter')
    normalized = matrix / scale
    parameters, _, rank, singular = np.linalg.lstsq(normalized, target, rcond=None)
    if rank != len(parameters):
        raise ValueError('rank-deficient hiPhive fit')
    fcp = ForceConstantPotential(cs, parameters / scale)
    fcp.write(str(folder / 'harmonic_model.fcp'))
    fc = fcp.get_force_constants(reference).get_fc_array(2)
    model, range_record = real_space_from_supercell(unit, reps, fc)
    model.save(folder / 'hiphive_model.npz')
    finite = finite_periodic_modes(model, (1, 1, 1))
    np.savez_compressed(folder / 'bath.npz', **finite, cell_m=np.asarray(unit.cell) * 1e-10)
    u = np.array(displacements[cfg['training_pairs']:])
    prediction = -np.einsum('ijab,sjb->sia', fc, u)
    actual = np.array(exact[cfg['training_pairs']:])
    odd_actual = np.array(odd[cfg['training_pairs']:])
    heldout = float(np.linalg.norm(prediction - actual) / np.linalg.norm(actual))
    odd_error = float(np.linalg.norm(prediction - odd_actual) / np.linalg.norm(odd_actual))
    # Direct finite differences in the requested periodic 4-atom specimen are
    # independent of the hiPhive training supercell and fitted parameters.
    fd_energies = []
    for delta in (0.005, 0.0025):
        columns, _, _ = central_snap_force_constants(unit, (1, 1, 1), delta, coeff, param)
        raw = complete_force_constants(columns, (1, 1, 1))
        corrected = correct_sum_rules(raw)
        hessian = corrected.transpose(0, 2, 1, 3).reshape(12, 12)
        translations = np.tile(np.eye(3), (4, 1)) / 2
        complement = null_space(translations.T)
        values = np.linalg.eigvalsh(complement.T @ hessian @ complement)
        if np.any(values <= 0):
            raise ValueError('independent finite differences have an unstable internal mode')
        omega = np.sqrt(values * constants.e / 1e-20 / (cfg['mass_u'] * constants.atomic_mass))
        fd_energies.append(omega * constants.hbar / (constants.e * 1e-3))
        np.savez(folder / f'finite_difference_{delta:.4f}.npz', raw_ev_a2=raw, corrected_ev_a2=corrected)
    quanta = finite['omega_rad_s'] * constants.hbar / (constants.e * 1e-3)
    fit_energy_error = float(np.max(abs(quanta - fd_energies[-1])))
    fd_change = float(np.max(abs(fd_energies[-1] - fd_energies[0])))
    np.savez_compressed(folder / 'training_validation.npz', displacements_angstrom=displacements,
                        forces_plus_ev_per_angstrom=exact, odd_forces_ev_per_angstrom=odd)
    record = dict(config_digest=fingerprint(cfg), rank=int(rank), parameters=len(parameters),
                  condition_number=float(singular[0] / singular[-1]),
                  training_odd_relative_error=float(np.linalg.norm(matrix @ (parameters / scale) - target) / np.linalg.norm(target)),
                  heldout_relative_force_error=heldout, heldout_odd_relative_error=odd_error,
                  mode_quanta_mev=quanta, fitted_vs_finite_difference_max_mev=fit_energy_error,
                  finite_difference_refinement_max_mev=fd_change, range=range_record,
                  equilibrium=equilibrium, rigid_translation_modes_removed=3, active_mode_count=len(quanta),
                  approximation='Static zero-pressure volume; no thermal or zero-point expansion',
                  boundary='1x1x1 periodic nuclear cell; vacuum-padded z electron potential')
    record['passed'] = bool(heldout < 0.01 and odd_error < 0.001 and fit_energy_error < 0.05 and fd_change < 0.05
                            and np.max(abs(equilibrium['forces_ev_per_angstrom'])) < 1e-6
                            and np.max(abs(equilibrium['pressure_bar_voigt'])) < 0.1
                            and len(quanta) == 9)
    json_write(folder / 'preparation.json', record)
    if not record['passed']:
        raise ValueError('SNAP/hiPhive independent validation failed; inspect preparation.json')
    log(f'Phonons validated: 9 internal modes; fit frequency error {fit_energy_error:.6g} meV')


class FullDisplacementAction:
    """Exact working-Fock phase blocks, all Cartesian displacements, FFT Galerkin.

    Frequency folding evaluates the declared finite z Fourier field at the
    independent propagation midpoints. M=2N-1 prevents transverse cyclic alias.
    Only one dense local CxC channel matrix field is resident during propagation.
    """
    def __init__(self, bath, cfg, folder, tile=4):
        self.bath, self.cfg = bath, cfg
        self.n, self.steps = cfg['n'], cfg['steps']
        fields, dimension = cfg['field_points'], cfg['working_dimension']
        self.basis = channel_basis(len(bath['omega_rad_s']), cfg['p'])
        self.c = len(self.basis)
        if any(cfg[key] < 1 or cfg[key] % 2 != 1 for key in ('n', 'steps', 'field_points')):
            raise ValueError('positive odd electron, propagation, and field grids required')
        if cfg['working_dimension'] <= cfg['p']:
            raise ValueError('working oscillator dimension must exceed the occupation cutoff')
        self.shape = (self.c, self.n, self.n)
        self.pad = 2 * self.n - 1
        # Include simultaneous Fourier/FFT/local field copies and work vectors.
        estimated_rss = 5 * self.pad**2 * self.c**2 * 16 + 64 * self.c * self.pad**2 * 16 + 2 * 1024**3
        if estimated_rss > cfg.get('maximum_rss_bytes', 64 * 1024**3):
            raise MemoryError(f'conservative local-field estimate {estimated_rss} exceeds configured budget')
        cell = bath['cell_m']
        self.length = cell[2, 2] + 2 * cfg['padding_m']
        self.dz = self.length / self.steps
        energy = cfg['electron_energy_ev'] * constants.e
        self.hv = constants.hbar * electron_speed(energy)
        self.channel_j = self.basis @ (constants.hbar * bath['omega_rad_s'])
        beams = np.arange(-(self.n // 2), self.n // 2 + 1)
        self.ix = beams % self.pad
        qx, qy = np.meshgrid(2 * np.pi * beams / cell[0, 0], 2 * np.pi * beams / cell[1, 1])
        self.qxy = np.stack((qx, qy), axis=-1)
        self.kinetic_j = self.hv * (qx*qx + qy*qy) / (2 * electron_wavenumber(energy))
        difference = np.rint(np.fft.fftfreq(self.pad) * self.pad).astype(int)
        qx, qy = np.meshgrid(2 * np.pi * difference / cell[0, 0], 2 * np.pi * difference / cell[1, 1])
        Q = np.column_stack((qx.ravel(), qy.ravel()))
        kz = 2 * np.pi * np.fft.fftfreq(fields, self.length / fields)
        frequencies = np.rint(np.fft.fftfreq(fields) * fields).astype(int)
        negative = (-np.arange(self.pad)) % self.pad
        opposite = (negative[:, None] * self.pad + negative[None, :]).ravel()
        canonical = np.flatnonzero(np.arange(len(Q)) <= opposite)
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        contract = dict(config=cfg, bath={k: np.asarray(v).tolist() for k, v in bath.items()},
                        source_sha256=sha256(Path(__file__)))
        contract_digest = fingerprint(contract)
        manifest = folder / 'manifest.json'
        path = folder / 'channel_fields.npy'
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved['digest'] != contract_digest or saved['sha256'] != sha256(path):
                raise ValueError('field cache configuration/source/content mismatch')
        else:
            if path.exists():
                raise FileExistsError('uncommitted field cache exists; preserve and inspect before restarting')
            field = np.lib.format.open_memmap(path, mode='w+', dtype=complex,
                                            shape=(self.steps, len(Q), self.c, self.c))
            potential = LobatoSilicon.from_abtem_table()
            for lo in range(0, len(canonical), tile):
                selected = canonical[lo:lo + tile]
                wavevectors = np.empty((len(selected), fields, 3))
                wavevectors[:, :, :2] = Q[selected, None, :]
                wavevectors[:, :, 2] = kz
                kernel = potential.fourier_energy_j_m3(np.sum(wavevectors**2, axis=-1)) / np.prod(np.diag(cell)[:2]) / self.length
                folded = np.zeros((len(selected), self.steps, self.c, self.c), complex)
                for atom, pos in enumerate(bath['positions_m']):
                    xi = wavevectors @ bath['displacements_m'][atom]
                    blocks = phase_blocks(self.basis, xi.reshape(-1, xi.shape[-1]), dimension)
                    blocks = blocks.reshape(len(selected), fields, self.c, self.c)
                    coefficient = kernel * np.exp(-1j * (Q[selected] @ pos[:2])[:, None]
                                                -1j * kz * (pos[2] + cfg['padding_m'] - self.dz/2))
                    blocks *= coefficient[:, :, None, None]
                    for s in range(self.steps):
                        folded[:, s] += blocks[:, frequencies % self.steps == s].sum(axis=1)
                values = np.fft.ifft(folded, axis=1) * self.steps
                if selected[0] == 0:
                    residual = np.linalg.norm(values[0].imag) / max(np.linalg.norm(values[0].real), np.finfo(float).tiny)
                    if residual > 1e-11:
                        raise ValueError(f'zero-transfer field is not real: {residual}')
                    values[0] = values[0].real
                field[:, selected] = values.swapaxes(0, 1)
                field[:, opposite[selected]] = values.conj().swapaxes(0, 1)
                if lo % (tile * 100) == 0:
                    field.flush()
                    import mmap
                    field._mmap.madvise(mmap.MADV_DONTNEED)
                    log(f'Full 3D channel fields: {lo}/{len(canonical)} transfers')
            field.flush()
            del field
            json_write(manifest, dict(digest=contract_digest, sha256=sha256(path),
                                     channel_count=self.c, full_cartesian_displacements=True))
        self.fields = np.load(path, mmap_mode='r')
        self.cached_index, self.potential = None, None

    def set_slice(self, index):
        if self.cached_index == index:
            return
        self.potential = None
        fourier = np.array(self.fields[index])
        self.trace_potential = np.trace(fourier[0]).real * self.n**2
        local = np.fft.ifft2(fourier.reshape(self.pad, self.pad, self.c, self.c), axes=(0, 1)) * self.pad**2
        del fourier
        import mmap
        self.fields._mmap.madvise(mmap.MADV_DONTNEED)
        # Check in bounded tiles so validation does not double the field RSS.
        norm2, difference2 = 0., 0.
        for row in local:
            delta = row - row.swapaxes(-1, -2).conj()
            norm2 += float(np.vdot(row, row).real)
            difference2 += float(np.vdot(delta, delta).real)
        residual = np.sqrt(difference2 / max(norm2, np.finfo(float).tiny))
        if residual > 1e-10:
            raise ValueError(f'full-displacement local field is not Hermitian: {residual}')
        self.potential = local.reshape(-1, self.c, self.c)
        self.cached_index = index

    def apply(self, wave, index):
        self.set_slice(index)
        wave = np.asarray(wave).reshape(self.shape)
        result = (self.channel_j[:, None, None] + self.kinetic_j) * wave
        embedded = np.zeros((self.c, self.pad, self.pad), complex)
        embedded[:, self.ix[:, None], self.ix[None, :]] = wave
        position = np.fft.ifft2(embedded, axes=(-2, -1), norm='ortho').reshape(self.c, -1)
        product = np.einsum('qij,jq->iq', self.potential, position, optimize=True)
        reciprocal = np.fft.fft2(product.reshape(embedded.shape), axes=(-2, -1), norm='ortho')
        return result + reciprocal[:, self.ix[:, None], self.ix[None, :]]

    def exponential_action(self, wave, index):
        self.set_slice(index)
        factor = -1j * self.dz / self.hv
        size = int(np.prod(self.shape))
        def action(x, scale):
            return (scale * self.apply(np.asarray(x).reshape(self.shape), index)).ravel()
        def batch(x, scale):
            return np.column_stack([action(x[:, i], scale) for i in range(x.shape[1])])
        operator = LinearOperator((size, size), dtype=complex,
            matvec=lambda x: action(x, factor), rmatvec=lambda x: action(x, factor.conjugate()),
            matmat=lambda x: batch(x, factor), rmatmat=lambda x: batch(x, factor.conjugate()))
        trace = self.n**2 * self.channel_j.sum() + self.c * self.kinetic_j.sum() + self.trace_potential
        return expm_multiply(operator, wave.ravel(), traceA=factor * trace).reshape(self.shape)


def run(out, cfg):
    with np.load(out / 'phonons/bath.npz') as data:
        bath = {k: data[k] for k in ('positions_m', 'cell_m', 'displacements_m', 'omega_rad_s')}
    h = FullDisplacementAction(bath, cfg, out / 'fields')
    weights = thermal_channel_weights(h.basis, constants.hbar * bath['omega_rad_s'], cfg['temperature_k'])
    # Retain each appreciably populated initial state. Bound omitted mass using
    # log products, avoiding cancellation of tiny Bose populations at 10 K.
    ratios = np.exp(-constants.hbar * bath['omega_rad_s'] / (constants.k * cfg['temperature_k']))
    log_vacuum = np.log1p(-ratios).sum()
    nonvacuum_mass = -np.expm1(log_vacuum)
    initial_indices = np.flatnonzero(weights > cfg['initial_weight_threshold'])
    executed_nonvacuum = weights[initial_indices[initial_indices != 0]].sum()
    omitted = max(0., float(nonvacuum_mass - executed_nonvacuum))
    if omitted > cfg['maximum_omitted_thermal_probability']:
        raise ValueError(f'thermal omitted-mass bound {omitted} exceeds configured tolerance')
    log(f'Propagating {len(initial_indices)} thermal initial states; omitted probability bound {omitted:.3g}')
    checkpoint = out / 'checkpoints'
    checkpoint.mkdir(exist_ok=True)
    waves, norm_drifts = [], []
    for initial_index in initial_indices:
        wave = np.zeros(h.shape, complex)
        wave[initial_index, h.n//2, h.n//2] = 1
        completed, drift = 0, 0.
        state_path = checkpoint / f'initial_{initial_index}.json'
        contract = fingerprint(dict(config=cfg, source=sha256(Path(__file__)),
                                    bath=sha256(out / 'phonons/bath.npz'), initial=int(initial_index)))
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['contract'] != contract or sha256(checkpoint / state['wave_file']) != state['wave_sha256']:
                raise ValueError('checkpoint configuration/source/wave mismatch')
            wave = np.load(checkpoint / state['wave_file'])
            completed, drift = state['completed_slices'], state['max_norm_drift']
        for s in range(completed, h.steps):
            wave = h.exponential_action(wave, s)
            norm = float(np.sum(abs(wave)**2))
            if not np.isfinite(norm):
                raise ValueError('non-finite propagated probability')
            drift = max(drift, abs(norm - 1))
            if drift > 1e-9:
                raise ValueError(f'probability norm drift {drift} exceeds 1e-9')
            filename = f'initial_{initial_index}_slice_{s+1}.npy'
            temporary = checkpoint / (filename + '.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, wave)
            temporary.replace(checkpoint / filename)
            previous = state['wave_file'] if state_path.exists() else None
            state = dict(contract=contract, completed_slices=s+1, total_slices=h.steps,
                         wave_file=filename, wave_sha256=sha256(checkpoint / filename), max_norm_drift=drift)
            json_write(state_path, state)
            if previous and previous != filename:
                (checkpoint / previous).unlink()
            log(f'Initial {initial_index}: slice {s+1}/{h.steps}; norm drift {drift:.3g}')
        waves.append(wave)
        norm_drifts.append(drift)
    angle, masks = angular_detector_masks(h.qxy, electron_wavenumber(cfg['electron_energy_ev'] * constants.e), ((0, 30),))
    lines = transition_lines(np.array(waves), initial_indices, h.basis, constants.hbar * bath['omega_rad_s'], weights[initial_indices], masks)
    axis = np.linspace(-180, 180, 7201)
    spectra = broaden_transition_lines(lines, axis, cfg['fwhm_mev'])
    np.savez_compressed(out / 'exit.npz', exit_coefficients=waves, initial_indices=initial_indices,
                        basis=h.basis, qxy_m_inv=h.qxy, angles_mrad=angle, thermal_weights=weights,
                        omitted_initial_probability=omitted, mode_quanta_j=constants.hbar*bath['omega_rad_s'])
    np.savez_compressed(out / 'spectra.npz', **lines, **spectra)
    np.savetxt(out / 'spectrum.csv', np.column_stack((axis, spectra['inclusive_per_mev'][0],
               spectra['changed_per_mev'][0], spectra['returning_per_mev'][0])), delimiter=',',
               header='energy_mev,inclusive_probability_per_mev,changed_state_probability_per_mev,returning_state_probability_per_mev', comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(axis, spectra['inclusive_per_mev'][0], label='Inclusive')
    ax.plot(axis, spectra['changed_per_mev'][0], label='Changed phonon state')
    ax.set(xlabel='Electron energy loss (meV)', ylabel='Absolute probability / meV',
           title='Si [110] 1×1×1 · 10 K · all 9 internal modes · p=2 · 0–30 mrad', xlim=(-180, 180))
    ax.set_yscale('symlog', linthresh=1e-12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'spectrum.png', dpi=180)
    plt.close(fig)
    json_write(out / 'result.json', dict(status='COMPLETE', config=cfg, max_norm_drift=max(norm_drifts),
               active_modes=len(bath['omega_rad_s']), channels=len(h.basis), thermal_initial_states=initial_indices,
               omitted_initial_probability_bound=omitted, detector_probability=lines['probability'].sum(),
               changed_state_probability=(lines['probability'] * (~lines['returning_state'])[:, :, None]).sum(),
               peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               source_sha256=sha256(Path(__file__)), configuration_digest=fingerprint(cfg),
               validation_scope='Requested finite discretization only; no independent convergence certificate'))
    log('COMPLETE: spectra, absolute channel probabilities, exit waves, and PNG saved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.output / 'inputs/config.json').read_text())
    if (args.output / 'result.json').exists():
        raise FileExistsError('completed run is immutable')
    if args.stage == 'prepare':
        prepare(args.output, cfg)
    else:
        preparation = json.loads((args.output / 'phonons/preparation.json').read_text())
        if not preparation['passed'] or preparation['config_digest'] != fingerprint(cfg):
            raise ValueError('validated matching hiPhive preparation is required')
        run(args.output, cfg)


if __name__ == '__main__':
    main()
