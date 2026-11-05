#!/usr/bin/env python3
"""Si [110] finite-cell SNAP/hiPhive harmonic, full 3D nonlinear channel run.

Keep inputs and generated arrays in the external --output directory. Each
potential matrix element uses the requested working-Fock exponential before
projection into the total-occupation basis; longitudinal phases are retained.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from scipy import constants
from scipy import fft as threaded_fft
from scipy.linalg import eigh_tridiagonal
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


def preparation_contract(cfg):
    """Static force fit and finite bath identity, independent of electron grids."""
    keys = ('system', 'specimen_repetitions', 'mass_u', 'fit_supercell',
            'harmonic_cutoff_angstrom', 'fit_sigma_angstrom', 'training_pairs',
            'validation_pairs', 'fit_seed', 'input_sha256')
    return dict(geometry='static zero-pressure orthorhombic SNAP; uniform isotope mass',
                settings={key: cfg[key] for key in keys})


def calculation_settings(cfg):
    execution_keys = {'field_workers', 'propagation_workers', 'hamiltonian_threads', 'fft_threads',
                      'resource_request', 'maximum_rss_bytes', 'maximum_seconds'}
    return {k: v for k, v in cfg.items() if k not in execution_keys}


def validate_preparation(out, cfg):
    folder = out / 'phonons'
    record = json.loads((folder / 'preparation.json').read_text())
    saved = json.loads((folder / 'contract.json').read_text())
    if not record['passed'] or saved['phonon_contract'] != preparation_contract(cfg):
        raise ValueError('validated phonon inputs/settings do not match this calculation')
    for name, digest in saved['files'].items():
        if sha256(folder / name) != digest:
            raise ValueError('validated phonon artifact changed: ' + name)
    for name, digest in cfg['input_sha256'].items():
        file = out / ('inputs' if name.endswith('.lmp') else 'models') / name
        if sha256(file) != digest:
            raise ValueError('supplied material input changed: ' + name)
    return record


def reuse_force_constants(out, cfg, previous):
    """Reuse the validated bulk fit, constructing the requested finite bath anew."""
    from yoshioka.lattice_dynamics import RealSpaceForceConstants, finite_periodic_modes
    previous = Path(previous)
    old_cfg = json.loads((previous / 'inputs/config.json').read_text())
    old_record = validate_preparation(previous, old_cfg)
    old_fit, new_fit = preparation_contract(old_cfg), preparation_contract(cfg)
    old_fit['settings'].pop('specimen_repetitions')
    new_fit['settings'].pop('specimen_repetitions')
    if old_fit != new_fit:
        raise ValueError('force-fit settings or material inputs differ from the validated source')
    folder = out / 'phonons'
    folder.mkdir(exist_ok=True)
    for source in (previous / 'phonons').iterdir():
        if source.is_file() and source.name not in {'bath.npz', 'preparation.json', 'contract.json'}:
            shutil.copy2(source, folder / source.name)
    model = RealSpaceForceConstants.load(folder / 'hiphive_model.npz')
    reps = np.asarray(cfg['specimen_repetitions'], dtype=int)
    finite = finite_periodic_modes(model, reps)
    np.savez_compressed(folder / 'bath.npz', **finite, cell_m=reps[:, None] * model.cell_m)
    record = dict(old_record, config_digest=fingerprint(cfg),
                  mode_quanta_mev=finite['omega_rad_s'] * constants.hbar / (constants.e * 1e-3),
                  active_mode_count=len(finite['omega_rad_s']), specimen_atoms=len(finite['positions_m']),
                  specimen_repetitions=reps.tolist(),
                  boundary='x'.join(map(str, reps)) + ' periodic nuclear cell; vacuum-padded z electron potential',
                  reused_force_constants_from=str(previous),
                  source_preparation_sha256=sha256(previous / 'phonons/preparation.json'))
    if len(finite['omega_rad_s']) != 3 * len(finite['positions_m']) - 3:
        raise ValueError('finite bath does not contain all internal modes')
    json_write(folder / 'preparation.json', record)
    json_write(folder / 'contract.json', dict(phonon_contract=preparation_contract(cfg),
        files={p.name: sha256(p) for p in folder.iterdir() if p.is_file()}))
    log(f'Reused validated force fit; built {len(finite["omega_rad_s"])} internal modes for {reps.tolist()}')


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
        validate_preparation(out, cfg)
        log('Using completed phonon preparation')
        return
    if cfg.get('reuse_force_constants_from'):
        reuse_force_constants(out, cfg, cfg['reuse_force_constants_from'])
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
    finite_unit = finite_periodic_modes(model, (1, 1, 1))
    specimen_reps = np.asarray(cfg['specimen_repetitions'], dtype=int)
    finite = finite_periodic_modes(model, specimen_reps)
    np.savez_compressed(folder / 'bath.npz', **finite, cell_m=specimen_reps[:, None] * np.asarray(unit.cell) * 1e-10)
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
    unit_quanta = finite_unit['omega_rad_s'] * constants.hbar / (constants.e * 1e-3)
    fit_energy_error = float(np.max(abs(unit_quanta - fd_energies[-1])))
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
                  specimen_atoms=len(finite['positions_m']), specimen_repetitions=specimen_reps.tolist(),
                  boundary='x'.join(map(str, specimen_reps)) + ' periodic nuclear cell; vacuum-padded z electron potential')
    record['passed'] = bool(heldout < 0.01 and odd_error < 0.001 and fit_energy_error < 0.05 and fd_change < 0.05
                            and np.max(abs(equilibrium['forces_ev_per_angstrom'])) < 1e-6
                            and np.max(abs(equilibrium['pressure_bar_voigt'])) < 0.1
                            and len(quanta) == 3 * len(finite['positions_m']) - 3)
    json_write(folder / 'preparation.json', record)
    if not record['passed']:
        raise ValueError('SNAP/hiPhive independent validation failed; inspect preparation.json')
    json_write(folder / 'contract.json', dict(phonon_contract=preparation_contract(cfg),
        files={p.name: sha256(p) for p in folder.iterdir() if p.is_file()}))
    log(f'Phonons validated: {len(quanta)} internal modes; fit frequency error {fit_energy_error:.6g} meV')


_FIELD_CONTEXT = None


def single_occupation_phase_blocks(basis, xi, dimension):
    """Exact p=1 product of the same working-Fock single-mode blocks.

    Factor out the vacuum product instead of multiplying a full CxC matrix
    for every mode. Tiny denominators fall back to the generic exact product.
    """
    if np.any(np.sum(basis, axis=1) > 1) or np.any(basis[0]):
        return phase_blocks(basis, xi, dimension)
    nodes, vectors = eigh_tridiagonal(np.zeros(dimension), np.sqrt(np.arange(1, dimension)))
    phase = np.exp(-1j * np.asarray(xi)[:, :, None] * nodes)
    f00 = phase @ (vectors[0] ** 2)
    if np.any(abs(f00) < 1e-10):
        return phase_blocks(basis, xi, dimension)
    f01 = phase @ (vectors[0] * vectors[1])
    f11 = phase @ (vectors[1] ** 2)
    occupied = np.argmax(basis[1:], axis=1)
    ratio = np.column_stack((np.ones(len(xi)), (f01 / f00)[:, occupied]))
    result = np.prod(f00, axis=1)[:, None, None] * ratio[:, :, None] * ratio[:, None, :]
    diagonal = np.arange(1, len(basis))
    result[:, diagonal, diagonal] = np.prod(f00, axis=1)[:, None] * (f11 / f00)[:, occupied]
    return result


def initialize_field_worker(context):
    """One copy of the small bath/grids per process; no nested BLAS threads."""
    global _FIELD_CONTEXT
    from threadpoolctl import threadpool_limits
    _FIELD_CONTEXT = dict(context, potential=LobatoSilicon.from_abtem_table(),
                          thread_limit=threadpool_limits(limits=1))


def build_field_tile(task):
    """Write one deterministic canonical tile, never the shared complete field."""
    context = _FIELD_CONTEXT
    selected, path = task
    path = Path(path)
    record_path = path.with_suffix('.json')
    identity = dict(field_digest=context['digest'], indices=selected.tolist())
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record['identity'] != identity or record['sha256'] != sha256(path):
            raise ValueError('field tile identity/content mismatch: ' + str(path))
        return record
    if path.exists() or path.with_suffix('.tmp.npy').exists():
        raise FileExistsError('uncommitted tile exists; preserve and inspect: ' + str(path))
    started = time.monotonic()
    bath, cfg, basis = context['bath'], context['cfg'], context['basis']
    Q, kz = context['Q'][selected], context['kz']
    steps, fields = cfg['steps'], cfg['field_points']
    cell, length, dz = bath['cell_m'], context['length'], context['dz']
    wavevectors = np.empty((len(selected), fields, 3))
    wavevectors[:, :, :2], wavevectors[:, :, 2] = Q[:, None, :], kz
    kernel = context['potential'].fourier_energy_j_m3(np.sum(wavevectors**2, axis=-1)) / np.prod(np.diag(cell)[:2]) / length
    folded = np.zeros((len(selected), steps, len(basis), len(basis)), complex)
    for atom, pos in enumerate(bath['positions_m']):
        xi = wavevectors @ bath['displacements_m'][atom]
        phase_function = single_occupation_phase_blocks if cfg['p'] == 1 else phase_blocks
        blocks = phase_function(basis, xi.reshape(-1, xi.shape[-1]), cfg['working_dimension'])
        blocks = blocks.reshape(len(selected), fields, len(basis), len(basis))
        coefficient = kernel * np.exp(-1j*(Q @ pos[:2])[:, None] - 1j*kz*(pos[2] + cfg['padding_m'] - dz/2))
        blocks *= coefficient[:, :, None, None]
        for s in range(steps):
            folded[:, s] += blocks[:, context['frequencies'] % steps == s].sum(axis=1)
    values = np.fft.ifft(folded, axis=1) * steps
    if selected[0] == 0:
        residual = np.linalg.norm(values[0].imag) / max(np.linalg.norm(values[0].real), np.finfo(float).tiny)
        if residual > 1e-11:
            raise ValueError(f'zero-transfer field is not real: {residual}')
        values[0] = values[0].real
    if not np.all(np.isfinite(values)):
        raise ValueError('non-finite interaction field tile')
    temporary = path.with_suffix('.tmp.npy')
    np.save(temporary, values.swapaxes(0, 1))
    temporary.replace(path)
    record = dict(identity=identity, file=path.name, sha256=sha256(path),
                  seconds=time.monotonic()-started,
                  worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    json_write(record_path, record)
    return record


class FullDisplacementAction:
    """Exact working-Fock phase blocks, all Cartesian displacements, FFT Galerkin.

    Frequency folding evaluates the declared finite z Fourier field at the
    independent propagation midpoints. M=2N-1 prevents transverse cyclic alias.
    Only one dense local CxC channel matrix field is resident during propagation.
    """
    def __init__(self, bath, cfg, folder, tile=4, validated_field=None):
        self.bath, self.cfg = bath, cfg
        self.n, self.steps = cfg['n'], cfg['steps']
        fields, dimension = cfg['field_points'], cfg['working_dimension']
        self.basis = channel_basis(len(bath['omega_rad_s']), cfg['p'])
        self.c = len(self.basis)
        if cfg['n'] < 1 or any(cfg[key] < 1 or cfg[key] % 2 != 1 for key in ('steps', 'field_points')):
            raise ValueError('positive electron grid and positive odd propagation/field grids required')
        if cfg['working_dimension'] <= cfg['p']:
            raise ValueError('working oscillator dimension must exceed the occupation cutoff')
        self.shape = (self.c, self.n, self.n)
        self.pad = 2 * self.n - 1
        self.threads = cfg.get('hamiltonian_threads', 1)
        self.fft_threads = cfg.get('fft_threads', self.threads)
        if not isinstance(self.threads, int) or self.threads < 1:
            raise ValueError('hamiltonian_threads must be a positive integer')
        if not isinstance(self.fft_threads, int) or self.fft_threads < 1:
            raise ValueError('fft_threads must be a positive integer')
        self.thread_pool = ThreadPoolExecutor(max_workers=self.threads) if self.threads > 1 else None
        workers = cfg.get('field_workers', 1)
        if not isinstance(workers, int) or workers < 1:
            raise ValueError('field_workers must be a positive integer')
        # Include simultaneous Fourier/FFT/local field copies and work vectors.
        estimated_rss = 5 * self.pad**2 * self.c**2 * 16 + 64 * self.c * self.pad**2 * 16 + 2 * 1024**3
        estimated_rss += workers * (3 * tile * fields * self.c**2 * 16 + 256 * 1024**2)
        if estimated_rss > cfg.get('maximum_rss_bytes', 64 * 1024**3):
            raise MemoryError(f'conservative local-field estimate {estimated_rss} exceeds configured budget')
        cell = bath['cell_m']
        self.length = cell[2, 2] + 2 * cfg['padding_m']
        self.dz = self.length / self.steps
        energy = cfg['electron_energy_ev'] * constants.e
        self.hv = constants.hbar * electron_speed(energy)
        self.channel_j = self.basis @ (constants.hbar * bath['omega_rad_s'])
        beams = np.arange(self.n) - self.n // 2
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
        contract = dict(config=calculation_settings(cfg),
                        bath={k: np.asarray(v).tolist() for k, v in bath.items()},
                        source_sha256=sha256(Path(__file__)))
        contract_digest = fingerprint(contract)
        manifest = folder / 'manifest.json'
        path = folder / 'channel_fields.npy'
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            stat = path.stat()
            signature = dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            parent_validated = validated_field is not None and validated_field == dict(
                digest=contract_digest, sha256=saved['sha256'], stat=signature)
            if saved['digest'] != contract_digest or (not parent_validated and saved['sha256'] != sha256(path)):
                raise ValueError('field cache configuration/source/content mismatch')
        else:
            if path.exists():
                raise FileExistsError('uncommitted field cache exists; preserve and inspect before restarting')
            chunks = folder / 'chunks'
            chunks.mkdir(exist_ok=True)
            context = dict(bath=bath, cfg=cfg, basis=self.basis, Q=Q, kz=kz,
                           frequencies=frequencies, length=self.length, dz=self.dz, digest=contract_digest)
            tasks = [(canonical[lo:lo+tile], str(chunks/f'tile_{lo:06d}.npy'))
                     for lo in range(0, len(canonical), tile)]
            log(f'Building {len(canonical)} canonical transfers in {len(tasks)} tiles with {workers} worker(s)')
            if workers == 1:
                initialize_field_worker(context)
                records = [build_field_tile(task) for task in tasks]
            else:
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                         initializer=initialize_field_worker, initargs=(context,)) as pool:
                    futures = [pool.submit(build_field_tile, task) for task in tasks]
                    records = []
                    for future in as_completed(futures):
                        records.append(future.result())
                        if len(records) % 10 == 0 or len(records) == len(tasks):
                            log(f'Channel field tiles: {len(records)}/{len(tasks)} complete')
            covered = sorted(i for record in records for i in record['identity']['indices'])
            if covered != canonical.tolist():
                raise ValueError('field tiles do not cover canonical transfers exactly once')
            temporary = folder / 'channel_fields.tmp.npy'
            if temporary.exists():
                raise FileExistsError('uncommitted assembled field exists; preserve and inspect')
            field = np.lib.format.open_memmap(temporary, mode='w+', dtype=complex,
                                             shape=(self.steps, len(Q), self.c, self.c))
            for record in records:
                selected = np.array(record['identity']['indices'])
                file = chunks / record['file']
                if sha256(file) != record['sha256']:
                    raise ValueError('completed tile checksum changed')
                values = np.load(file, allow_pickle=False)
                if values.shape != (self.steps, len(selected), self.c, self.c) or values.dtype != np.complex128 or not np.all(np.isfinite(values)):
                    raise ValueError('invalid completed field tile')
                field[:, selected] = values
                field[:, opposite[selected]] = values.conj()
            field.flush()
            del field
            temporary.replace(path)
            json_write(manifest, dict(digest=contract_digest, sha256=sha256(path),
                                     channel_count=self.c, full_cartesian_displacements=True,
                                     workers=workers, chunks=records, estimated_peak_rss_bytes=estimated_rss))
        self.fields = np.load(path, mmap_mode='r')
        stat = path.stat()
        self.field_identity = dict(digest=contract_digest, sha256=json.loads(manifest.read_text())['sha256'],
            stat=dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        self.cached_index, self.potential = None, None

    def set_slice(self, index):
        if self.cached_index == index:
            return
        self.potential = None
        fourier = np.array(self.fields[index])
        self.trace_potential = np.trace(fourier[0]).real * self.n**2
        local = threaded_fft.ifft2(fourier.reshape(self.pad, self.pad, self.c, self.c), axes=(0, 1), workers=self.fft_threads) * self.pad**2
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
        position = threaded_fft.ifft2(embedded, axes=(-2, -1), norm='ortho', workers=self.fft_threads).reshape(self.c, -1)
        if self.thread_pool is None:
            product = np.einsum('qij,jq->iq', self.potential, position, optimize=True)
        else:
            product = np.empty_like(position)
            def contract_block(bounds):
                lo, hi = bounds
                product[:, lo:hi] = np.einsum('qij,jq->iq', self.potential[lo:hi], position[:, lo:hi], optimize=True)
            edges = np.linspace(0, self.pad**2, self.threads + 1, dtype=int)
            list(self.thread_pool.map(contract_block, zip(edges[:-1], edges[1:])))
        reciprocal = threaded_fft.fft2(product.reshape(embedded.shape), axes=(-2, -1), norm='ortho', workers=self.fft_threads)
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


def save_exit_probabilities(path, waves, **metadata):
    """Save unweighted momentum probabilities; preserve float64 precision."""
    probability = np.abs(waves).astype(np.float64, copy=False)
    np.square(probability, out=probability)
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temporary, exit_probabilities=probability, **metadata)
    temporary.replace(path)


def prune_completed_checkpoints(folder, initial_indices, probability_file):
    """Remove only verified completed waves after the final probability save."""
    if not probability_file.is_file():
        raise FileNotFoundError('final probability output must exist before checkpoint cleanup')
    completed = []
    for initial_index in initial_indices:
        state_path = folder / f'initial_{initial_index}.json'
        state = json.loads(state_path.read_text())
        expected = f'initial_{initial_index}_slice_{state["total_slices"]}.npy'
        if state['completed_slices'] != state['total_slices'] or state['wave_file'] != expected:
            raise ValueError('cleanup requires this initial state\'s completed checkpoint')
        wave_path = folder / expected
        if sha256(wave_path) != state['wave_sha256']:
            raise ValueError('completed checkpoint checksum mismatch; preserving wave')
        completed.append((state_path, state, wave_path))
    # Validate every record before removing any completed wave.
    for state_path, state, wave_path in completed:
        wave_path.unlink()
        state.update(wave_file=None, wave_sha256=None, wave_retained=False,
                     final_probability_file='../' + probability_file.name)
        json_write(state_path, state)


def run(out, cfg):
    from parallel_propagation import propagate_initial_states
    started = time.monotonic()
    with np.load(out / 'phonons/bath.npz') as data:
        bath = {k: data[k] for k in ('positions_m', 'cell_m', 'displacements_m', 'omega_rad_s')}
    h = FullDisplacementAction(bath, cfg, out / 'fields')
    log(f'Interaction fields ready after {time.monotonic() - started:.3f} seconds')
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
    state_workers = min(cfg.get('propagation_workers', 1), len(initial_indices))
    per_state_bytes = 5 * h.pad**2 * h.c**2 * 16 + 64 * h.c * h.pad**2 * 16 + 512 * 1024**2
    propagation_bytes = state_workers * per_state_bytes + h.fields.nbytes + 2 * 1024**3
    if propagation_bytes > cfg.get('maximum_rss_bytes', 64 * 1024**3):
        raise MemoryError('aggregate parallel-propagation memory estimate exceeds configured node budget')
    cpus = cfg.get('resource_request', {}).get('cpus')
    if cpus and (state_workers * cfg.get('hamiltonian_threads', 1) > cpus or cfg.get('field_workers', 1) > cpus):
        raise ValueError('declared process/thread plan exceeds allocated CPU count')
    log(f'Propagating {len(initial_indices)} thermal initial states with {state_workers} processes, '
        f'{cfg.get("hamiltonian_threads", 1)} contraction threads each; omitted probability bound {omitted:.3g}')
    checkpoint = out / 'checkpoints'
    checkpoint.mkdir(exist_ok=True)
    helper_source = Path(__file__).with_name('parallel_propagation.py')
    contract_context = dict(config=calculation_settings(cfg), source=sha256(Path(__file__)),
                            propagation_source=sha256(helper_source), bath=sha256(out / 'phonons/bath.npz'))
    waves, norm_drifts = propagate_initial_states(FullDisplacementAction, bath, cfg, out / 'fields',
        checkpoint, initial_indices, h.field_identity, contract_context)
    angle, masks = angular_detector_masks(h.qxy, electron_wavenumber(cfg['electron_energy_ev'] * constants.e), ((0, 30),))
    lines = transition_lines(np.array(waves), initial_indices, h.basis, constants.hbar * bath['omega_rad_s'], weights[initial_indices], masks)
    axis = np.linspace(-180, 180, 7201)
    spectra = broaden_transition_lines(lines, axis, cfg['fwhm_mev'])
    probability_file = out / 'exit_probabilities.npz'
    save_exit_probabilities(probability_file, waves, initial_indices=initial_indices,
        basis=h.basis, qxy_m_inv=h.qxy, angles_mrad=angle, detector_masks=masks,
        detector_bounds_mrad=np.array([[0., 30.]]), thermal_weights=weights,
        executed_initial_weights=weights[initial_indices], omitted_initial_probability=omitted,
        mode_quanta_j=constants.hbar*bath['omega_rad_s'])
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
    specimen = '×'.join(map(str, cfg['specimen_repetitions']))
    ax.set(xlabel='Electron energy loss (meV)', ylabel='Absolute probability / meV',
           title=f'Si [110] {specimen} · {cfg["temperature_k"]:g} K · all {len(bath["omega_rad_s"])} internal modes · p={cfg["p"]} · 0–30 mrad',
           xlim=(-180, 180))
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
               child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
               exit_output_file=probability_file.name, exit_output_representation='unweighted |A|^2, float64',
               source_sha256=sha256(Path(__file__)), configuration_digest=fingerprint(cfg),
               propagation_source_sha256=sha256(helper_source), elapsed_seconds=time.monotonic()-started,
               validation_scope='Requested finite discretization only; no independent convergence certificate'))
    prune_completed_checkpoints(checkpoint, initial_indices, probability_file)
    log('COMPLETE: spectra, momentum probabilities, and PNG saved; temporary complex checkpoints removed')


def benchmark(out, cfg):
    """Measure 1/2/4 processes on the actual field workload, then retain the best."""
    with np.load(out / 'phonons/bath.npz') as data:
        bath = {k: data[k] for k in ('positions_m', 'cell_m', 'displacements_m', 'omega_rad_s')}
    identity = dict(config_digest=fingerprint(calculation_settings(cfg)), source_sha256=sha256(Path(__file__)),
                    bath_sha256=sha256(out / 'phonons/bath.npz'))
    if (out / 'benchmark.json').exists():
        saved = json.loads((out / 'benchmark.json').read_text())
        if saved['identity'] != identity:
            raise ValueError('saved benchmark has different source, bath, or numerical settings')
        best = saved['selected_workers']
        if not (out / 'fields').exists():
            (out / 'benchmark' / f'workers_{best}').rename(out / 'fields')
        return dict(cfg, field_workers=best)
    reference, timings = None, []
    for workers in (1, 2, 4):
        folder = out / 'benchmark' / f'workers_{workers}'
        saved_timing = folder / 'timing.json'
        if folder.exists() and not saved_timing.exists():
            raise FileExistsError('partial benchmark has no comparable full-build timing; preserve it before a fresh measurement')
        settings = dict(cfg, field_workers=workers)
        started = time.monotonic()
        h = FullDisplacementAction(bath, settings, folder)
        elapsed = time.monotonic() - started
        if saved_timing.exists():
            timing = json.loads(saved_timing.read_text())
            if timing['identity'] != identity:
                raise ValueError('saved field timing has different numerical inputs')
            elapsed = timing['seconds']
        else:
            json_write(saved_timing, dict(identity=identity, seconds=elapsed))
        if reference is None:
            reference = np.array(h.fields)
        difference = np.linalg.norm(np.asarray(h.fields) - reference) / np.linalg.norm(reference)
        if not np.isfinite(difference) or difference > 1e-13:
            raise ValueError('parallel field benchmark disagrees with serial fields')
        for s in range(h.steps):
            h.set_slice(s)
        timings.append(dict(workers=workers, seconds=elapsed, relative_field_error=float(difference)))
        del h
        log(f'Field benchmark: {workers} worker(s), {elapsed:.3f} seconds, relative error {difference:.3g}')
    best = min(timings, key=lambda item: item['seconds'])['workers']
    if (out / 'fields').exists():
        raise FileExistsError('existing run field directory must be preserved before selection')
    json_write(out / 'benchmark.json', dict(identity=identity, timings=timings, selected_workers=best,
        speedup_vs_serial=timings[0]['seconds'] / min(item['seconds'] for item in timings),
        note='Elapsed times include process startup, tile writes, assembly, and hashing; all workers use one BLAS thread'))
    (out / 'benchmark' / f'workers_{best}').rename(out / 'fields')
    return dict(cfg, field_workers=best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run', 'benchmark-and-run'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.output / 'inputs/config.json').read_text())
    if (args.output / 'result.json').exists():
        raise FileExistsError('completed run is immutable')
    if args.stage == 'prepare':
        prepare(args.output, cfg)
    else:
        validate_preparation(args.output, cfg)
        if args.stage == 'benchmark-and-run':
            cfg = benchmark(args.output, cfg)
        elif (args.output / 'benchmark.json').exists():
            cfg = dict(cfg, field_workers=json.loads((args.output / 'benchmark.json').read_text())['selected_workers'])
        run(args.output, cfg)


if __name__ == '__main__':
    main()
