# Yoshioka EELS

Python tools for paraxial electron propagation, coupled electron–phonon
channels, and electron energy-loss spectra. Install the distribution
`yoshioka-eels` and import it as `yoshioka`.

## Install

From this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m yoshioka --version
```

The core dependencies are NumPy, SciPy, Matplotlib, and psutil. Optional
material interfaces can be installed as needed:

```bash
.venv/bin/python -m pip install -e '.[elastic-reference]'
.venv/bin/python -m pip install -e '.[lattice-reference]'
```

`elastic-reference` provides abTEM's silicon potential table and ASE.
`lattice-reference` provides LAMMPS, MPI, hiPhive, phonopy, and ASE. Material
structures, force constants, and potential files are supplied by the caller.

## Quick start

```python
import numpy as np
from scipy.constants import elementary_charge
from yoshioka import FreePropagator, TransverseGrid, electron_wavenumber

grid = TransverseGrid(8e-9, 8e-9, 64, 64)
x, y = grid.xy_m
wave = np.exp(-(x**2 + y**2) / (2 * (0.4e-9)**2)).astype(complex)
k0 = electron_wavenumber(60e3 * elementary_charge)
propagated = FreePropagator(grid, k0).propagate(wave, 10e-9)
```

Run the complete example to check normalization and reversibility:

```bash
.venv/bin/python examples/free_electron.py
```

It prints numerical checks and saves no files unless `--output` is supplied.
See [usage and units](docs/usage.md) and the [API guide](docs/api.md).

## Layout

```text
yoshioka/       Python package
tests/          Unit tests and independent analytical references
examples/       Small runnable examples
docs/           Usage and API documentation
pyproject.toml  Package metadata and dependencies
```

This directory contains package source, documentation, examples, and tests.
Keep simulation inputs, research runners, results, and benchmark reports in
separate research directories.

## Tests

From this directory, install the elastic extra for the material-field tests:

```bash
.venv/bin/python -m pip install -e '.[elastic-reference]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

The tests use analytical references and small synthetic inputs; no saved
research results or benchmark checkout is required. They check software and
numerical identities, not convergence of a particular material calculation.
