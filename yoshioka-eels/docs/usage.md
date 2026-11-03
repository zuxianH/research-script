# Using Yoshioka EELS

Install the package as described in the [README](../README.md). Python imports
use `yoshioka`; the installed `yoshioka-eels` command prints package information.

## Units and array conventions

- Electron kinetic energies are in joules; multiply electronvolts by
  `scipy.constants.elementary_charge`.
- Lengths and displacements are in metres. Wavevectors are angular
  wavevectors in radians per metre, not cycles per metre.
- `TransverseGrid(lx_m, ly_m, nx, ny)` describes a periodic transverse grid.
  Spatial arrays use shape `(ny, nx)`; channel waves have leading channel axes.
- `grid_norm_squared(wave, grid)` includes the transverse pixel area.
- Functions with explicit unit suffixes, such as `_mev` or `_angstrom`, use
  those named units. Check each function's docstring before mixing interfaces.

## Free propagation

```python
import numpy as np
from scipy.constants import elementary_charge
from yoshioka import (
    FreePropagator, TransverseGrid, electron_wavenumber, grid_norm_squared,
)

grid = TransverseGrid(8e-9, 8e-9, 64, 64)
x, y = grid.xy_m
initial = np.exp(-(x**2 + y**2) / (2 * (0.4e-9)**2)).astype(complex)
initial /= np.sqrt(grid_norm_squared(initial, grid))
k0 = electron_wavenumber(60e3 * elementary_charge)
final = FreePropagator(grid, k0).propagate(initial, 10e-9)
print(float(grid_norm_squared(final, grid)))
```

The [free-electron example](../examples/free_electron.py) also checks reverse
propagation. By default it only prints checks. To save wave arrays and a small
JSON record, give it a fresh directory outside the source checkout:

```bash
python examples/free_electron.py --output /path/to/fresh/run
```

An existing output directory is rejected to protect earlier calculations.
Free propagation is an example of the API, not a material EELS calculation.

## Harmonic oscillator

```python
import numpy as np
from yoshioka import HarmonicOscillator, quadrature_vacuum_amplitudes

oscillator = HarmonicOscillator.from_amu_mev(28.085, 60)
amplitudes = quadrature_vacuum_amplitudes(np.sqrt(0.1), 8, 64)
energy_losses_j = oscillator.energy_losses_j(8)
probabilities = np.abs(amplitudes)**2
```

The amplitudes are complex and retain their phases. Finite cutoffs can omit
probability; do not normalize away that deficit when checking convergence.

## Material calculations

The optional material interfaces accept explicit inputs such as structures,
potential files, and fitted force constants. The package supplies numerical
building blocks; the caller defines the specimen, phonon bath, detector,
cutoffs, integration grids, and output locations.

Keep each calculation's inputs and outputs outside this package. Establish
convergence and applicable physical assumptions for that calculation; passing
unit tests alone does not establish material accuracy.
