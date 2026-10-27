# Source from any directory to select the isolated TRPMD runtime.
_silicon_runtime="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.venv"
export SCIENCE_ACTIVATE=/dev/null
export PYTHON_BIN="$_silicon_runtime/bin/python"
export LMP_BIN="$_silicon_runtime/bin/lmp"
export IPI_BIN="$_silicon_runtime/bin/i-pi"
unset _silicon_runtime
