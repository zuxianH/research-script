#!/bin/bash
# Sourced by run.sh/continue.sh; their EXIT trap owns process/socket cleanup.
monitor_run() {
  local target=$1 log=$2 start=$SECONDS pid
  while kill -0 "$IPI_PID" 2>/dev/null; do
    if (( SECONDS - start > ${MONITOR_TIMEOUT_S:-180} )); then
      echo "Timed out; inspect $log and client logs." >&2
      return 1
    fi
    for pid in "${client_pids[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null && kill -0 "$IPI_PID" 2>/dev/null; then
        # i-PI closes clients just before exiting; allow that normal shutdown.
        sleep 1
        if kill -0 "$IPI_PID" 2>/dev/null; then
          echo "LAMMPS client $pid exited before i-PI; inspect client logs." >&2
          return 1
        fi
      fi
    done
    sleep 0.2
  done
  wait "$IPI_PID" || { echo "i-PI failed; inspect $log." >&2; return 1; }
  "$PYTHON_BIN" - "$target" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

target = int(sys.argv[1])
restart = Path('RESTART')
if not restart.exists():
    restart = Path('simulation.restart')
step = int(ET.parse(restart).findtext('step'))
values = np.loadtxt('simulation.out', ndmin=2)
if step != target or int(values[-1, 0]) != target or not np.isfinite(values).all():
    raise SystemExit(f'Failed completion/finite-output check: restart step {step}, target {target}')
print(f'Validated target step {step}; property output is finite.')
PY
}
