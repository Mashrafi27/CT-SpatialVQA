#!/usr/bin/env bash
set -euo pipefail

# Ensure the active conda env's pip/python are used.
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX is not set. Activate a conda env first."
  exit 1
fi

export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

echo "Using python: $(command -v python)"
echo "Using pip:    $(command -v pip)"
python -m pip --version
