#!/usr/bin/env bash
set -euo pipefail
python -m train --config configs/egoaistpp_cuda.yaml "$@"
