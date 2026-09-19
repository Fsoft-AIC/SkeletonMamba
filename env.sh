#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash env.sh [--check]

Install the CUDA training, preprocessing, and test dependencies in .venv-cuda.
Requires Linux x86_64, Python 3.10-3.12 with development headers, a C compiler,
and an accessible NVIDIA GPU. Precompiled wheels do not require nvcc.

  --check  Check prerequisites without installing packages.

SKELETONMAMBA_PYTHON selects a Python executable (default: python3).
On a cluster, run CUDA setup inside your allocated GPU job.
EOF
}

if (( $# > 1 )); then
    usage >&2
    exit 2
fi
case "${1:-}" in
    ""|--check) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$repo_dir"
python_bin="${SKELETONMAMBA_PYTHON:-python3}"
"$python_bin" - <<'PY'
import importlib.util
import platform
import sys
import sysconfig
from pathlib import Path

if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit("This pinned environment requires Python 3.10, 3.11 or 3.12.")
if importlib.util.find_spec("pip") is None:
    raise SystemExit("Install pip for the selected Python first (pip >= 22.3 is required).")
import pip
if tuple(map(int, pip.__version__.split(".")[:2])) < (22, 3):
    raise SystemExit("Upgrade pip for the selected Python to >= 22.3 first.")
if sys.platform != "linux" or platform.machine() != "x86_64":
    raise SystemExit("The pinned CUDA wheels require Linux x86_64.")
if not (Path(sysconfig.get_path("include")) / "Python.h").is_file():
    raise SystemExit("Install development headers for this Python (Python.h), required by Triton.")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found. Run setup inside your GPU allocation." >&2
    exit 1
fi
if ! gpu_inventory="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" || [[ -z "$gpu_inventory" ]]; then
    echo "No NVIDIA GPU is accessible. Check the driver or enter your GPU allocation." >&2
    exit 1
fi
if ! command -v "${CC:-gcc}" >/dev/null 2>&1; then
    echo "Triton requires a C compiler. Install/load GCC or set CC to its executable." >&2
    exit 1
fi
printf 'Visible GPU(s):\n%s\n' "$gpu_inventory"
venv_dir="$repo_dir/.venv-cuda"

if [[ "${1:-}" == "--check" ]]; then
    printf 'Setup prerequisites passed. No packages were installed.\n'
    exit 0
fi

# Bootstrap with system pip on clusters where ensurepip is disabled.
"$python_bin" -m venv --without-pip "$venv_dir"
"$python_bin" -m pip --python "$venv_dir/bin/python" install pip==24.3.1 setuptools==78.1.0 wheel==0.48.0
venv_python="$venv_dir/bin/python"
"$venv_python" -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
"$venv_python" - <<'PY'
import torch

if torch.__version__ != "2.5.1+cu124" or torch.version.cuda != "12.4":
    raise SystemExit("Expected the official PyTorch 2.5.1 CUDA 12.4 wheel.")
if torch._C._GLIBCXX_USE_CXX11_ABI:
    raise SystemExit("The pinned Mamba/causal-conv1d wheels require C++11 ABI=False.")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA. Check the GPU allocation and NVIDIA driver.")
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print(f"CUDA runtime ready: {torch.cuda.get_device_name(0)}")
PY
"$venv_python" -m pip install --only-binary=mamba-ssm,causal-conv1d -r requirements-cuda.txt
"$venv_python" -m pip install --no-build-isolation --no-deps -e .
"$venv_python" -m pip check
"$venv_python" - <<'PY'
import causal_conv1d
import causal_conv1d_cuda
import mamba_ssm
from mamba_ssm.modules.mamba2 import Mamba2

print(f"Mamba {mamba_ssm.__version__}: {mamba_ssm.__file__}")
print(f"causal-conv1d {causal_conv1d.__version__}: {causal_conv1d.__file__}")
print("Run python -m scripts.check_cuda before training to verify fused kernels.")
PY
printf '\nActivate with: source %q/bin/activate\n' "$venv_dir"
