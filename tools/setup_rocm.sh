#!/usr/bin/env bash
# Swap the lean CPU venv for a ROCm one that can train Mamba-3 on the AMD GPU.
#
#   torch / torchvision   ROCm 7.2 wheels from the PyTorch index.
#   mamba-ssm             ChrisLundquist/mamba@rocm, i.e. state-spaces/mamba PR #914
#                         (+ #915). Upstream Mamba-3 calls NVIDIA PTX inline asm for
#                         cos/sin/tanh and passes `maxnreg` to autotune; neither
#                         compiles on AMD. Pinned to one commit so a rebuild is the
#                         same code. Then tools/mamba3_rdna2.patch, below.
#
# mamba-ssm goes in with --no-deps: its tilelang, quack-kernels and
# nvidia-cutlass-dsl are CUDA-only and serve only the MIMO and decode paths, whose
# imports Mamba3 already wraps in try/except. MAMBA_SKIP_CUDA_BUILD skips the
# Mamba-1 CUDA extension, which nothing here uses.
#
# Undo: `uv pip install -r build/venv_cpu_freeze.txt` restores the CPU venv.
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH=2.14.1+rocm7.2
VISION=0.29.1+rocm7.2
INDEX=https://download.pytorch.org/whl/rocm7.2
MAMBA_COMMIT=f6c3ee41723e28f7ea1a9b35b8f2eef8dee182fa

mkdir -p build
[[ -f build/venv_cpu_freeze.txt ]] || uv pip freeze > build/venv_cpu_freeze.txt

uv pip install --extra-index-url "$INDEX" --index-strategy unsafe-best-match \
    "torch==$TORCH" "torchvision==$VISION"
uv pip install einops transformers ninja packaging
# --no-config: pyproject's `override-dependencies = ["mamba-ssm>=2.3.2"]` would
# otherwise replace this requirement. The fork calls itself 2.3.1, so uv silently
# installed PyPI's 2.3.2.post1 -- the PTX build -- in its place.
MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install --no-config --reinstall-package mamba-ssm \
    --no-deps --no-build-isolation \
    "mamba-ssm @ git+https://github.com/ChrisLundquist/mamba@$MAMBA_COMMIT"

# PR #914 was tested on RDNA4. RDNA2 (this RX 6700 XT, gfx1030) has no bf16 dot
# instruction, so the backward kernels, which upstream feeds bf16, fail to compile.
# The patch feeds them fp32 on gfx10xx only. --reinstall-package above normally
# gives a clean copy, but the step is also safe to re-run on its own: a patch that
# reverses cleanly is already applied and is skipped; otherwise --forward applies
# it, and refuses rather than undoes a patch it finds applied.
SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
if patch -p1 -d "$SITE" --dry-run --reverse --force --silent \
    < tools/mamba3_rdna2.patch > /dev/null 2>&1; then
    echo "tools/mamba3_rdna2.patch already applied"
else
    patch -p1 -d "$SITE" --forward < tools/mamba3_rdna2.patch
fi

.venv/bin/python - <<'EOF'
import torch, triton
from mamba_ssm import Mamba3
print("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__)
print("gpu", torch.cuda.is_available(), torch.cuda.get_device_name(0))
EOF
