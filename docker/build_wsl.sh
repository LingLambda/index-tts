#!/usr/bin/env bash
set -euo pipefail

REPO_WIN_PATH="${REPO_WIN_PATH:-/mnt/d/CodeWork/Node/IndexTTS2}"
WORK_ROOT="${WORK_ROOT:-$HOME/indextts2-docker}"
SRC_DIR="$WORK_ROOT/src"
DATA_DIR="$WORK_ROOT/data"
IMAGE_NAME="${IMAGE_NAME:-indextts2:cu128-ds}"
BASE_IMAGE="${BASE_IMAGE:-docker.m.daocloud.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.ustc.edu.cn/pypi/simple}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

mkdir -p "$SRC_DIR" "$DATA_DIR/checkpoints"

echo "==> Sync source into WSL ext4: $SRC_DIR"
rsync -a --delete \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude "checkpoints" \
  --exclude "outputs" \
  --exclude "logs" \
  "$REPO_WIN_PATH"/ "$SRC_DIR"/

echo "==> Sync checkpoints into WSL ext4: $DATA_DIR/checkpoints"
rsync -a --info=progress2 "$REPO_WIN_PATH/checkpoints"/ "$DATA_DIR/checkpoints"/

echo "==> Verify Docker mirror config"
docker info | sed -n '/Registry Mirrors/,+5p' || true
docker pull docker.m.daocloud.io/library/busybox:latest >/dev/null

echo "==> Verify pip mirror config outside image"
python3 -m pip config set global.index-url "$PIP_INDEX_URL"
python3 -m pip config set global.trusted-host mirrors.ustc.edu.cn
python3 -m pip config list
python3 -m pip index versions wheel | head -20

echo "==> Build image: $IMAGE_NAME"
cd "$SRC_DIR"
docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "PIP_INDEX_URL=$PIP_INDEX_URL" \
  --build-arg "INSTALL_VLLM=$INSTALL_VLLM" \
  -f docker/Dockerfile \
  -t "$IMAGE_NAME" \
  .

echo "==> Quick runtime verification"
docker run --rm --gpus all \
  -v "$DATA_DIR/checkpoints:/app/checkpoints" \
  "$IMAGE_NAME" \
  python3 /opt/indextts/verify_runtime.py

cat <<EOF

Image built: $IMAGE_NAME

Run WebUI from Windows PowerShell:
powershell -ExecutionPolicy Bypass -File .\\docker\\run_webui_wsl.ps1

Stop WebUI from Windows PowerShell:
powershell -ExecutionPolicy Bypass -File .\\docker\\stop_webui_wsl.ps1

Run WebUI from an already-open, long-lived WSL shell:
docker rm -f indextts2-webui >/dev/null 2>&1 || true
docker run --name indextts2-webui --gpus all --init --ipc=host \\
  --security-opt seccomp=unconfined \\
  -e TORCH_CUDA_ARCH_LIST="8.0;12.0" \\
  -p 7860:7860 \\
  -v "$DATA_DIR/checkpoints:/app/checkpoints" \\
  -v "$SRC_DIR/outputs:/app/outputs" \\
  -v "$WORK_ROOT/cache/torch_extensions:/root/.cache/torch_extensions" \\
  -v "$WORK_ROOT/cache/bigvgan_s2mel_cuda_build:/app/indextts/s2mel/modules/bigvgan/alias_free_activation/cuda/build" \\
  -v "$WORK_ROOT/cache/bigvgan_cuda_build:/app/indextts/BigVGAN/alias_free_activation/cuda/build" \\
  "$IMAGE_NAME" \\
  python3 -u webui.py --host 0.0.0.0 --fp16 --cuda_kernel --deepspeed

Deep verification, including model load and BigVGAN CUDA kernel compile:
docker run --rm --gpus all \\
  -v "$DATA_DIR/checkpoints:/app/checkpoints" \\
  "$IMAGE_NAME" \\
  python3 /opt/indextts/verify_runtime.py --compile-cuda-kernel --load-model

Build an experimental official-accel image with flash-attn:
bash docker/build_accel_wsl.sh

Then deep-verify it with:
docker run --rm --gpus all \\
  -v "$DATA_DIR/checkpoints:/app/checkpoints" \\
  indextts2:cu128-ds-accel \\
  python3 /opt/indextts/verify_runtime.py --compile-cuda-kernel --load-model --use-accel
EOF
