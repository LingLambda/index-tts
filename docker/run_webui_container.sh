#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-indextts2:cu128-ds}"
WORK_ROOT="${2:-/root/indextts2-docker}"
PORT="${3:-7860}"
CONTAINER_NAME="${4:-indextts2-webui}"
WEBUI_ARGS=()
if [ "$#" -gt 4 ]; then
  shift 4
  WEBUI_ARGS=("$@")
fi

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

mkdir -p \
  "$WORK_ROOT/src/outputs" \
  "$WORK_ROOT/logs" \
  "$WORK_ROOT/cache/torch_extensions" \
  "$WORK_ROOT/cache/bigvgan_s2mel_cuda_build" \
  "$WORK_ROOT/cache/bigvgan_cuda_build"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

exec docker run --name "$CONTAINER_NAME" \
  --gpus all \
  --init \
  --ipc=host \
  --security-opt seccomp=unconfined \
  -e TORCH_CUDA_ARCH_LIST="8.0;12.0" \
  -p "${PORT}:7860" \
  -v "$WORK_ROOT/data/checkpoints:/app/checkpoints" \
  -v "$WORK_ROOT/src/outputs:/app/outputs" \
  -v "$WORK_ROOT/cache/torch_extensions:/root/.cache/torch_extensions" \
  -v "$WORK_ROOT/cache/bigvgan_s2mel_cuda_build:/app/indextts/s2mel/modules/bigvgan/alias_free_activation/cuda/build" \
  -v "$WORK_ROOT/cache/bigvgan_cuda_build:/app/indextts/BigVGAN/alias_free_activation/cuda/build" \
  "$IMAGE_NAME" \
  python3 -u webui.py --host 0.0.0.0 --port 7860 --fp16 --cuda_kernel --deepspeed "${WEBUI_ARGS[@]}"
