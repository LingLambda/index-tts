#!/usr/bin/env bash
set -euo pipefail

REPO_WIN_PATH="${REPO_WIN_PATH:-/mnt/d/CodeWork/Node/IndexTTS2}"
BASE_IMAGE="${BASE_IMAGE:-indextts2:cu128-ds}"
IMAGE_NAME="${IMAGE_NAME:-indextts2:cu128-ds-accel}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.ustc.edu.cn/pypi/simple}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
MAX_JOBS="${MAX_JOBS:-8}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

docker image inspect "$BASE_IMAGE" >/dev/null

echo "==> Build accel image: $IMAGE_NAME from $BASE_IMAGE"
cd "$REPO_WIN_PATH"
docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "PIP_INDEX_URL=$PIP_INDEX_URL" \
  --build-arg "FLASH_ATTN_VERSION=$FLASH_ATTN_VERSION" \
  --build-arg "MAX_JOBS=$MAX_JOBS" \
  -f docker/Dockerfile.accel \
  -t "$IMAGE_NAME" \
  .

echo "==> Quick accel runtime verification"
docker run --rm --gpus all \
  "$IMAGE_NAME" \
  python3 /opt/indextts/verify_runtime.py --use-accel

cat <<EOF

Accel image built: $IMAGE_NAME

Deep verification:
docker run --rm --gpus all \\
  -v "\$HOME/indextts2-docker/data/checkpoints:/app/checkpoints" \\
  "$IMAGE_NAME" \\
  python3 /opt/indextts/verify_runtime.py --compile-cuda-kernel --load-model --use-accel

Run WebUI with official accel from a long-lived WSL shell:
docker rm -f indextts2-webui-accel >/dev/null 2>&1 || true
docker run --name indextts2-webui-accel --gpus all --init --ipc=host \\
  --security-opt seccomp=unconfined \\
  -e TORCH_CUDA_ARCH_LIST="8.0;12.0" \\
  -p 7861:7860 \\
  -v "\$HOME/indextts2-docker/data/checkpoints:/app/checkpoints" \\
  -v "\$HOME/indextts2-docker/src/outputs:/app/outputs" \\
  -v "\$HOME/indextts2-docker/cache/torch_extensions:/root/.cache/torch_extensions" \\
  -v "\$HOME/indextts2-docker/cache/bigvgan_s2mel_cuda_build:/app/indextts/s2mel/modules/bigvgan/alias_free_activation/cuda/build" \\
  -v "\$HOME/indextts2-docker/cache/bigvgan_cuda_build:/app/indextts/BigVGAN/alias_free_activation/cuda/build" \\
  "$IMAGE_NAME" \\
  python3 -u webui.py --host 0.0.0.0 --port 7860 --fp16 --cuda_kernel --deepspeed --use_accel
EOF
