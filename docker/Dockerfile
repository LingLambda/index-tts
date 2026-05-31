ARG BASE_IMAGE=docker.m.daocloud.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple
ARG INSTALL_VLLM=0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=mirrors.ustc.edu.cn \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/checkpoints/hf_cache \
    HF_HUB_CACHE=/app/checkpoints/hf_cache \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
    TORCH_CUDA_ARCH_LIST="8.0;12.0" \
    CUDA_HOME=/usr/local/cuda

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@https://mirrors.ustc.edu.cn/ubuntu/@g; s@http://security.ubuntu.com/ubuntu/@https://mirrors.ustc.edu.cn/ubuntu/@g' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        rsync \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip config set global.index-url "${PIP_INDEX_URL}" \
    && python3 -m pip config set global.trusted-host mirrors.ustc.edu.cn \
    && python3 -m pip install --upgrade pip setuptools wheel packaging ninja tomli \
    && python3 -m pip config list \
    && python3 -m pip index versions wheel | head -20

WORKDIR /app

COPY pyproject.toml README.md LICENSE LICENSE_ZH.txt MANIFEST.in ./

# Install CUDA-enabled PyTorch from the configured PyPI mirror first. Torch 2.8
# wheels depend on CUDA 12.8 nvidia-* packages on Linux, which keeps pip on USTC.
RUN python3 -m pip install \
        "torch==2.8.0" \
        "torchaudio==2.8.0" \
    && python3 - <<'PY' >/tmp/indextts-requirements.txt
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

with open("pyproject.toml", "rb") as fh:
    project = tomllib.load(fh)["project"]

requirements = []
requirements.extend(project["dependencies"])
requirements.extend(project["optional-dependencies"]["webui"])

for req in requirements:
    normalized = req.strip().lower()
    if normalized.startswith(("torch==", "torchaudio==")):
        continue
    print(req)
PY
RUN python3 -m pip install -r /tmp/indextts-requirements.txt \
    && DS_BUILD_OPS=0 python3 -m pip install --no-build-isolation "deepspeed==0.17.1" \
    && if [ "${INSTALL_VLLM}" = "1" ]; then \
        echo "INSTALL_VLLM=1 is disabled for this stable image: current vLLM releases conflict with the IndexTTS2 pinned runtime. Build a separate experimental vLLM image instead." >&2; \
        exit 2; \
       fi \
    && python3 -m pip check

COPY indextts ./indextts
COPY tools ./tools
COPY examples ./examples
COPY webui.py ./
RUN python3 -m pip install -e ".[webui]" --no-deps \
    && python3 -m pip check

COPY docker/verify_runtime.py /opt/indextts/verify_runtime.py

EXPOSE 7860

CMD ["python3", "webui.py", "--host", "0.0.0.0", "--fp16", "--cuda_kernel", "--deepspeed"]
