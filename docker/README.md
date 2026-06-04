# IndexTTS2 vLLM Docker

本目录保留 vLLM + CUDA 版镜像构建和本仓库内的 Compose 启动方式。日常使用建议走 WSL 里的持久化配置目录：

```bash
cd /root/docker-config/indextts2-vllm
docker compose up -d
```

默认只启动 API；WebUI 仍保留在 `webui` profile 中，避免 API 和 WebUI 同时加载模型占用双份显存。

## 持久化目录

运行数据放在 Linux 文件系统里，避免 `/mnt/*` 虚拟盘性能问题：

```text
/root/docker-data/indextts2-vllm
├── checkpoints/IndexTTS-2-vLLM
├── cache
├── logs
└── outputs
```

模型权重不打进镜像，需要挂载到 `/models/IndexTTS-2-vLLM`。镜像默认名为：

```text
nanaoto/index-tts:vllm-cu
```

## 启动 WebUI

```bash
cd /mnt/d/CodeWork/Node/IndexTTS2
./docker/start_vllm_webui.sh
```

访问：

```text
http://127.0.0.1:7860
```

WebUI 内已保留“流式输出”，并挂载了 `/api/tts_stream` 表单流式接口。

## 启动 API

```bash
cd /mnt/d/CodeWork/Node/IndexTTS2
./docker/start_vllm_api.sh
```

健康检查：

```text
http://127.0.0.1:6006/health
```

API 支持：

- `POST /tts_url`
- `POST /tts_stream`
- `POST /api/tts_stream`

## 停止

```bash
cd /mnt/d/CodeWork/Node/IndexTTS2
./docker/stop_vllm.sh
```

## 构建镜像

```bash
docker buildx build \
  -f docker/Dockerfile.vllm \
  -t nanaoto/index-tts:vllm-cu \
  .
```

GitHub Actions 工作流 `.github/workflows/docker-publish.yml` 可手动构建并推送到 Docker Hub。需要配置：

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

API 和 WebUI 默认不会同时启动；启动脚本会先停掉旧容器，再启动目标服务，避免重复加载 vLLM 模型占满显存。
