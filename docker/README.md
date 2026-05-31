# IndexTTS2 Docker Build Notes

This build is intended to run from WSL ext4 storage, not directly from `/mnt/*`.

## Known Good Build

Recorded on 2026-06-01, WSL Ubuntu, RTX 5060 Ti.

- Stable image: `indextts2:cu128-ds`
  - Image id: `sha256:26a89ea022c4cd5e6e720e318b1cabe75cdfe8d034ca3d9c8b03bd74d04b7af4`
  - Created: `2026-06-01T00:57:14+08:00`
  - Verified stack: `torch==2.8.0+cu128`, `torchaudio==2.8.0+cu128`, `deepspeed==0.17.1`, `gradio==5.45.0`, `transformers==4.52.1`, `modelscope==1.27.0`
- Official acceleration image: `indextts2:cu128-ds-accel`
  - Image id: `sha256:b4542c6ad768655505762391844e89822902a534438491727532953ccdfbf024`
  - Created: `2026-06-01T01:53:16+08:00`
  - Additional package: `flash-attn==2.8.3`
  - Verified command: `python3 /opt/indextts/verify_runtime.py --compile-cuda-kernel --load-model --use-accel`
  - Verified WebUI command: `python3 -u webui.py --host 0.0.0.0 --port 7860 --fp16 --cuda_kernel --deepspeed --use_accel`
  - Runtime checks passed: CUDA 12.8, GPU capability `(12, 0)`, BigVGAN CUDA kernel, DeepSpeed inference kernel, official GPT2 acceleration engine, `pip check`, `/api/health`, and multipart `/api/tts_stream`.
  - Streaming API smoke test: first wav part arrived at `18.31s`; later parts arrived before request completion, confirming incremental audio output.

This `indextts2:cu128-ds-accel` image is the current usable accelerated baseline.

## Docker Hub via GitHub Actions

The workflow is in `.github/workflows/dockerhub.yml`. By default it builds and
pushes:

- `linglambda/index-tts:cu128-ds`
- `linglambda/index-tts:cu128-ds-<git-sha>`
- `linglambda/index-tts:cu128-ds-accel`
- `linglambda/index-tts:cu128-ds-accel-<git-sha>`

Configure this GitHub repository secret before running it:

- `DOCKERHUB_TOKEN`

Run it manually from GitHub Actions with:

- `image_repo`: Docker Hub repository, default `linglambda/index-tts`
- `runs_on`: `ubuntu-latest` or a self-hosted runner label
- `build_accel`: `true`

The images are large. `ubuntu-latest` may still run out of disk even after the
workflow frees common preinstalled toolchains. A self-hosted Linux runner with
at least 80GB free disk is the safer path for repeatable builds.

Default image:

```bash
cd /mnt/d/CodeWork/Node/IndexTTS2
bash docker/build_wsl.sh
```

The script:

- unsets proxy variables for the build process;
- syncs source and checkpoints into `$HOME/indextts2-docker` on WSL ext4;
- verifies Docker can pull through `docker.m.daocloud.io`;
- verifies pip is using `https://mirrors.ustc.edu.cn/pypi/simple`;
- builds `indextts2:cu128-ds`;
- runs a quick GPU/import verification.

For RTX 5060 Ti, the image uses CUDA 12.8 and `TORCH_CUDA_ARCH_LIST=8.0;12.0`.
The BigVGAN CUDA loader in this repo also emits `sm_120` flags when CUDA 12+
is present.

vLLM is not enabled in this stable image. I checked the official issue/PR trail
and the referenced `Ksuriuri/index-tts-vllm` implementation: that route needs a
separate vLLM-specific GPT checkpoint layout and currently conflicts with the
strict IndexTTS2 runtime pins. For safety, `INSTALL_VLLM=1` fails fast instead
of pulling an unpinned vLLM release into `indextts2:cu128-ds`.

The upstream repository has merged its own experimental GPT2 acceleration path.
The WebUI exposes it as `--use_accel`, but it requires a compatible
`flash_attn` build. A temporary container test passed with
`flash-attn==2.8.3`, `torch==2.8.0+cu128`, and RTX 5060 Ti `sm_120`.
Build it as a separate experimental image:

```bash
bash docker/build_accel_wsl.sh
```

Deep verification:

```bash
docker run --rm --gpus all \
  -v "$HOME/indextts2-docker/data/checkpoints:/app/checkpoints" \
  indextts2:cu128-ds-accel \
  python3 /opt/indextts/verify_runtime.py --compile-cuda-kernel --load-model --use-accel
```

Run WebUI after the image is built.

On WSL, a detached GPU container can be stopped when the Windows-side `wsl.exe`
session exits. The PowerShell launcher below keeps a foreground `docker run`
attached to a hidden `wsl.exe` process, so the WebUI stays alive:

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\run_webui_wsl.ps1
```

Open http://127.0.0.1:7860/ after the launcher prints that the container is
running. Stop it with:

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\stop_webui_wsl.ps1
```

Run the experimental official acceleration image on another port:

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\run_webui_wsl.ps1 `
  -ImageName indextts2:cu128-ds-accel `
  -Port 7861 `
  -ContainerName indextts2-webui-accel `
  -WebuiArgs '--use_accel'
```

Stop that container with:

```powershell
powershell -ExecutionPolicy Bypass -File .\docker\stop_webui_wsl.ps1 `
  -ContainerName indextts2-webui-accel
```

If you are already inside a long-lived WSL shell, you can also run it directly:

```bash
docker rm -f indextts2-webui >/dev/null 2>&1 || true
docker run --name indextts2-webui --gpus all --init --ipc=host \
  --security-opt seccomp=unconfined \
  -e TORCH_CUDA_ARCH_LIST="8.0;12.0" \
  -p 7860:7860 \
  -v "$HOME/indextts2-docker/data/checkpoints:/app/checkpoints" \
  -v "$HOME/indextts2-docker/src/outputs:/app/outputs" \
  -v "$HOME/indextts2-docker/cache/torch_extensions:/root/.cache/torch_extensions" \
  -v "$HOME/indextts2-docker/cache/bigvgan_s2mel_cuda_build:/app/indextts/s2mel/modules/bigvgan/alias_free_activation/cuda/build" \
  -v "$HOME/indextts2-docker/cache/bigvgan_cuda_build:/app/indextts/BigVGAN/alias_free_activation/cuda/build" \
  indextts2:cu128-ds \
  python3 -u webui.py --host 0.0.0.0 --fp16 --cuda_kernel --deepspeed
```

Streaming API:

```bash
curl --no-buffer \
  -F "prompt_audio=@examples/voice_01.wav;type=audio/wav" \
  -F "text=Streaming API test. Segment one. Segment two." \
  -F "max_text_tokens_per_segment=20" \
  http://127.0.0.1:7860/api/tts_stream
```

The response is `multipart/x-mixed-replace`; each part is one `audio/wav`
segment. Health check:

```bash
curl http://127.0.0.1:7860/api/health
```
