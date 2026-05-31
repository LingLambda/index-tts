import argparse
import importlib
import os
import sys

import torch


def require_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(f"failed to import {name}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--compile-cuda-kernel", action="store_true")
    parser.add_argument("--use-accel", action="store_true")
    parser.add_argument("--model-dir", default="/app/checkpoints")
    args = parser.parse_args()

    print("python:", sys.version)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        assert torch.cuda.get_device_capability(0) == (12, 0), "expected RTX 5060 Ti sm_120"

    for module_name in ("gradio", "deepspeed", "torchaudio", "transformers", "modelscope"):
        module = require_module(module_name)
        print(module_name, getattr(module, "__version__", "unknown"))

    if args.use_accel:
        module = require_module("flash_attn")
        print("flash_attn", getattr(module, "__version__", "unknown"))

    print("PIP_INDEX_URL:", os.environ.get("PIP_INDEX_URL"))
    print("TORCH_CUDA_ARCH_LIST:", os.environ.get("TORCH_CUDA_ARCH_LIST"))

    if args.compile_cuda_kernel:
        from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import activation1d

        print("bigvgan cuda kernel:", activation1d.anti_alias_activation_cuda)

    if args.load_model:
        from indextts.infer_v2 import IndexTTS2

        tts = IndexTTS2(
            cfg_path=os.path.join(args.model_dir, "config.yaml"),
            model_dir=args.model_dir,
            use_fp16=True,
            use_cuda_kernel=args.compile_cuda_kernel,
            use_deepspeed=True,
            use_accel=args.use_accel,
        )
        print("model loaded:", type(tts).__name__)


if __name__ == "__main__":
    main()
