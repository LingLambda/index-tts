import argparse
import io
import os
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from indextts import infer_vllm_v2 as infer_vllm

if os.environ.get("INDEXTTS2_DISABLE_QWEN_EMO", "0") == "1":
    class DisabledQwenEmotion:
        def __init__(self, *args, **kwargs):
            print("[api] Qwen emotion text control is disabled by INDEXTTS2_DISABLE_QWEN_EMO=1", flush=True)

        async def inference(self, *args, **kwargs):
            raise RuntimeError("Qwen emotion text control is disabled in this API profile.")

    infer_vllm.QwenEmotion = DisabledQwenEmotion

IndexTTS2 = infer_vllm.IndexTTS2

tts = None


def split_tokenized_text(tokenized, max_text_tokens_per_sentence=120, quick_streaming_tokens=0):
    tokenizer = tts.tokenizer
    max_tokens = max(1, int(max_text_tokens_per_sentence))
    quick_tokens = max(0, int(quick_streaming_tokens or 0))
    split_limit = min(max_tokens, quick_tokens) if quick_tokens else max_tokens

    if hasattr(tokenizer, "split_segments"):
        try:
            return tokenizer.split_segments(
                tokenized,
                max_text_tokens_per_segment=max_tokens,
                quick_streaming_tokens=quick_tokens,
            )
        except TypeError:
            return tokenizer.split_segments(tokenized, max_text_tokens_per_segment=split_limit)

    if hasattr(tokenizer, "split_sentences"):
        return tokenizer.split_sentences(tokenized, max_tokens_per_sentence=split_limit)

    return [tokenized[i:i + split_limit] for i in range(0, len(tokenized), split_limit)]


def detokenize_segment(segment):
    sp_model = getattr(tts.tokenizer, "sp_model", None)
    if sp_model is not None and hasattr(sp_model, "DecodePieces"):
        return sp_model.DecodePieces(segment).strip()
    return "".join(segment).replace("\u2581", " ").strip()


def split_text_for_stream(text, max_text_tokens_per_sentence=120, quick_streaming_tokens=0):
    tokenized = tts.tokenizer.tokenize(text)
    segments = split_tokenized_text(tokenized, max_text_tokens_per_sentence, quick_streaming_tokens)
    chunks = [detokenize_segment(segment) for segment in segments]
    chunks = [chunk for chunk in chunks if chunk]
    return chunks or [text]


def wav_response(sr, wav):
    with io.BytesIO() as wav_buffer:
        sf.write(wav_buffer, wav, sr, format="WAV", subtype="PCM_16")
        return wav_buffer.getvalue()


async def save_upload_file(upload: UploadFile) -> str:
    os.makedirs("outputs/tasks", exist_ok=True)
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(prefix="indextts2_", suffix=suffix, dir="outputs/tasks")
    try:
        with os.fdopen(fd, "wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    return path


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        use_cuda_kernel=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        qwenemo_gpu_memory_utilization=args.qwenemo_gpu_memory_utilization,
    )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    if tts is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "message": "TTS model not initialized"})
    return {"status": "healthy", "message": "Service is running", "timestamp": time.time()}


def build_infer_kwargs(data, output_path=None):
    emo_control_method = data.get("emo_control_method", 0)
    if type(emo_control_method) is not int:
        emo_control_method = int(emo_control_method)

    emo_ref_path = data.get("emo_ref_path")
    emo_weight = data.get("emo_weight", 1.0)
    emo_vec = data.get("emo_vec", [0] * 8)
    if emo_control_method == 0:
        emo_ref_path = None
        emo_weight = 1.0
        vec = None
    elif emo_control_method == 2:
        vec = emo_vec
    else:
        vec = None

    return {
        "spk_audio_prompt": data["spk_audio_path"],
        "text": data["text"],
        "output_path": output_path,
        "emo_audio_prompt": emo_ref_path,
        "emo_alpha": emo_weight,
        "emo_vector": vec,
        "use_emo_text": emo_control_method == 3,
        "emo_text": data.get("emo_text"),
        "use_random": data.get("emo_random", False),
        "max_text_tokens_per_sentence": int(data.get("max_text_tokens_per_sentence", 120)),
    }


@app.post("/tts_url")
async def tts_url(request: Request):
    try:
        data = await request.json()
        sr, wav = await tts.infer(**build_infer_kwargs(data, output_path=None))
        return Response(content=wav_response(sr, wav), media_type="audio/wav")
    except Exception as ex:
        tb_str = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(status_code=500, content={"status": "error", "error": tb_str})


@app.post("/tts_stream")
async def tts_stream(request: Request):
    try:
        data = await request.json()
        quick_streaming_tokens = int(data.get("quick_streaming_tokens", 0))
        max_tokens = int(data.get("max_text_tokens_per_sentence", 120))
        boundary = "indextts2-wav"

        async def iter_wav_parts():
            chunks = split_text_for_stream(data["text"], max_tokens, quick_streaming_tokens)
            for index, chunk in enumerate(chunks, start=1):
                chunk_data = dict(data)
                chunk_data["text"] = chunk
                sr, wav = await tts.infer(**build_infer_kwargs(chunk_data, output_path=None))
                wav_bytes = wav_response(sr, wav)
                header = (
                    f"--{boundary}\r\n"
                    "Content-Type: audio/wav\r\n"
                    f"Content-Disposition: inline; filename=\"segment_{index}.wav\"\r\n"
                    f"Content-Length: {len(wav_bytes)}\r\n\r\n"
                )
                yield header.encode("ascii")
                yield wav_bytes
                yield b"\r\n"
            yield f"--{boundary}--\r\n".encode("ascii")

        return StreamingResponse(
            iter_wav_parts(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
            headers={"X-Accel-Buffering": "no"},
        )
    except Exception as ex:
        tb_str = "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(status_code=500, content={"status": "error", "error": tb_str})


@app.post("/api/tts_stream")
async def api_tts_stream(
    text: str = Form(...),
    prompt_audio: UploadFile = File(...),
    emo_control_method: int = Form(0),
    emo_ref_audio: Optional[UploadFile] = File(None),
    emo_weight: float = Form(1.0),
    max_text_tokens_per_sentence: int = Form(120),
    quick_streaming_tokens: int = Form(0),
):
    prompt_path = await save_upload_file(prompt_audio)
    emo_ref_path = await save_upload_file(emo_ref_audio) if emo_ref_audio is not None else None
    cleanup_paths = [p for p in [prompt_path, emo_ref_path] if p]
    boundary = "indextts2-wav"

    async def iter_wav_parts():
        try:
            chunks = split_text_for_stream(text, max_text_tokens_per_sentence, quick_streaming_tokens)
            for index, chunk in enumerate(chunks, start=1):
                data = {
                    "text": chunk,
                    "spk_audio_path": prompt_path,
                    "emo_control_method": emo_control_method,
                    "emo_ref_path": emo_ref_path,
                    "emo_weight": emo_weight,
                    "max_text_tokens_per_sentence": max_text_tokens_per_sentence,
                }
                sr, wav = await tts.infer(**build_infer_kwargs(data, output_path=None))
                wav_bytes = wav_response(sr, wav)
                header = (
                    f"--{boundary}\r\n"
                    "Content-Type: audio/wav\r\n"
                    f"Content-Disposition: inline; filename=\"segment_{index}.wav\"\r\n"
                    f"Content-Length: {len(wav_bytes)}\r\n\r\n"
                )
                yield header.encode("ascii")
                yield wav_bytes
                yield b"\r\n"
            yield f"--{boundary}--\r\n".encode("ascii")
        finally:
            for path in cleanup_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

    return StreamingResponse(
        iter_wav_parts(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--model_dir", type=str, default="checkpoints/IndexTTS-2-vLLM")
    parser.add_argument("--is_fp16", action="store_true", default=False)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.08)
    parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.15)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    os.makedirs("outputs", exist_ok=True)
    logger.add("logs/api_vllm_stream.log", rotation="10 MB", retention=10, level="DEBUG", enqueue=True)

    import uvicorn

    uvicorn.run(app=app, host=args.host, port=args.port)
