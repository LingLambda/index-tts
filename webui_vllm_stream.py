import html
import io
import json
import os
import sys
import threading
import tempfile
import time
import traceback
import wave
from typing import Optional

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

for proxy_key in ("NO_PROXY", "no_proxy"):
    proxy_value = os.environ.get(proxy_key)
    if proxy_value:
        os.environ[proxy_key] = proxy_value.replace("“", "").replace("”", "")

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--is_fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.08, help="vLLM GPT GPU memory utilization")
parser.add_argument("--qwenemo_gpu_memory_utilization", type=float, default=0.15, help="vLLM Qwen emotion GPU memory utilization")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from indextts import infer_vllm_v2 as infer_vllm
from tools.i18n.i18n import I18nAuto

if os.environ.get("INDEXTTS2_DISABLE_QWEN_EMO", "0") == "1":
    class DisabledQwenEmotion:
        def __init__(self, *args, **kwargs):
            print("[webui] Qwen emotion text control is disabled by INDEXTTS2_DISABLE_QWEN_EMO=1", flush=True)

        async def inference(self, *args, **kwargs):
            raise RuntimeError("Qwen emotion text control is disabled in this WebUI profile.")

    infer_vllm.QwenEmotion = DisabledQwenEmotion

IndexTTS2 = infer_vllm.IndexTTS2

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = None

FRONTEND_DEBUG_HEAD = """
<script>
(function () {
  if (window.__indextts2FrontendDebugInstalled) return;
  window.__indextts2FrontendDebugInstalled = true;
  const sessionId = Math.random().toString(36).slice(2);
  function toText(value) {
    if (value === undefined) return "undefined";
    if (value === null) return "null";
    if (value instanceof Error) return value.stack || value.message || String(value);
    if (typeof value === "object") {
      try { return JSON.stringify(value); } catch (_) { return String(value); }
    }
    return String(value);
  }
  function send(kind, data) {
    const payload = {
      kind,
      data: data || {},
      href: location.href,
      userAgent: navigator.userAgent,
      sessionId,
      ts: new Date().toISOString()
    };
    try {
      fetch("/api/frontend_log", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
        keepalive: true
      }).catch(function () {});
    } catch (_) {}
  }
  window.addEventListener("error", function (event) {
    send("window.error", {
      message: event.message,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      error: toText(event.error)
    });
  });
  window.addEventListener("unhandledrejection", function (event) {
    send("window.unhandledrejection", {reason: toText(event.reason)});
  });
  document.addEventListener("click", function (event) {
    const target = event.target && event.target.closest ? event.target.closest("button, summary, [role='button'], label") : event.target;
    const text = target && target.innerText ? target.innerText.trim() : "";
    if (text.includes("高级生成参数设置") || text.includes("预览分句结果") || text.includes("Advanced") || text.includes("Segments")) {
      send("ui.click", {text: text.slice(0, 200)});
    }
  }, true);
  send("debug.installed", {});
})();
</script>
"""


def init_tts():
    global tts
    if tts is None:
        tts = IndexTTS2(
            model_dir=cmd_args.model_dir,
            is_fp16=cmd_args.is_fp16,
            use_cuda_kernel=False,
            gpu_memory_utilization=cmd_args.gpu_memory_utilization,
            qwenemo_gpu_memory_utilization=cmd_args.qwenemo_gpu_memory_utilization,
        )
        normalizer = getattr(tts, "normalizer", None)
        if normalizer is not None:
            if not hasattr(normalizer, "term_glossary"):
                normalizer.term_glossary = {}
            if not hasattr(normalizer, "enable_glossary"):
                normalizer.enable_glossary = False
        if not hasattr(tts, "glossary_path"):
            tts.glossary_path = os.path.join("prompts", "term_glossary.yaml")
    return tts
# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # skip experimental features

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
example_cases = []
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None

        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES_ALL[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0),
                             ])

def get_example_cases(include_experimental = False):
    if include_experimental:
        return example_cases  # show every example

    # exclude emotion control mode 3 (emotion from text description)
    return [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]

def format_glossary_markdown():
    """将词汇表转换为Markdown表格格式"""
    if not tts.normalizer.term_glossary:
        return i18n("暂无术语")

    lines = [f"| {i18n('术语')} | {i18n('中文读法')} | {i18n('英文读法')} |"]
    lines.append("|---|---|---|")

    for term, reading in tts.normalizer.term_glossary.items():
        zh = reading.get("zh", "") if isinstance(reading, dict) else reading
        en = reading.get("en", "") if isinstance(reading, dict) else reading
        lines.append(f"| {term} | {zh} | {en} |")

    return "\n".join(lines)

def wav_tensor_to_bytes(wav, sampling_rate=22050):
    wav = wav.detach().cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.clamp(-32767.0, 32767.0).type(torch.int16)
    wav_data = wav.transpose(0, 1).contiguous().numpy()
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(wav_data.shape[1])
            wav_file.setsampwidth(2)
            wav_file.setframerate(sampling_rate)
            wav_file.writeframes(wav_data.tobytes())
        return buffer.getvalue()

def normalize_emo_vector(vec):
    if hasattr(tts, "normalize_emo_vec"):
        return tts.normalize_emo_vec(vec, apply_bias=True)
    return vec

def split_tokenized_text(tokenized, max_text_tokens_per_segment=120, quick_streaming_tokens=0):
    tokenizer = tts.tokenizer
    max_tokens = max(1, int(max_text_tokens_per_segment))
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

def detokenize_segment(tokens):
    sp_model = getattr(getattr(tts, "tokenizer", None), "sp_model", None)
    if sp_model is not None and hasattr(sp_model, "DecodePieces"):
        return sp_model.DecodePieces(tokens).strip()
    return "".join(tokens).replace("\u2581", " ").strip()


def format_segments_preview(text, max_text_tokens_per_segment):
    if not text:
        return i18n("请输入文本后预览分句。")

    max_tokens = max(1, int(float(max_text_tokens_per_segment or 120)))
    text_tokens_list = tts.tokenizer.tokenize(text)
    segments = split_tokenized_text(text_tokens_list, max_tokens)
    if not segments:
        return i18n("暂无分句结果。")

    lines = []
    for index, segment in enumerate(segments, start=1):
        segment_text = detokenize_segment(segment)
        lines.append(f"{index}. [{len(segment)} tokens] {segment_text}")
    return "\n".join(lines)


def split_text_for_vllm_stream(text, max_text_tokens_per_segment, quick_streaming_tokens):
    tokenized = tts.tokenizer.tokenize(text)
    segments = split_tokenized_text(tokenized, max_text_tokens_per_segment, quick_streaming_tokens)
    chunks = [detokenize_segment(segment) for segment in segments]
    chunks = [chunk for chunk in chunks if chunk]
    return chunks or [text]

async def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
               stream_output=False, quick_streaming_tokens=0,
                *args, progress=gr.Progress()):
    try:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
        # set gradio progress
        tts.gr_progress = progress
        do_sample, top_p, top_k, temperature, \
            length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
            # "typical_sampling": bool(typical_sampling),
            # "typical_mass": float(typical_mass),
        }
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:  # emotion from speaker
            emo_ref_path = None  # remove external reference audio
        if emo_control_method == 1:  # emotion from reference audio
            pass
        if emo_control_method == 2:  # emotion from custom vectors
            vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            vec = normalize_emo_vector(vec)
        else:
            # don't use the emotion vector inputs for the other modes
            vec = None

        if emo_text == "":
            # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
            emo_text = None

        print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}", flush=True)
        infer_kwargs = dict(
            spk_audio_prompt=prompt, text=text,
            output_path=output_path,
            emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
            emo_vector=vec,
            use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
            verbose=cmd_args.verbose,
            max_text_tokens_per_sentence=int(max_text_tokens_per_segment),
            **kwargs
        )
        if stream_output:
            print("[webui] stream generation started", flush=True)
            chunks = split_text_for_vllm_stream(text, max_text_tokens_per_segment, quick_streaming_tokens)
            for idx, chunk in enumerate(chunks, start=1):
                chunk_output_path = os.path.join("outputs", f"spk_{int(time.time())}_{idx}.wav")
                chunk_kwargs = dict(infer_kwargs)
                chunk_kwargs["text"] = chunk
                chunk_kwargs["output_path"] = chunk_output_path
                chunk_kwargs["max_text_tokens_per_sentence"] = int(max_text_tokens_per_segment)
                output = await tts.infer(**chunk_kwargs)
                with open(output, "rb") as f:
                    yield f.read()
            print("[webui] stream generation finished", flush=True)
            return

        output = await tts.infer(**infer_kwargs)
        yield gr.update(value=output, visible=True)
    except Exception:
        print("[webui] generation failed", flush=True)
        traceback.print_exc()
        raise

def parse_emo_control_method(value):
    if isinstance(value, int):
        return value
    if hasattr(value, "value"):
        return int(value.value)
    value = str(value)
    if value.isdigit():
        return int(value)
    for idx, choice in enumerate(EMO_CHOICES_ALL):
        if value == choice:
            return idx
    raise ValueError(f"Unknown emotion control method: {value}")

async def save_upload_file(upload: UploadFile) -> str:
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

def create_api_app() -> FastAPI:
    app = FastAPI(title="IndexTTS2 API")

    @app.get("/api/health")
    async def api_health():
        return {"status": "ok", "model": "IndexTTS2"}

    @app.post("/api/frontend_log")
    async def api_frontend_log(request: Request):
        try:
            payload = await request.json()
        except Exception as exc:
            payload = {"parse_error": repr(exc)}
        print(f"[frontend] {json.dumps(payload, ensure_ascii=False)}", flush=True)
        return {"ok": True}

    @app.post("/api/tts_stream")
    async def api_tts_stream(
        prompt_audio: UploadFile = File(...),
        text: str = Form(...),
        emo_ref_audio: Optional[UploadFile] = File(None),
        emo_control_method: str = Form("0"),
        emo_weight: float = Form(0.65),
        emo_random: bool = Form(False),
        emo_text: str = Form(""),
        vec1: float = Form(0.0),
        vec2: float = Form(0.0),
        vec3: float = Form(0.0),
        vec4: float = Form(0.0),
        vec5: float = Form(0.0),
        vec6: float = Form(0.0),
        vec7: float = Form(0.0),
        vec8: float = Form(0.0),
        max_text_tokens_per_segment: int = Form(120),
        quick_streaming_tokens: int = Form(0),
        do_sample: bool = Form(True),
        top_p: float = Form(0.8),
        top_k: int = Form(30),
        temperature: float = Form(0.8),
        length_penalty: float = Form(0.0),
        num_beams: int = Form(3),
        repetition_penalty: float = Form(10.0),
        max_mel_tokens: int = Form(1500),
    ):
        try:
            method = parse_emo_control_method(emo_control_method)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        prompt_path = await save_upload_file(prompt_audio)
        emo_ref_path = await save_upload_file(emo_ref_audio) if emo_ref_audio is not None else None
        cleanup_paths = [prompt_path]
        if emo_ref_path:
            cleanup_paths.append(emo_ref_path)

        if method == 0:
            emo_ref_path = None
            emo_vec = None
        elif method == 1:
            if emo_ref_path is None:
                for path in cleanup_paths:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                raise HTTPException(status_code=400, detail="emo_ref_audio is required when emo_control_method=1")
            emo_vec = None
        elif method == 2:
            emo_vec = normalize_emo_vector([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8])
        else:
            emo_vec = None

        if emo_text == "":
            emo_text = None

        infer_kwargs = dict(
            spk_audio_prompt=prompt_path,
            text=text,
            output_path=os.path.join("outputs", f"api_{int(time.time())}.wav"),
            emo_audio_prompt=emo_ref_path,
            emo_alpha=emo_weight,
            emo_vector=emo_vec,
            use_emo_text=(method == 3),
            emo_text=emo_text,
            use_random=emo_random,
            verbose=cmd_args.verbose,
            max_text_tokens_per_sentence=int(max_text_tokens_per_segment),
            do_sample=bool(do_sample),
            top_p=float(top_p),
            top_k=int(top_k) if int(top_k) > 0 else None,
            temperature=float(temperature),
            length_penalty=float(length_penalty),
            num_beams=int(num_beams),
            repetition_penalty=float(repetition_penalty),
            max_mel_tokens=int(max_mel_tokens),
        )

        boundary = "indextts2-wav"

        async def iter_wav_parts():
            try:
                chunks = split_text_for_vllm_stream(text, max_text_tokens_per_segment, quick_streaming_tokens)
                for segment_index, chunk in enumerate(chunks, start=1):
                    chunk_output_path = os.path.join("outputs", f"api_{int(time.time())}_{segment_index}.wav")
                    chunk_kwargs = dict(infer_kwargs)
                    chunk_kwargs["text"] = chunk
                    chunk_kwargs["output_path"] = chunk_output_path
                    chunk_kwargs["max_text_tokens_per_sentence"] = int(max_text_tokens_per_segment)
                    output = await tts.infer(**chunk_kwargs)
                    with open(output, "rb") as f:
                        data = f.read()
                    header = (
                        f"--{boundary}\r\n"
                        "Content-Type: audio/wav\r\n"
                        f"Content-Disposition: inline; filename=\"segment_{segment_index}.wav\"\r\n"
                        f"Content-Length: {len(data)}\r\n\r\n"
                    )
                    yield header.encode("ascii")
                    yield data
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

    return app

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

def reset_output_audio():
    return gr.update(value=None, visible=True)

def create_demo():
    print("[webui] building Gradio demo", flush=True)
    with gr.Blocks(title="IndexTTS Demo") as demo:
        mutex = threading.Lock()
        gr.HTML(value="", visible=False, head=FRONTEND_DEBUG_HEAD)
        gr.HTML('''
        <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
    <p align="center">
    <a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
    </p>
        ''')

        with gr.Tab(i18n("音频生成")):
            with gr.Row():
                os.makedirs("prompts",exist_ok=True)
                prompt_audio = gr.Audio(label=i18n("音色参考音频"),key="prompt_audio",
                                        sources=["upload","microphone"],type="filepath")
                prompt_list = os.listdir("prompts")
                default = ''
                if prompt_list:
                    default = prompt_list[0]
                with gr.Column():
                    input_text_single = gr.TextArea(label=i18n("文本"),key="input_text_single", placeholder=i18n("请输入目标文本"), info=f"{i18n('当前模型版本')}{getattr(tts, 'model_version', '2.0') or '2.0'}")
                    gen_button = gr.Button(i18n("生成语音"), key="gen_button",interactive=True)
                output_audio = gr.Audio(label=i18n("生成结果"), visible=True,key="output_audio", streaming=True, autoplay=True, format="wav")

            with gr.Row():
                experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)
                glossary_checkbox = gr.Checkbox(label=i18n("开启术语词汇读音"), value=tts.normalizer.enable_glossary)
            with gr.Accordion(i18n("功能设置")):
                # 情感控制选项部分
                with gr.Row():
                    emo_control_method = gr.Radio(
                        choices=EMO_CHOICES_OFFICIAL,
                        type="index",
                        value=EMO_CHOICES_OFFICIAL[0],label=i18n("情感控制方式"))
                    # we MUST have an extra, INVISIBLE list of *all* emotion control
                    # methods so that gr.Dataset() can fetch ALL control mode labels!
                    # otherwise, the gr.Dataset()'s experimental labels would be empty!
                    emo_control_method_all = gr.Radio(
                        choices=EMO_CHOICES_ALL,
                        type="index",
                        value=EMO_CHOICES_ALL[0], label=i18n("情感控制方式"),
                        visible=False)  # do not render
            # 情感参考音频部分
            with gr.Group(visible=False) as emotion_reference_group:
                with gr.Row():
                    emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

            # 情感随机采样
            with gr.Row(visible=False) as emotion_randomize_group:
                emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

            # 情感向量控制部分
            with gr.Group(visible=False) as emotion_vector_group:
                with gr.Row():
                    with gr.Column():
                        vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    with gr.Column():
                        vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

            with gr.Group(visible=False) as emo_text_group:
                create_experimental_warning_message()
                with gr.Row():
                    emo_text = gr.Textbox(label=i18n("情感描述文本"),
                                          placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                                          value="",
                                          info=i18n("例如：委屈巴巴、危险在悄悄逼近"))

            with gr.Row(visible=False) as emo_weight_group:
                emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)

            # 术语词汇表管理
            with gr.Accordion(i18n("自定义术语词汇读音"), open=False, visible=tts.normalizer.enable_glossary) as glossary_accordion:
                gr.Markdown(i18n("自定义个别专业术语的读音"))
                with gr.Row():
                    with gr.Column(scale=1):
                        glossary_term = gr.Textbox(
                            label=i18n("术语"),
                            placeholder="IndexTTS2",
                        )
                        glossary_reading_zh = gr.Textbox(
                            label=i18n("中文读法"),
                            placeholder="Index T-T-S 二",
                        )
                        glossary_reading_en = gr.Textbox(
                            label=i18n("英文读法"),
                            placeholder="Index T-T-S two",
                        )
                        btn_add_term = gr.Button(i18n("添加术语"), scale=1)
                    with gr.Column(scale=2):
                        glossary_table = gr.Markdown(
                            value=format_glossary_markdown()
                        )

            with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as advanced_settings_group:
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                        with gr.Row():
                            do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                            temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                        with gr.Row():
                            top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                            top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                            num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                        with gr.Row():
                            repetition_penalty = gr.Slider(label="repetition_penalty", value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                            length_penalty = gr.Slider(label="length_penalty", value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                        max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                        # with gr.Row():
                        #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                        #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                    with gr.Column(scale=2):
                        gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                        with gr.Row():
                            initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                            max_text_tokens_per_segment = gr.Slider(
                                label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                                info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                            )
                        with gr.Group():
                            with gr.Row():
                                stream_output = gr.Checkbox(
                                    label=i18n("流式输出"), value=False,
                                    info=i18n("边生成边自动拼接音频预览")
                                )
                                quick_streaming_tokens = gr.Slider(
                                    label=i18n("首段快速Token数"), value=80, minimum=0, maximum=tts.cfg.gpt.max_text_tokens, step=2,
                                    info=i18n("开启流式输出时生效，建议80以降低首段等待时间")
                                )
                            segments_preview = gr.Textbox(
                                label=i18n("预览分句结果"),
                                value=i18n("请输入文本后预览分句。"),
                                key="segments_preview",
                                lines=8,
                                max_lines=12,
                                interactive=False,
                                autoscroll=False,
                            )
                advanced_params = [
                    do_sample, top_p, top_k, temperature,
                    length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                    # typical_sampling, typical_mass,
                ]

            # we must use `gr.Dataset` to support dynamic UI rewrites, since `gr.Examples`
            # binds tightly to UI and always restores the initial state of all components,
            # such as the list of available choices in emo_control_method.
            example_table = gr.Dataset(label="Examples",
                samples_per_page=20,
                samples=get_example_cases(include_experimental=False),
                type="values",
                # these components are NOT "connected". it just reads the column labels/available
                # states from them, so we MUST link to the "all options" versions of all components,
                # such as `emo_control_method_all` (to be able to see EXPERIMENTAL text labels)!
                components=[prompt_audio,
                            emo_control_method_all,  # important: support all mode labels!
                            input_text_single,
                            emo_upload,
                            emo_weight,
                            emo_text,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            )

        def on_example_click(example):
            print(f"Example clicked: ({len(example)} values) = {example!r}")
            return (
                gr.update(value=example[0]),
                gr.update(value=example[1]),
                gr.update(value=example[2]),
                gr.update(value=example[3]),
                gr.update(value=example[4]),
                gr.update(value=example[5]),
                gr.update(value=example[6]),
                gr.update(value=example[7]),
                gr.update(value=example[8]),
                gr.update(value=example[9]),
                gr.update(value=example[10]),
                gr.update(value=example[11]),
                gr.update(value=example[12]),
                gr.update(value=example[13]),
            )

        # click() event works on both desktop and mobile UI
        example_table.click(on_example_click,
                            inputs=[example_table],
                            outputs=[prompt_audio,
                                     emo_control_method,
                                     input_text_single,
                                     emo_upload,
                                     emo_weight,
                                     emo_text,
                                     vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        )

        def on_input_text_change(text, max_text_tokens_per_segment):
            try:
                preview = format_segments_preview(text, max_text_tokens_per_segment)
                return gr.update(value=preview)
            except Exception as exc:
                print("[webui] failed to update segments preview", flush=True)
                traceback.print_exc()
                return gr.update(value=f"分句预览失败: {exc!r}")

        # 术语词汇表事件处理函数
        def on_add_glossary_term(term, reading_zh, reading_en):
            """添加术语到词汇表并自动保存"""
            term = term.rstrip()
            reading_zh = reading_zh.rstrip()
            reading_en = reading_en.rstrip()

            if not term:
                gr.Warning(i18n("请输入术语"))
                return gr.update()

            if not reading_zh and not reading_en:
                gr.Warning(i18n("请至少输入一种读法"))
                return gr.update()


            # 构建读法数据
            if reading_zh and reading_en:
                reading = {"zh": reading_zh, "en": reading_en}
            elif reading_zh:
                reading = {"zh": reading_zh}
            elif reading_en:
                reading = {"en": reading_en}
            else:
                reading = reading_zh or reading_en

            # 添加到词汇表
            tts.normalizer.term_glossary[term] = reading

            save_glossary = getattr(tts.normalizer, "save_glossary_to_yaml", None)
            if callable(save_glossary):
                try:
                    save_glossary(tts.glossary_path)
                except Exception as e:
                    gr.Error(i18n("保存词汇表时出错"))
                    print(f"Error details: {e}")
                    return gr.update()
            gr.Info(i18n("词汇表已更新"), duration=1)

            # 更新Markdown表格
            return gr.update(value=format_glossary_markdown())


        def on_method_change(emo_control_method):
            if emo_control_method == 1:  # emotion reference audio
                return (gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=True)
                        )
            elif emo_control_method == 2:  # emotion vectors
                return (gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=True)
                        )
            elif emo_control_method == 3:  # emotion text description
                return (gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=True)
                        )
            else:  # 0: same as speaker voice
                return (gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False)
                        )

        emo_control_method.change(on_method_change,
            inputs=[emo_control_method],
            outputs=[emotion_reference_group,
                     emotion_randomize_group,
                     emotion_vector_group,
                     emo_text_group,
                     emo_weight_group]
        )

        def on_experimental_change(is_experimental, current_mode_index):
            # 切换情感控制选项
            new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
            # if their current mode selection doesn't exist in new choices, reset to 0.
            # we don't verify that OLD index means the same in NEW list, since we KNOW it does.
            new_index = current_mode_index if current_mode_index < len(new_choices) else 0

            return (
                gr.update(choices=new_choices, value=new_choices[new_index]),
                gr.update(samples=get_example_cases(include_experimental=is_experimental)),
            )

        experimental_checkbox.change(
            on_experimental_change,
            inputs=[experimental_checkbox, emo_control_method],
            outputs=[emo_control_method, example_table]
        )

        def on_glossary_checkbox_change(is_enabled):
            """控制术语词汇表的可见性"""
            tts.normalizer.enable_glossary = is_enabled
            return gr.update(visible=is_enabled)

        glossary_checkbox.change(
            on_glossary_checkbox_change,
            inputs=[glossary_checkbox],
            outputs=[glossary_accordion]
        )

        input_text_single.change(
            on_input_text_change,
            inputs=[input_text_single, max_text_tokens_per_segment],
            outputs=[segments_preview]
        )

        max_text_tokens_per_segment.change(
            on_input_text_change,
            inputs=[input_text_single, max_text_tokens_per_segment],
            outputs=[segments_preview]
        )

        prompt_audio.upload(update_prompt_audio,
                             inputs=[],
                             outputs=[gen_button])

        def on_demo_load():
            """页面加载时重新加载glossary数据"""
            load_glossary = getattr(tts.normalizer, "load_glossary_from_yaml", None)
            if callable(load_glossary):
                try:
                    load_glossary(tts.glossary_path)
                except Exception as e:
                    gr.Error(i18n("加载词汇表时出错"))
                    print(f"Failed to reload glossary on page load: {e}")
            return gr.update(value=format_glossary_markdown())

        # 术语词汇表事件绑定
        btn_add_term.click(
            on_add_glossary_term,
            inputs=[glossary_term, glossary_reading_zh, glossary_reading_en],
            outputs=[glossary_table]
        )

        # 页面加载时重新加载glossary
        demo.load(
            on_demo_load,
            inputs=[],
            outputs=[glossary_table]
        )

        gen_click = gen_button.click(
            reset_output_audio,
            inputs=[],
            outputs=[output_audio],
            queue=False,
        )
        gen_click.then(gen_single,
                       inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                              vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                               emo_text,emo_random,
                               max_text_tokens_per_segment,
                               stream_output, quick_streaming_tokens,
                               *advanced_params,
                       ],
                       outputs=[output_audio])



    return demo


def main():
    init_tts()
    demo = create_demo()
    demo.queue(20)
    allowed_paths = [
        os.path.abspath("outputs"),
        os.path.abspath("examples"),
        tempfile.gettempdir(),
    ]
    app = create_api_app()
    app = gr.mount_gradio_app(
        app,
        demo,
        path="/",
        server_name=cmd_args.host,
        server_port=cmd_args.port,
        allowed_paths=allowed_paths,
        show_error=True,
    )
    import uvicorn
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port)


if __name__ == "__main__":
    main()
