import asyncio
import base64
import gc
import io
import json
import os
import queue
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import openvino as ov
import openvino_genai as ov_genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import (
    CHAT_HISTORY_DIR,
    DEFAULT_CONTEXT,
    DEFAULT_ENABLE_REASONING,
    DEFAULT_MAX_NEW_TOKENS,
    MODEL_ROOT,
)
from ..files import extract_text_from_bytes, format_attachments_context
from ..history import ChatHistorySaver
from ..memory import MemoryMonitor, powershell_gpu_memory
from ..metadata import check_device_compatibility, read_model_metadata
from ..models import find_models, model_size
from ..settings import load_settings, save_settings
from ..ui import human_bytes


# ============================================================
# STATE MANAGEMENT
# ============================================================

class ModelSession:
    """Encapsulates the loaded OpenVINO pipeline and its active parameters."""

    def __init__(self):
        self.pipe: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.generation_config: Optional[Any] = None
        self.metadata: Optional[Any] = None
        self.model_name: str = ""
        self.model_path: Optional[Path] = None
        self.device: str = "GPU"
        self.context_length: int = DEFAULT_CONTEXT
        self.max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
        self.reasoning_enabled: bool = DEFAULT_ENABLE_REASONING
        self.temperature: float = 0.7
        self.top_p: float = 0.95
        self.load_time: float = 0.0
        self.memory_monitor: Optional[MemoryMonitor] = None
        self.history_saver: Optional[ChatHistorySaver] = None
        self.lock = threading.Lock()

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def unload(self) -> None:
        with self.lock:
            if self.memory_monitor:
                self.memory_monitor.stop()
                self.memory_monitor = None
            self.pipe = None
            self.tokenizer = None
            self.generation_config = None
            self.metadata = None
            self.model_name = ""
            self.model_path = None
            gc.collect()


session = ModelSession()


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class LoadModelRequest(BaseModel):
    model_name: str
    device: str = "GPU"
    context_length: Optional[int] = DEFAULT_CONTEXT
    max_new_tokens: Optional[int] = DEFAULT_MAX_NEW_TOKENS
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    enable_reasoning: Optional[bool] = False


class ChatMessage(BaseModel):
    role: str
    content: str


class FileAttachment(BaseModel):
    filename: str
    data: str  # base64 data url or raw text
    size: Optional[int] = None


class ChatStreamRequest(BaseModel):
    messages: List[ChatMessage]
    images: Optional[List[str]] = None
    files: Optional[List[FileAttachment]] = None
    enable_reasoning: Optional[bool] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class SettingsRequest(BaseModel):
    model_root: Optional[str] = None
    history_dir: Optional[str] = None
    selected_model: Optional[str] = None
    selected_device: Optional[str] = None
    context_length: Optional[int] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    enable_reasoning: Optional[bool] = None


# ============================================================
# APP INITIALIZATION
# ============================================================

def create_app() -> FastAPI:
    app = FastAPI(title="OpenVINO GenAI Web UI", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def get_index():
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "OpenVINO GenAI Web UI running"}

    # ========================================================
    # MODEL & DEVICE APIS
    # ========================================================

    @app.get("/api/models")
    async def list_models():
        settings = load_settings()
        root = Path(settings.get("model_root", MODEL_ROOT))
        discovered = find_models(root)

        result = []
        for p in discovered:
            meta = read_model_metadata(p)
            size = model_size(p)
            result.append({
                "name": p.name,
                "path": str(p),
                "size_bytes": size,
                "size_human": human_bytes(size),
                "display_type": meta.display_type,
                "is_vlm": meta.is_vlm,
                "precision": meta.precision.upper(),
                "max_context": meta.max_position_embeddings,
                "supports_reasoning": meta.supports_reasoning,
            })
        return {"models": result, "root": str(root)}

    @app.get("/api/devices")
    async def get_device_options(model_name: Optional[str] = None):
        core = ov.Core()
        try:
            available_raw = list(core.available_devices)
        except Exception:
            available_raw = ["CPU"]

        target_meta = None
        if model_name:
            settings = load_settings()
            root = Path(settings.get("model_root", MODEL_ROOT))
            model_dir = root / model_name
            if model_dir.exists():
                target_meta = read_model_metadata(model_dir)

        devices = []
        for dev in available_raw:
            is_ok = True
            reason = None
            if target_meta:
                is_ok, reason = check_device_compatibility(dev, target_meta, core)

            devices.append({
                "name": dev,
                "compatible": is_ok,
                "reason": reason,
            })

        return {"devices": devices}

    @app.get("/api/model/status")
    async def model_status():
        current_mem = powershell_gpu_memory()
        return {
            "loaded": session.is_loaded(),
            "model_name": session.model_name,
            "device": session.device,
            "is_vlm": session.metadata.is_vlm if session.metadata else False,
            "display_type": session.metadata.display_type if session.metadata else "",
            "precision": session.metadata.precision.upper() if session.metadata else "",
            "context_length": session.context_length,
            "max_new_tokens": session.max_new_tokens,
            "reasoning_enabled": session.reasoning_enabled,
            "supports_reasoning": session.metadata.supports_reasoning if session.metadata else False,
            "temperature": session.temperature,
            "top_p": session.top_p,
            "load_time_seconds": round(session.load_time, 2),
            "gpu_memory_human": human_bytes(current_mem),
        }

    @app.post("/api/model/load")
    async def load_model(req: LoadModelRequest):
        settings = load_settings()
        root = Path(settings.get("model_root", MODEL_ROOT))
        model_path = root / req.model_name

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model directory '{req.model_name}' not found")

        meta = read_model_metadata(model_path)
        is_ok, reason = check_device_compatibility(req.device, meta)
        if not is_ok:
            raise HTTPException(
                status_code=400,
                detail=f"Device '{req.device}' cannot run this model: {reason}",
            )

        def _do_load():
            session.unload()
            with session.lock:
                session.model_name = req.model_name
                session.model_path = model_path
                session.device = req.device
                session.context_length = req.context_length or DEFAULT_CONTEXT
                session.max_new_tokens = req.max_new_tokens or DEFAULT_MAX_NEW_TOKENS
                session.metadata = meta
                session.reasoning_enabled = bool(req.enable_reasoning and meta.supports_reasoning)
                session.temperature = req.temperature if req.temperature is not None else 0.7
                session.top_p = req.top_p if req.top_p is not None else 0.95

                session.history_saver = ChatHistorySaver(
                    model_name=req.model_name,
                    device=req.device,
                    context_length=session.context_length,
                    reasoning_enabled=session.reasoning_enabled,
                )

                t0 = time.perf_counter()
                if meta.is_vlm:
                    session.pipe = ov_genai.VLMPipeline(str(model_path), req.device)
                else:
                    session.pipe = ov_genai.LLMPipeline(str(model_path), req.device)
                session.load_time = time.perf_counter() - t0

                session.tokenizer = session.pipe.get_tokenizer()
                session.generation_config = session.pipe.get_generation_config()
                session.generation_config.max_new_tokens = session.max_new_tokens
                if hasattr(session.generation_config, "temperature"):
                    session.generation_config.temperature = session.temperature
                if hasattr(session.generation_config, "top_p"):
                    session.generation_config.top_p = session.top_p

        try:
            await asyncio.to_thread(_do_load)
        except Exception as e:
            session.unload()
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

        return {
            "status": "loaded",
            "model_name": session.model_name,
            "device": session.device,
            "display_type": meta.display_type,
            "load_time_seconds": round(session.load_time, 2),
        }

    @app.post("/api/model/unload")
    async def unload_model():
        session.unload()
        return {"status": "unloaded"}

    # ========================================================
    # SETTINGS API
    # ========================================================

    @app.get("/api/settings")
    async def get_settings():
        return load_settings()

    @app.post("/api/settings")
    async def update_settings(req: SettingsRequest):
        data = req.model_dump(exclude_unset=True, exclude_none=True)
        updated = save_settings(data)
        return updated

    # ========================================================
    # CHAT HISTORY API
    # ========================================================

    @app.get("/api/history")
    async def list_history():
        settings = load_settings()
        hist_dir = Path(settings.get("history_dir", CHAT_HISTORY_DIR))
        hist_dir.mkdir(parents=True, exist_ok=True)

        files = []
        for file in hist_dir.glob("*.md"):
            try:
                stat = file.stat()
                txt = file.read_text(encoding="utf-8")

                # Extract model from YAML or title
                model_match = re.search(r"model:\s*['\"]?([^'\"\n]+)", txt)
                model_name = model_match.group(1) if model_match else "Unknown"

                # Extract preview of first user message
                preview = ""
                user_match = re.search(r"\*\*User:\*\*\s*\n+([^\n#]+)", txt)
                if user_match:
                    preview = user_match.group(1).strip()[:80]

                files.append({
                    "filename": file.name,
                    "model": model_name,
                    "preview": preview or file.stem,
                    "modified_time": stat.st_mtime,
                    "size_bytes": stat.st_size,
                })
            except Exception:
                pass

        files.sort(key=lambda x: x["modified_time"], reverse=True)
        return {"history": files, "directory": str(hist_dir)}

    @app.get("/api/history/{filename}")
    async def get_history_file(filename: str):
        settings = load_settings()
        hist_dir = Path(settings.get("history_dir", CHAT_HISTORY_DIR))
        target = hist_dir / filename

        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Chat history file not found")

        content = target.read_text(encoding="utf-8")
        return {"filename": filename, "content": content}

    @app.delete("/api/history/{filename}")
    async def delete_history_file(filename: str):
        settings = load_settings()
        hist_dir = Path(settings.get("history_dir", CHAT_HISTORY_DIR))
        target = hist_dir / filename

        if target.exists() and target.is_file():
            try:
                target.unlink()
                return {"status": "deleted", "filename": filename}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        raise HTTPException(status_code=404, detail="File not found")

    # ========================================================
    # CHAT STREAMING API (SSE)
    # ========================================================

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatStreamRequest):
        if not session.is_loaded():
            raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")

        # Process document/file attachments (for all models)
        if req.files and req.messages:
            file_items = []
            for f in req.files:
                b64_str = f.data
                if "," in b64_str:
                    _, b64_str = b64_str.split(",", 1)
                try:
                    raw_bytes = base64.b64decode(b64_str)
                    content_str = extract_text_from_bytes(f.filename, raw_bytes)
                    file_items.append({"filename": f.filename, "content": content_str})
                except Exception as ex:
                    file_items.append({"filename": f.filename, "content": f"[Error reading file: {ex}]"})

            if file_items:
                ctx_block = format_attachments_context(file_items)
                last_idx = len(req.messages) - 1
                for i in range(len(req.messages) - 1, -1, -1):
                    if req.messages[i].role == "user":
                        last_idx = i
                        break
                req.messages[last_idx].content = (
                    f"{ctx_block}\n\n{req.messages[last_idx].content}"
                )

        # Process multimodal images (for VLM models)
        image_tensors: List[ov.Tensor] = []
        if req.images and session.metadata and session.metadata.is_vlm:
            import numpy as np
            from PIL import Image

            for img_str in req.images:
                try:
                    if "," in img_str:
                        _, b64data = img_str.split(",", 1)
                    else:
                        b64data = img_str
                    raw_bytes = base64.b64decode(b64data)
                    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    arr = np.array(pil_img)
                    image_tensors.append(ov.Tensor(arr))
                except Exception as e:
                    print("Failed to decode image input:", e)

        # Build ChatHistory
        chat_hist = ov_genai.ChatHistory()
        for msg in req.messages:
            chat_hist.append({"role": msg.role, "content": msg.content})

        reasoning = (
            req.enable_reasoning
            if req.enable_reasoning is not None
            else session.reasoning_enabled
        )
        reasoning = reasoning and (session.metadata.supports_reasoning if session.metadata else False)

        q: queue.Queue = queue.Queue()

        def streamer_callback(chunk: str) -> bool:
            q.put({"type": "chunk", "text": chunk})
            return False

        def worker():
            try:
                gen_cfg = session.pipe.get_generation_config()
                if req.max_new_tokens:
                    gen_cfg.max_new_tokens = req.max_new_tokens
                if req.temperature is not None and hasattr(gen_cfg, "temperature"):
                    gen_cfg.temperature = req.temperature
                if req.top_p is not None and hasattr(gen_cfg, "top_p"):
                    gen_cfg.top_p = req.top_p

                generate_kwargs = {
                    "generation_config": gen_cfg,
                    "streamer": streamer_callback,
                    "extra_context": {"enable_thinking": reasoning},
                }
                if session.metadata and session.metadata.is_vlm and image_tensors:
                    generate_kwargs["images"] = image_tensors

                res = session.pipe.generate(
                    chat_hist,
                    **generate_kwargs
                )

                # Extract metrics
                metrics_data = {}
                try:
                    m = res.perf_metrics
                    metrics_data = {
                        "input_tokens": m.get_num_input_tokens(),
                        "output_tokens": m.get_num_generated_tokens(),
                        "throughput": round(m.get_throughput().mean, 2),
                        "ttft": round(m.get_ttft().mean, 1),
                    }
                except Exception:
                    pass

                # Save turn to history saver
                if session.history_saver and req.messages:
                    last_user = req.messages[-1].content if req.messages[-1].role == "user" else ""
                    full_resp = res.texts[0] if hasattr(res, "texts") and res.texts else ""
                    session.history_saver.add_turn(
                        user_prompt=last_user,
                        assistant_response=full_resp,
                        metrics=metrics_data,
                        reasoning_used=reasoning,
                    )

                q.put({"type": "done", "metrics": metrics_data})
            except Exception as exc:
                q.put({"type": "error", "error": str(exc)})

        # Start inference in background thread
        threading.Thread(target=worker, daemon=True).start()

        async def event_generator():
            while True:
                try:
                    # Poll queue with timeout so we don't lock the event loop
                    item = await asyncio.to_thread(q.get, timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.02)
                    continue

                event_type = item.get("type")
                if event_type == "chunk":
                    data = json.dumps({"chunk": item.get("text", "")})
                    yield f"data: {data}\n\n"
                elif event_type == "done":
                    data = json.dumps({"done": True, "metrics": item.get("metrics", {})})
                    yield f"data: {data}\n\n"
                    break
                elif event_type == "error":
                    data = json.dumps({"error": item.get("error", "Unknown generation error")})
                    yield f"data: {data}\n\n"
                    break

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ========================================================
    # OPENAI-COMPATIBLE API (For Open WebUI & External Tools)
    # ========================================================

    @app.get("/v1/models")
    @app.get("/v1/models")
    @app.get("/models")
    async def openai_list_models():
        settings = load_settings()
        root = Path(settings.get("model_root", MODEL_ROOT))
        models = find_models(root)

        now_ts = int(time.time())
        model_objects = []
        for m in models:
            # Default entry (uses configured default device)
            model_objects.append({
                "id": m.name,
                "object": "model",
                "created": int(time.time()),
                "created": now_ts,
                "owned_by": "openvino",
                "permission": [],
                "root": m.name,
                "parent": None,
            })
            # Explicit GPU entry
            model_objects.append({
                "id": f"{m.name} (GPU)",
                "object": "model",
                "created": now_ts,
                "owned_by": "openvino",
                "permission": [],
                "root": m.name,
                "parent": None,
            })
            # Explicit CPU entry
            model_objects.append({
                "id": f"{m.name} (CPU)",
                "object": "model",
                "created": now_ts,
                "owned_by": "openvino",
                "permission": [],
                "root": m.name,
                "parent": None,
            })
        return {"object": "list", "data": model_objects}

    @app.get("/v1/models/{model_id:path}")
    async def openai_get_model(model_id: str):
        settings = load_settings()
        root = Path(settings.get("model_root", MODEL_ROOT))
        models = [m.name for m in find_models(root)]
        if model_id not in models:

        clean_name = model_id
        for suffix in (" (GPU)", " (CPU)", ":GPU", ":CPU", "/gpu", "/cpu"):
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)].strip()
                break

        if clean_name not in models:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        return {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openvino",
            "permission": [],
        }

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def openai_chat_completions(request: Request):
        body = await request.json()
        req_model = body.get("model")
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.95)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_NEW_TOKENS

        settings = load_settings()
        root = Path(settings.get("model_root", MODEL_ROOT))
        available_models = [m.name for m in find_models(root)]

        if not req_model and available_models:
            req_model = available_models[0]
        # Extract device target from model name if present
        target_dev = None
        clean_model_name = req_model or ""

        if req_model and req_model in available_models:
            # Auto-load or switch model if needed
            if not session.is_loaded() or session.model_name != req_model:
                dev = settings.get("selected_device", "GPU")
        for suffix, dev_val in [
            (" (GPU)", "GPU"),
            (" (CPU)", "CPU"),
            (":GPU", "GPU"),
            (":CPU", "CPU"),
            ("/gpu", "GPU"),
            ("/cpu", "CPU"),
        ]:
            if clean_model_name.endswith(suffix):
                target_dev = dev_val
                clean_model_name = clean_model_name[:-len(suffix)].strip()
                break

        if not target_dev:
            target_dev = settings.get("selected_device", "GPU")

        if not clean_model_name and available_models:
            clean_model_name = available_models[0]

        if clean_model_name and clean_model_name in available_models:
            # Auto-load or switch model/device if needed
            if not session.is_loaded() or session.model_name != clean_model_name or session.device != target_dev:
                await load_model(LoadModelRequest(
                    model_name=req_model,
                    device=dev,
                    model_name=clean_model_name,
                    device=target_dev,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                ))

        if not session.is_loaded():
            raise HTTPException(status_code=400, detail="No model loaded and could not resolve model.")

        # Process messages and multimodal images
        image_tensors: List[ov.Tensor] = []
        parsed_history = []

        for msg in messages:
            role = msg.get("role", "user")
            raw_content = msg.get("content", "")

            text_parts = []
            if isinstance(raw_content, str):
                text_parts.append(raw_content)
            elif isinstance(raw_content, list):
                for part in raw_content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            img_url = part.get("image_url", {}).get("url", "")
                            if img_url and session.metadata and session.metadata.is_vlm:
                                import numpy as np
                                from PIL import Image
                                try:
                                    if "," in img_url:
                                        _, b64data = img_url.split(",", 1)
                                    else:
                                        b64data = img_url
                                    raw_bytes = base64.b64decode(b64data)
                                    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                                    arr = np.array(pil_img)
                                    image_tensors.append(ov.Tensor(arr))
                                except Exception as e:
                                    print("Failed to decode OpenAI image_url:", e)

            combined_text = "\n".join(text_parts).strip()
            parsed_history.append({"role": role, "content": combined_text})

        chat_hist = ov_genai.ChatHistory()
        for m in parsed_history:
            chat_hist.append({"role": m["role"], "content": m["content"]})

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        if stream:
            q = queue.Queue()

            def streamer_callback(chunk: str) -> bool:
                q.put(chunk)
                return False

            def worker():
                try:
                    gen_cfg = session.pipe.get_generation_config()
                    if max_tokens:
                        gen_cfg.max_new_tokens = max_tokens
                    if hasattr(gen_cfg, "temperature"):
                        gen_cfg.temperature = temperature
                    if hasattr(gen_cfg, "top_p"):
                        gen_cfg.top_p = top_p

                    extra_ctx = {"enable_thinking": session.reasoning_enabled}
                    if session.metadata and session.metadata.is_vlm:
                        session.pipe.generate(chat_hist, images=image_tensors, generation_config=gen_cfg, streamer=streamer_callback)
                    else:
                        session.pipe.generate(chat_hist, generation_config=gen_cfg, streamer=streamer_callback, extra_context=extra_ctx)
                except Exception as ex:
                    q.put(ex)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()

            async def event_generator():
                # Initial role chunk
                init_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": session.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(init_chunk)}\n\n"

                while True:
                    await asyncio.sleep(0.005)
                    try:
                        item = q.get_nowait()
                    except queue.Empty:
                        continue

                    if item is None:
                        # Final stop chunk
                        stop_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": session.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(stop_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    elif isinstance(item, Exception):
                        err_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": session.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": f"\n[Error: {str(item)}]"},
                                "finish_reason": "error"
                            }]
                        }
                        yield f"data: {json.dumps(err_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    else:
                        chunk_obj = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": session.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": item},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk_obj)}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming call
            collected_text = []
            def collect_streamer(chunk: str) -> bool:
                collected_text.append(chunk)
                return False

            def non_stream_worker():
                gen_cfg = session.pipe.get_generation_config()
                if max_tokens:
                    gen_cfg.max_new_tokens = max_tokens
                if hasattr(gen_cfg, "temperature"):
                    gen_cfg.temperature = temperature
                if hasattr(gen_cfg, "top_p"):
                    gen_cfg.top_p = top_p
                extra_ctx = {"enable_thinking": session.reasoning_enabled}
                if session.metadata and session.metadata.is_vlm:
                    session.pipe.generate(chat_hist, images=image_tensors, generation_config=gen_cfg, streamer=collect_streamer)
                else:
                    session.pipe.generate(chat_hist, generation_config=gen_cfg, streamer=collect_streamer, extra_context=extra_ctx)

            await asyncio.to_thread(non_stream_worker)
            full_reply = "".join(collected_text)

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": session.model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_reply
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(str(parsed_history)) // 4,
                    "completion_tokens": len(full_reply) // 4,
                    "total_tokens": (len(str(parsed_history)) + len(full_reply)) // 4
                }
            }

    return app


app = create_app()
