"""
OpenVINO GenAI - OpenAI-Compatible API Server.
High-performance backend for Open WebUI and standard OpenAI clients.
Supports streaming chat completions, multimodal vision inputs, and dynamic GPU/CPU model loading.
"""

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
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..config import (
    DEFAULT_CONTEXT,
    DEFAULT_ENABLE_REASONING,
    DEFAULT_MAX_NEW_TOKENS,
    MODEL_ROOT,
)
from ..memory import powershell_gpu_memory
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
        self.lock = threading.Lock()

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def unload(self) -> None:
        with self.lock:
            self.pipe = None
            self.tokenizer = None
            self.generation_config = None
            self.metadata = None
            self.model_name = ""
            self.model_path = None
            gc.collect()


session = ModelSession()


class LoadModelRequest(BaseModel):
    model_name: str
    device: str = "GPU"
    context_length: Optional[int] = DEFAULT_CONTEXT
    max_new_tokens: Optional[int] = DEFAULT_MAX_NEW_TOKENS
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    enable_reasoning: Optional[bool] = False


async def load_model_internal(req: LoadModelRequest) -> Dict[str, Any]:
    """Internal helper to load or switch the active OpenVINO pipeline."""
    settings = load_settings()
    root = Path(settings.get("model_root", MODEL_ROOT))
    model_path = root / req.model_name

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model directory '{req.model_name}' not found")

    meta = read_model_metadata(model_path)
    is_ok, reason = check_device_compatibility(req.device, meta)
    if not is_ok:
        # Fallback to CPU if requested device is incompatible
        if req.device.upper() != "CPU":
            req.device = "CPU"
        else:
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


# ============================================================
# APP INITIALIZATION
# ============================================================

def create_app() -> FastAPI:
    app = FastAPI(title="OpenVINO GenAI - OpenAI-Compatible API Server", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def get_index():
        return {
            "status": "online",
            "service": "OpenVINO GenAI - OpenAI-Compatible API Server",
            "version": "0.1.0",
            "active_model": session.model_name or None,
            "active_device": session.device if session.is_loaded() else None,
            "endpoints": {
                "models": "/v1/models",
                "chat_completions": "/v1/chat/completions",
                "health": "/health",
            },
        }

    @app.get("/health")
    async def health_check():
        current_mem = powershell_gpu_memory()
        return {
            "status": "healthy",
            "loaded": session.is_loaded(),
            "model_name": session.model_name,
            "device": session.device,
            "display_type": session.metadata.display_type if session.metadata else "",
            "precision": session.metadata.precision.upper() if session.metadata else "",
            "gpu_memory_human": human_bytes(current_mem),
        }

    @app.post("/api/unload")
    async def unload_model():
        session.unload()
        return {"status": "unloaded"}

    # ========================================================
    # OPENAI-COMPATIBLE API (For Open WebUI & External Clients)
    # ========================================================

    @app.get("/v1/models")
    @app.get("/models")
    async def openai_list_models():
        settings = load_settings()
        root = Path(settings.get("model_root", MODEL_ROOT))
        models = find_models(root)

        now_ts = int(time.time())
        model_objects = []
        for m in models:
            # 1. Base model entry (uses default configured device)
            model_objects.append({
                "id": m.name,
                "object": "model",
                "created": now_ts,
                "owned_by": "openvino",
                "permission": [],
                "root": m.name,
                "parent": None,
            })
            # 2. Explicit GPU target entry
            model_objects.append({
                "id": f"{m.name} (GPU)",
                "object": "model",
                "created": now_ts,
                "owned_by": "openvino",
                "permission": [],
                "root": m.name,
                "parent": None,
            })
            # 3. Explicit CPU target entry
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

        # Extract device target from model name if present
        target_dev = None
        clean_model_name = req_model or ""

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
                await load_model_internal(LoadModelRequest(
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

