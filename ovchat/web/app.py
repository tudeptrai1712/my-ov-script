"""
OpenVINO GenAI - OpenAI-Compatible API Server.
High-performance backend for Open WebUI, OpenAI SDKs, RAG, and Agentic frameworks.
Supports:
  - Standard Chat Completions: POST /v1/chat/completions
  - OpenAI Responses API: POST /v1/responses
  - Tool calling / Function execution for agents
  - Multimodal vision inputs (VLMs)
  - Heterogeneous Pipeline Parallelism (HETERO:GPU,CPU)
  - Mixture of Experts (MoE) disk offloading
"""

import asyncio
import base64
import io
import json
import os
import queue
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import openvino as ov
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..config import (
    DEFAULT_CONTEXT,
    DEFAULT_ENABLE_MOE_OFFLOAD,
    DEFAULT_ENABLE_REASONING,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_DISTRIBUTION_POLICY,
    DEFAULT_MOE_OFFLOAD_RATIO,
    MODEL_ROOT,
)
from ..engine import InferenceEngine
from ..hardware.monitor import powershell_gpu_memory
from ..metadata import check_device_compatibility, read_model_metadata
from ..models import find_models, model_size
from ..settings import load_settings, save_settings
from ..ui import human_bytes


# ============================================================
# UNIFIED ENGINE INSTANCE
# ============================================================

engine = InferenceEngine()

# Backward-compatibility alias
session = engine


class LoadModelRequest(BaseModel):
    model_name: str
    device: str = "GPU"
    context_length: Optional[int] = DEFAULT_CONTEXT
    max_new_tokens: Optional[int] = DEFAULT_MAX_NEW_TOKENS
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    enable_reasoning: Optional[bool] = False
    enable_moe_offload: Optional[bool] = None
    moe_offload_ratio: Optional[int] = None
    model_distribution_policy: Optional[str] = None


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
        if req.device.upper() != "CPU":
            req.device = "CPU"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Device '{req.device}' cannot run this model: {reason}",
            )

    moe_offload = (
        req.enable_moe_offload
        if req.enable_moe_offload is not None
        else settings.get("enable_moe_offload", DEFAULT_ENABLE_MOE_OFFLOAD)
    )
    moe_ratio = (
        req.moe_offload_ratio
        if req.moe_offload_ratio is not None
        else settings.get("moe_offload_ratio", DEFAULT_MOE_OFFLOAD_RATIO)
    )
    policy = (
        req.model_distribution_policy
        if req.model_distribution_policy
        else settings.get("model_distribution_policy", DEFAULT_MODEL_DISTRIBUTION_POLICY)
    )

    try:
        await asyncio.to_thread(
            engine.load_model,
            model_path=model_path,
            device=req.device,
            context_length=req.context_length or DEFAULT_CONTEXT,
            max_new_tokens=req.max_new_tokens or DEFAULT_MAX_NEW_TOKENS,
            temperature=req.temperature if req.temperature is not None else 0.7,
            top_p=req.top_p if req.top_p is not None else 0.95,
            enable_reasoning=req.enable_reasoning,
            enable_moe_offload=moe_offload,
            moe_offload_ratio=moe_ratio,
            model_distribution_policy=policy,
        )
    except Exception as e:
        engine.unload()
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    return {
        "status": "loaded",
        "model_name": engine.model_name,
        "device": engine.device,
        "display_type": meta.display_type,
        "is_moe": meta.is_moe,
        "moe_summary": meta.moe_summary,
        "load_time_seconds": round(engine.load_time, 2),
    }


# ============================================================
# AGENTIC TOOL & MULTIMODAL HELPERS
# ============================================================

def _decode_image_tensor(img_url: Optional[str]) -> Optional[ov.Tensor]:
    """Decodes a base64 or URL string into an OpenVINO image tensor."""
    if not img_url or not engine.metadata or not engine.metadata.is_vlm:
        return None
    try:
        import numpy as np
        from PIL import Image

        if "," in img_url:
            _, b64data = img_url.split(",", 1)
        else:
            b64data = img_url
        raw_bytes = base64.b64decode(b64data)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return ov.Tensor(np.array(pil_img))
    except Exception as e:
        print("Failed to decode image data:", e)
        return None


def _format_tools_prompt(tools: List[Dict[str, Any]]) -> str:
    """Formats tools schema into system prompt instructions for agentic models."""
    tool_lines = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function", t)
        name = fn.get("name", "tool")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        tool_lines.append(f"- `{name}`: {desc}\n  Parameters: {json.dumps(params)}")

    return (
        "\n\nYou have access to the following tools:\n"
        + "\n".join(tool_lines)
        + "\n\nWhen a tool is required, output your response in this exact format:\n"
        + "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"val\"}}</tool_call>\n"
    )


def _extract_tool_calls(text: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Inspects generated text for tool calls and returns (tool_calls, cleaned_text)."""
    tool_calls = []
    matches = re.findall(r"<tool_call>\s*({.*?})\s*</tool_call>", text, flags=re.DOTALL)
    cleaned = text

    for m in matches:
        try:
            parsed = json.loads(m.strip())
            name = parsed.get("name") or parsed.get("function")
            args = parsed.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if name:
                call_id = f"call_{uuid.uuid4().hex[:12]}"
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": str(name),
                        "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                    },
                })
                cleaned = cleaned.replace(f"<tool_call>{m}</tool_call>", "").strip()
        except Exception:
            pass

    return (tool_calls if tool_calls else None, cleaned)


async def _resolve_and_load_model(
    req_model: Optional[str],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: Optional[int] = None,
) -> None:
    """Resolves requested model and device target, auto-switching pipeline if needed."""
    settings = load_settings()
    root = Path(settings.get("model_root", MODEL_ROOT))
    available_models = [m.name for m in find_models(root)]

    target_dev = None
    clean_model_name = req_model or ""

    for suffix, dev_val in [
        (" (HETERO:GPU,CPU)", "HETERO:GPU,CPU"),
        (" (HETERO)", "HETERO:GPU,CPU"),
        (":HETERO", "HETERO:GPU,CPU"),
        ("/hetero", "HETERO:GPU,CPU"),
        (" (GPU)", "GPU"),
        (" (NPU)", "NPU"),
        (" (CPU)", "CPU"),
        (":GPU", "GPU"),
        (":NPU", "NPU"),
        (":CPU", "CPU"),
        ("/gpu", "GPU"),
        ("/npu", "NPU"),
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
        if not engine.is_loaded or engine.model_name != clean_model_name or engine.device != target_dev:
            await load_model_internal(LoadModelRequest(
                model_name=clean_model_name,
                device=target_dev,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
            ))

    if not engine.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded and could not resolve model.")


# ============================================================
# APP INITIALIZATION
# ============================================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenVINO GenAI - OpenAI-Compatible API Server",
        version="0.3.0",
        description="High-performance backend supporting Chat Completions and the OpenAI Responses API",
    )

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
            "version": "0.3.0",
            "active_model": engine.model_name or None,
            "active_device": engine.device if engine.is_loaded else None,
            "hetero_support": "Pipeline Parallelism Preview (HETERO:GPU,CPU)",
            "moe_offloading": "Mixture of Experts disk offloading (OFFLOAD_RATIO)",
            "endpoints": {
                "models": "/v1/models",
                "chat_completions": "/v1/chat/completions",
                "responses": "/v1/responses",
                "health": "/health",
            },
        }

    @app.get("/health")
    async def health_check():
        current_mem = powershell_gpu_memory()
        return {
            "status": "healthy",
            "loaded": engine.is_loaded,
            "model_name": engine.model_name,
            "device": engine.device,
            "display_type": engine.metadata.display_type if engine.metadata else "",
            "precision": engine.metadata.precision.upper() if engine.metadata else "",
            "is_moe": engine.metadata.is_moe if engine.metadata else False,
            "moe_summary": engine.metadata.moe_summary if engine.metadata else "",
            "moe_offload_ratio": (
                engine.moe_offload_ratio
                if (engine.is_loaded and engine.metadata and engine.metadata.is_moe)
                else None
            ),
            "model_distribution_policy": (
                engine.model_distribution_policy
                if (engine.is_loaded and engine.device.upper().startswith("HETERO"))
                else None
            ),
            "gpu_memory_human": human_bytes(current_mem),
        }

    @app.post("/api/unload")
    async def unload_model():
        engine.unload()
        return {"status": "unloaded"}

    # ========================================================
    # MODELS CATALOG
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
            model_objects.append({
                "id": m.name,
                "object": "model",
                "created": now_ts,
                "owned_by": "openvino",
                "permission": [],
                "root": m.name,
                "parent": None,
            })
            model_objects.append({
                "id": f"{m.name} (GPU)",
                "object": "model",
                "created": now_ts,
                "owned_by": "openvino",
                "permission": [],
                "root": m.name,
                "parent": None,
            })
            model_objects.append({
                "id": f"{m.name} (HETERO)",
                "object": "model",
                "created": now_ts,
                "owned_by": "openvino",
                "permission": [],
                "root": m.name,
                "parent": None,
            })
            model_objects.append({
                "id": f"{m.name} (NPU)",
                "object": "model",
                "created": now_ts,
                "owned_by": "openvino",
                "permission": [],
                "root": m.name,
                "parent": None,
            })
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
        for suffix in (
            " (HETERO:GPU,CPU)",
            " (HETERO)",
            ":HETERO",
            "/hetero",
            " (GPU)",
            " (NPU)",
            " (CPU)",
            ":GPU",
            ":NPU",
            ":CPU",
            "/gpu",
            "/npu",
            "/cpu",
        ):
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

    # ========================================================
    # 1. OPENAI CHAT COMPLETIONS (Standard endpoint)
    # ========================================================

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def openai_chat_completions(request: Request):
        body = await request.json()
        req_model = body.get("model")
        messages = body.get("messages", [])
        tools = body.get("tools")
        stream = body.get("stream", False)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.95)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_NEW_TOKENS

        await _resolve_and_load_model(req_model, temperature, top_p, max_tokens)

        # Process messages and multimodal images
        image_tensors: List[ov.Tensor] = []
        parsed_history: List[Dict[str, str]] = []

        # If tools provided, enrich system instructions
        if tools and isinstance(tools, list):
            has_system = False
            for m in messages:
                if m.get("role") == "system":
                    m["content"] = str(m.get("content", "")) + _format_tools_prompt(tools)
                    has_system = True
                    break
            if not has_system:
                parsed_history.append({"role": "system", "content": _format_tools_prompt(tools)})

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
                            tensor = _decode_image_tensor(img_url)
                            if tensor is not None:
                                image_tensors.append(tensor)

            combined_text = "\n".join(text_parts).strip()
            parsed_history.append({"role": role, "content": combined_text})

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        if stream:
            q = queue.Queue()

            def streamer_callback(chunk: str) -> bool:
                q.put(chunk)
                return False

            def worker():
                try:
                    engine.generate_chat(
                        messages=parsed_history,
                        images=image_tensors,
                        streaming_callback=streamer_callback,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                except Exception as ex:
                    q.put(ex)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()

            async def event_generator():
                init_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": engine.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(init_chunk)}\n\n"

                while True:
                    await asyncio.sleep(0.005)
                    try:
                        item = q.get_nowait()
                    except queue.Empty:
                        continue

                    if item is None:
                        stop_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": engine.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }],
                        }
                        yield f"data: {json.dumps(stop_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    elif isinstance(item, Exception):
                        err_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": engine.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": f"\n[Error: {str(item)}]"},
                                "finish_reason": "error",
                            }],
                        }
                        yield f"data: {json.dumps(err_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    else:
                        chunk_obj = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": engine.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": item},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk_obj)}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            result = await asyncio.to_thread(
                engine.generate_chat,
                messages=parsed_history,
                images=image_tensors,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            tool_calls, clean_text = _extract_tool_calls(result.text)

            msg_payload: Dict[str, Any] = {
                "role": "assistant",
                "content": clean_text if clean_text else (None if tool_calls else result.text),
            }
            if tool_calls:
                msg_payload["tool_calls"] = tool_calls

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": engine.model_name,
                "choices": [{
                    "index": 0,
                    "message": msg_payload,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.tokens_generated,
                    "total_tokens": result.prompt_tokens + result.tokens_generated,
                },
            }

    # ========================================================
    # 2. OPENAI RESPONSES API (client.responses.create)
    # ========================================================

    @app.post("/v1/responses")
    @app.post("/responses")
    async def openai_responses(request: Request):
        body = await request.json()
        req_model = body.get("model")
        input_data = body.get("input", "")
        instructions = body.get("instructions", "")
        tools = body.get("tools") or []
        stream = body.get("stream", False)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.95)
        max_tokens = (
            body.get("max_output_tokens")
            or body.get("max_tokens")
            or DEFAULT_MAX_NEW_TOKENS
        )

        await _resolve_and_load_model(req_model, temperature, top_p, max_tokens)

        # Parse messages from instructions, tools, and input
        parsed_history: List[Dict[str, str]] = []
        image_tensors: List[ov.Tensor] = []

        system_prompt = instructions or ""
        if tools and isinstance(tools, list):
            system_prompt += _format_tools_prompt(tools)

        if system_prompt:
            parsed_history.append({"role": "system", "content": system_prompt})

        # Parse input payload (str or list of items)
        if isinstance(input_data, str):
            if input_data.strip():
                parsed_history.append({"role": "user", "content": input_data})
        elif isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, str):
                    parsed_history.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    item_type = item.get("type", "message")
                    role = item.get("role", "user")

                    if item_type == "function_call_output":
                        parsed_history.append({
                            "role": "tool",
                            "content": str(item.get("output", "")),
                        })
                        continue

                    raw_content = item.get("content", "")
                    text_parts = []
                    if isinstance(raw_content, str):
                        text_parts.append(raw_content)
                    elif isinstance(raw_content, list):
                        for part in raw_content:
                            if isinstance(part, dict):
                                p_type = part.get("type")
                                if p_type in ("input_text", "text"):
                                    text_parts.append(part.get("text", ""))
                                elif p_type in ("input_image", "image_url"):
                                    url_val = (
                                        part.get("image_url", {}).get("url")
                                        if isinstance(part.get("image_url"), dict)
                                        else part.get("image_url")
                                    )
                                    tensor = _decode_image_tensor(url_val)
                                    if tensor is not None:
                                        image_tensors.append(tensor)
                    combined_text = "\n".join(text_parts).strip()
                    if combined_text:
                        parsed_history.append({"role": role, "content": combined_text})

        if not parsed_history:
            parsed_history.append({"role": "user", "content": ""})

        response_id = f"resp_{uuid.uuid4().hex[:16]}"
        msg_id = f"msg_{uuid.uuid4().hex[:16]}"
        created_ts = int(time.time())

        if stream:
            q = queue.Queue()

            def streamer_callback(chunk: str) -> bool:
                q.put(chunk)
                return False

            def worker():
                try:
                    engine.generate_chat(
                        messages=parsed_history,
                        images=image_tensors,
                        streaming_callback=streamer_callback,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                except Exception as ex:
                    q.put(ex)
                finally:
                    q.put(None)

            threading.Thread(target=worker, daemon=True).start()

            async def responses_event_generator():
                # 1. response.created
                init_resp = {
                    "id": response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": engine.model_name,
                    "created_at": created_ts,
                    "output": [],
                }
                yield f"event: response.created\ndata: {json.dumps({'response': init_resp})}\n\n"

                # 2. output_item.added
                item_obj = {
                    "id": msg_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                }
                yield f"event: response.output_item.added\ndata: {json.dumps({'output_item': item_obj})}\n\n"

                # 3. content_part.added
                yield f"event: response.content_part.added\ndata: {json.dumps({'part': {'type': 'output_text', 'text': ''}})}\n\n"

                accumulated_text = []
                while True:
                    await asyncio.sleep(0.005)
                    try:
                        chunk = q.get_nowait()
                    except queue.Empty:
                        continue

                    if chunk is None:
                        full_reply = "".join(accumulated_text)
                        tool_calls, clean_text = _extract_tool_calls(full_reply)

                        output_items = []
                        if tool_calls:
                            for tc in tool_calls:
                                output_items.append({
                                    "id": f"fc_{uuid.uuid4().hex[:12]}",
                                    "type": "function_call",
                                    "status": "completed",
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                    "call_id": tc["id"],
                                })
                        if clean_text or not tool_calls:
                            output_items.append({
                                "id": msg_id,
                                "type": "message",
                                "status": "completed",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": clean_text or full_reply}],
                            })

                        yield f"event: response.output_text.done\ndata: {json.dumps({'text': full_reply})}\n\n"

                        final_resp = {
                            "id": response_id,
                            "object": "response",
                            "status": "completed",
                            "model": engine.model_name,
                            "output": output_items,
                            "usage": {
                                "total_tokens": len(full_reply) // 4,
                                "input_tokens": len(str(parsed_history)) // 4,
                                "output_tokens": len(full_reply) // 4,
                            },
                        }
                        yield f"event: response.completed\ndata: {json.dumps({'response': final_resp})}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    elif isinstance(chunk, Exception):
                        err_resp = {
                            "id": response_id,
                            "object": "response",
                            "status": "failed",
                            "error": {"message": str(chunk)},
                        }
                        yield f"event: response.failed\ndata: {json.dumps({'response': err_resp})}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    else:
                        accumulated_text.append(chunk)
                        yield f"event: response.output_text.delta\ndata: {json.dumps({'delta': chunk})}\n\n"

            return StreamingResponse(
                responses_event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming
            result = await asyncio.to_thread(
                engine.generate_chat,
                messages=parsed_history,
                images=image_tensors,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            tool_calls, clean_text = _extract_tool_calls(result.text)
            output_items = []
            if tool_calls:
                for tc in tool_calls:
                    output_items.append({
                        "id": f"fc_{uuid.uuid4().hex[:12]}",
                        "type": "function_call",
                        "status": "completed",
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                        "call_id": tc["id"],
                    })
            if clean_text or not tool_calls:
                output_items.append({
                    "id": msg_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": clean_text or result.text}],
                })

            return {
                "id": response_id,
                "object": "response",
                "created_at": created_ts,
                "status": "completed",
                "model": engine.model_name,
                "output": output_items,
                "usage": {
                    "total_tokens": result.prompt_tokens + result.tokens_generated,
                    "input_tokens": result.prompt_tokens,
                    "output_tokens": result.tokens_generated,
                },
            }

    return app


app = create_app()

__all__ = [
    "create_app",
    "app",
    "engine",
    "session",
    "LoadModelRequest",
    "load_model_internal",
]
