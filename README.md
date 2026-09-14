# OpenVINO GenAI - OpenAI-Compatible API Server & CLI

A high-performance local AI inference suite and OpenAI-compatible serving backend for [**OpenVINO GenAI (`openvinotoolkit/openvino.genai`)**](https://github.com/openvinotoolkit/openvino.genai), featuring dual **LLM** (Text) and **VLM** (Vision-Language) multimodal support, native Intel Arc GPU (XMX) acceleration, an interactive terminal CLI, and direct integration with [**Open WebUI**](https://github.com/open-webui/open-webui).

> [!NOTE]
> **Open WebUI Architecture & Intel Guide**:
> In accordance with the official Intel tutorial [Demonstrating integration of Open WebUI with OpenVINO Model Server](https://docs.openvino.ai/2025/model-server/ovms_demos_integration_with_open_webui.html), local inference is served via a dedicated OpenAI-compatible API server (`python run_server.py` on port `8000`) and connected to a full-featured [Open WebUI](https://github.com/open-webui/open-webui) instance (on port `3000`). This gives you instant access to RAG, web search, document indexing, voice TTS, and conversation branching with native Windows Intel Arc acceleration.

> [!WARNING]
> **Intel NPU Advisory (Currently Not Working)**:
> Intel NPU execution is **currently not working / unstable** in the current OpenVINO GenAI release. Please choose **`GPU`** (or `CPU`) as your target hardware device for all generation and chat workflows until upstream fixes are delivered.

> [!WARNING]
> **Performance Notice (Web UI vs. CLI)**:
> While Open WebUI provides a rich graphical interface, browser DOM updates, network streaming, and HTTP serialization introduce measurable overhead. For benchmarking or maximum raw token generation speed (highest tokens/second and minimal latency), **use the interactive CLI instead (`python ovchat.py`)**.

> [!IMPORTANT]
> **Intel Hardware Requirements (Intel Hardware Only)**:
> This application is specifically tailored for Intel hardware architectures:
> - **GPU**: Requires **Intel Alchemist (1st Gen Arc / Xe-HPG) or newer** (including Battlemage B-series, Meteor Lake Xe-LPG, and Lunar Lake / Arrow Lake Xe2). Older graphics generations (Intel UHD, early Iris Xe) lack the required Matrix/DPAS instructions and will either run slowly or fail compilation.
> - **NPU**: Requires an integrated Intel Neural Processing Unit on **Intel Core Ultra Series 1 (Meteor Lake) and newer**, or **Intel Core Series 3 and newer**. Note that in OpenVINO GenAI, VLMs (vision-language models) require GPU or CPU; the NPU is supported for compatible text-only causal LLMs.
> - **NPU**: Currently **not working / disabled** due to upstream runtime instability. Please select **`GPU`** or **`CPU`**.

## Key Features

- **Dual LLM & VLM Multimodal Support**:
  - Automatically identifies model architecture via metadata inspection.
  - Dynamically activates `VLMPipeline` for multimodal vision models (e.g., Gemma-4-E4B) and `LLMPipeline` for text models (Llama, Qwen, Mistral, DeepSeek).
  - Attach images via paperclip file picker, clipboard paste (`Ctrl+V`), drag & drop, or CLI `/image <path>`.

- **Universal File & Document Extraction**:
  - Upload documents (`.pdf`, `.txt`, `.py`, `.json`, `.csv`, `.md`, `.log`, `.yaml`, etc.).
  - Built-in PDF parser powered by `pypdf` extracts page-by-page text context directly into the prompt.
  - Multi-file staging with thumbnail cards and one-click removal.

- **LM Studio-Style Model Load / Unload & Eject**:
  - **⏏ Eject Button** in top navigation header, sidebar, and settings modal.
  - Frees GPU VRAM and system memory on demand (`gc.collect()`).
  - Swap models dynamically in Web UI or via CLI `/load` and `/unload` commands without restarting Python.

- **ChatGPT-Style Markdown & Table Rendering**:
  - 100% offline GitHub Flavored Markdown (GFM) powered by bundled `marked.js`.
  - Crisp tables with distinct headers, zebra-striped rows, borders, and hover effects.
  - Code blocks with language badges and instant `"Copy"` buttons.
  - Structured typography (headings, blockquotes, ordered/unordered lists).

- **Output Action Bar (Copy & Retry)**:
  - **Copy**: One-click clipboard copy of clean output (internal reasoning tokens automatically filtered).
  - **Retry**: Regenerate any turn on the fly with the original prompt, attachments, and configuration.

- **Real-Time Hardware Telemetry**:
  - Continuous sampling of **CPU**, **RAM**, **GPU Compute**, **Intel Arc XMX / Neural Engine**, **GPU 3D Engine**, and **Dedicated / Shared VRAM**.
  - Visualized via real-time progress bars in the sidebar and a compact glanceable top-nav telemetry pill.

- **Console Server Kill & Stop Controls**:
  - Press `q` or `k` in the console, or type `quit` / `kill` + `Enter` to gracefully stop the API server.
  - Kill running instances from any terminal via `python kill_server.py` or `python ovchat.py --kill`.

- **Reasoning / Thinking Mode (`/think`)**:
  - Native support for reasoning models (Gemma 4, DeepSeek, etc.).
  - Live toggle via CLI `/think on` / `/think off` or configuration parameter prompt.

- **Persistent Markdown Chat History**:
  - Automatically saves formatted sessions with YAML frontmatter, timestamps, token counts, throughput, and TTFT directly to:
    `D:\AI models\openvino-genai\chat history`

- **Self-Healing Dependency Verifier**:
  - Automatically checks all required pip packages on every launch.
  - If any dependency is missing, it auto-installs it via pip before the application initializes.
  - Takes only ~10 ms when all packages are installed.

- **Standalone Single-File Fallback Script**:
  - [`ovchat_fallback.py`](ovchat_fallback.py) is a completely self-contained single file with zero local module dependencies, containing full LLM/VLM, document/image upload, reasoning, memory tracking, and chat logging.

---

## Project Architecture

```
my ov script/
├── ovchat/                     # Core application package
│   ├── __init__.py             # Package exports & startup dependency verification
│   ├── verifier.py             # Automatic pip dependency verifier & diagnostics
│   ├── sysmon.py               # Hardware telemetry engine (CPU, RAM, GPU Compute, XMX, VRAM)
│   ├── config.py               # Constants, directories, and context defaults
│   ├── files.py                # Document & PDF text extraction utilities
│   ├── metadata.py             # Model metadata inspection & VLM detection
│   ├── devices.py              # Hardware device discovery & compatibility filtering
│   ├── memory.py               # Live GPU/shared VRAM counter monitoring
│   ├── history.py              # Markdown session transcript auto-saver
│   ├── settings.py             # Persistent user settings manager (user_config.json)
│   ├── chat.py                 # Multi-turn interactive chat engine & CLI commands
│   ├── cli.py                  # CLI runner, menus, and launch sequence
│   ├── ui.py                   # ANSI terminal formatting & helpers
│   └── web/                    # FastAPI OpenAI-compatible serving backend
│       ├── __init__.py
│       ├── __main__.py         # Run via python -m ovchat.web
│       └── app.py              # FastAPI server (OpenAI /v1 API: models & chat completions)
├── requirements.txt            # Python dependencies
├── user_config.json            # Auto-persisted user preferences
├── kill_server.py              # Instant server terminator (Port 8000)
├── run_server.py               # Dedicated OpenAI-compatible API server (Port 8000)
├── ovchat.py                   # Primary entry point (CLI & server)
└── ovchat_fallback.py          # Standalone single-file fallback script
```

---

## Requirements & Prerequisites

- **Python**: 3.10 – 3.14 (64-bit)
- **Hardware**:
  - Intel Arc GPU (Alchemist or newer, e.g. Arc A-series, B-series B390/B580, Core Ultra Xe-LPG/Xe2)
  - Intel NPU (Core Ultra Series 1+, Core Series 3+)
- **Operating System**: Windows 10/11 or Linux

### Dependencies

Dependencies are checked and installed **automatically on every launch**. Alternatively, install them manually:

```bash
pip install -r requirements.txt
```

---

## Quickstart Guide

### 1. Launching the OpenAI-Compatible API Server (Port 8000)

Start the high-performance local server:

```bash
python run_server.py --port 8000
# or using the main runner:
python ovchat.py --server

# Optional flags:
python run_server.py --device CPU        # Force CPU inference
python run_server.py --device GPU        # Force Intel Arc GPU inference
python run_server.py -y                  # Skip parameter confirmation prompt
python run_server.py --kill              # Stop running server
```

- **Load Parameter Confirmation**: When launched interactively, the server displays current parameters (Device, Context Length, Max Tokens, Temperature, Top-P, Reasoning) and lets you change them before startup.
- **OpenAI API Base URL**: `http://127.0.0.1:8000/v1`
- **Models List Endpoint**: `http://127.0.0.1:8000/v1/models` (exposes both default, `(GPU)`, and `(CPU)` model entries for selection in Open WebUI)
- **Chat Completions**: `http://127.0.0.1:8000/v1/chat/completions`
- **To stop the server**: Press `q` or `k` in console, type `quit`, or run `python kill_server.py`.

---

### 2. Setting Up & Connecting Open WebUI (Port 3000)

Following the [OpenVINO Open WebUI Integration Demo](https://docs.openvino.ai/2025/model-server/ovms_demos_integration_with_open_webui.html), launch Open WebUI using either method:

#### Option A: Via Python / Pip (Recommended on Windows)
```bash
pip install open-webui
open-webui serve --port 3000
```

#### Option B: Via Docker
```bash
docker run -d -p 3000:8080 -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
```

#### Connecting in Open WebUI:
1. Open your browser to **`http://localhost:3000`**.
2. Go to **Settings (gear icon) > Connections > OpenAI API**.
3. Set **API Base URL**:
   - If running natively: `http://127.0.0.1:8000/v1`
   - If running inside Docker: `http://host.docker.internal:8000/v1`
4. Set **API Key**: `openvino` (or any string).
5. Click **Verify / Save Connection**.
6. Open WebUI will instantly pull your local models (e.g. `Gemma-4-E4B`) from your server with complete streaming, document upload, and vision support!

---

### 3. Launching the Interactive Terminal CLI (Peak Speed)

For maximum inference speed without any HTTP/browser serialization overhead:

```bash
python ovchat.py
```

The interactive CLI will:
1. Auto-verify dependencies.
2. Let you select any model from `D:\AI models\openvino-genai`.
3. Filter compatible hardware (Intel Arc GPU, CPU, NPU).
4. Run with maximum raw tokens/second throughput.

---

### 4. Managing Models (Download, Convert & Delete)

Use the dedicated model manager to inspect disk usage, download pre-converted OpenVINO models, convert raw Hugging Face models, or safely delete models:

#### Interactive Menu
```bash
python manage_models.py
# or using the main runner:
python ovchat.py --manage
```

#### Command-Line Operations:
```bash
# List local models with sizes, precision, and supported devices
python manage_models.py list

# Download a pre-converted OpenVINO INT4 model from Hugging Face:
python manage_models.py download OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov

# Convert and export a raw Hugging Face model or local directory to OpenVINO INT4:
python manage_models.py convert Qwen/Qwen2.5-0.5B-Instruct

# Delete a model to free disk space:
python manage_models.py delete Gemma-4-E4B
```

> [!TIP]
> **Pre-Converted Models vs. Local Conversion**:
> Whenever possible, download official pre-converted models from the `OpenVINO/` namespace on Hugging Face (e.g. `OpenVINO/Llama-3.2-3B-Instruct-int4-ov`, `OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov`). Models that are already in OpenVINO INT4 format are downloaded directly without unnecessary re-conversion. When converting raw models, the manager always converts them into optimized **INT4** format for Intel GPU and NPU acceleration.

---

### 5. Standalone Fallback Script

If you ever need a 100% self-contained single script requiring no package folders:

```bash
python ovchat_fallback.py
```

### 6. Checking Dependency Health

Inspect all packages and their exact installed versions:

```bash
python ovchat.py --deps
```

---

## Terminal CLI Commands

While chatting in the terminal (`python ovchat.py` or `python ovchat_fallback.py`), use these slash commands:

| Command | Description |
| :--- | :--- |
| `/clear` | Clears conversation history and starts a fresh transcript |
| `/info` | Displays active model, hardware device, memory usage, and load time |
| `/file <path>` | Attaches a file or PDF document context to your next prompt |
| `/image <path>` | Attaches an image file for vision understanding (VLMs only) |
| `/think` | Toggles reasoning mode (`/think on` or `/think off`) |
| `/unload` | Ejects the active model from memory and frees GPU VRAM |
| `/load` | Prompts with a list of available models to load without restarting |
| `/quit` | Exits the chat session |

---

## Configuration & Environment Variables

User defaults are saved automatically in [`user_config.json`](user_config.json). You can also configure default paths via environment variables:

| Environment Variable | Default Path | Description |
| :--- | :--- | :--- |
| `OVCHAT_MODEL_ROOT` | `D:\AI models\openvino-genai` | Root directory where OpenVINO models are stored |
| `OVCHAT_HISTORY_DIR` | `D:\AI models\openvino-genai\chat history` | Directory where chat session markdown transcripts are saved |

### Sample `user_config.json`:
```json
{
  "model_root": "D:\\AI models\\openvino-genai",
  "history_dir": "D:\\AI models\\openvino-genai\\chat history",
  "selected_model": "Gemma-4-E4B",
  "selected_device": "GPU",
  "context_length": 32768,
  "max_new_tokens": 8192,
  "temperature": 0.7,
  "top_p": 0.95,
  "enable_reasoning": false
}
```

---

## License

MIT License. Feel free to use and modify for your own local AI workflows!

