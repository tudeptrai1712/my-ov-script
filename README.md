# OpenVINO GenAI Chat & Web UI

A high-performance local AI chat suite powered by **Intel OpenVINO GenAI**, featuring dual **LLM** (Text) and **VLM** (Vision-Language) multimodal support, an **OpenWebUI / LM Studio** inspired web interface, real-time streaming, interactive terminal CLI, and an offline-first architecture.

---

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

- **Reasoning / Thinking Mode (`/think`)**:
  - Native support for reasoning models (Gemma 4, DeepSeek, etc.).
  - Web UI renders thoughts inside a collapsible `<details>` container (`Thinking Process`), leaving answers neat.
  - Live toggle via top-bar badge or CLI `/think on` / `/think off`.

- **Persistent Markdown Chat History**:
  - Automatically saves formatted sessions with YAML frontmatter, timestamps, token counts, throughput, and TTFT directly to:
    `D:\AI models\openvino-genai\chat history`
  - Searchable sidebar history list with turn replay and chat deletion.

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
│   └── web/                    # FastAPI Web UI backend & static assets
│       ├── __init__.py
│       ├── __main__.py         # Run via python -m ovchat.web
│       ├── app.py              # FastAPI server (REST endpoints & SSE streaming)
│       └── static/
│           ├── index.html      # OpenWebUI dark layout with eject & attachments
│           ├── style.css       # ChatGPT table styling, dark theme, metrics
│           ├── app.js          # Reactive client, SSE stream reader, actions
│           └── marked.min.js   # Bundled offline GFM markdown parser
├── requirements.txt            # Python dependencies
├── user_config.json            # Auto-persisted user preferences
├── run_web.py                  # Direct Web UI launcher (with auto-browser opening)
├── ovchat.py                   # Main entry point (CLI & Web UI)
└── ovchat_fallback.py          # Standalone single-file fallback script
```

---

## Requirements & Prerequisites

- **Python**: 3.10 – 3.14 (64-bit)
- **Hardware**:
  - Intel Arc GPU / Intel Iris Xe / Integrated Graphics (or Intel CPU)
  - Intel NPU (supported for compatible text LLMs)
- **Operating System**: Windows 10/11 or Linux

### Dependencies

Dependencies are checked and installed **automatically on every launch**. Alternatively, install them manually:

```bash
pip install -r requirements.txt
```

Packages used:
- `openvino` & `openvino-genai` (Inference engine & pipelines)
- `fastapi` & `uvicorn` (Web UI backend & streaming server)
- `Pillow` & `numpy` (Image processing & tensor conversion for VLMs)
- `pypdf` (PDF text extraction)
- `httpx` (HTTP/SSE client)

---

## Quickstart Guide

### 1. Launching the Web UI

Run either of the following commands:

```bash
# Using the main runner:
python ovchat.py --web

# Or using the dedicated web launcher:
python run_web.py
```

Options:
- `--port 8080`: Specify custom port (default: 8080)
- `--host 127.0.0.1`: Specify host address
- `--no-browser`: Do not automatically open the browser tab

Open **`http://127.0.0.1:8080`** in your browser.

### 2. Launching the Terminal CLI

```bash
python ovchat.py
```

The interactive CLI will:
1. Verify dependencies.
2. Prompt you to choose an available model from your model directory.
3. Automatically filter and show only compatible hardware devices.
4. Prompt for reasoning mode (if supported by the model).
5. Start the chat session.

### 3. Using the Standalone Fallback Script

If you want a portable, single-file script that requires no package folder:

```bash
python ovchat_fallback.py
```

### 4. Checking Dependency Health

Inspect all packages and their exact installed versions:

```bash
python ovchat.py --deps
# or
python run_web.py --status
```

---

## Web UI Guide

### Top Navigation Bar
- **Model Display Badge**: Shows active model name, hardware device (`GPU`, `CPU`), and type (`VLM` / `LLM`). Click to open Model Configuration.
- **⏏ Eject Button**: Instantly unloads the model and frees GPU VRAM.
- **Thinking Toggle**: Click to toggle reasoning mode (`ON` / `OFF`) in real time.
- **Clear Chat**: Clears current conversation history.

### Universal Attachments
- **Paperclip `📎` Button**: Click to select any document (`.pdf`, `.txt`, `.py`, `.json`, `.csv`) or image (`.png`, `.jpg`, `.webp`).
- **Clipboard Paste (`Ctrl+V`)**: Paste screenshots or images directly into the chat prompt.
- **Drag & Drop**: Drag files from Windows Explorer directly onto the chat area.
- Staged files appear in the attachment bar with badges, file sizes, and `×` removal buttons.

### Output Actions
- **📋 Copy**: Copies the full answer markdown to your clipboard with a confirmation badge.
- **🔄 Retry**: Regenerates the current turn.

### Settings Modal (LM Studio Style)
- **Model Selector**: Switch between discovered local models with disk size indicators.
- **Hardware Target**: Choose `GPU`, `CPU`, or `NPU`. Incompatible devices (e.g. NPU for VLM) are disabled with explanatory tooltips.
- **Generation Sliders**: Configure Context Length (bounded by model max), Max Output Tokens, Temperature, and Top-P.
- **Save as Default**: Persists your choices to `user_config.json`.
- **Apply & Load / Eject Model**: Compiles models on the fly.

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

## Hardware Compatibility Notes

- **GPU (Intel Arc & Integrated Graphics)**:
  - Supports both **LLMs** and **VLMs** (Vision-Language Models).
  - Recommended for highest throughput and lowest latency.
- **CPU**:
  - Universal fallback for all models and architectures.
- **NPU (Neural Processing Unit)**:
  - In OpenVINO GenAI, `VLMPipeline` is restricted to `GPU` and `CPU`. Attempting to run a VLM on NPU will result in compiler platform errors (`0x78000004`).
  - Supported for compatible text-only LLMs (e.g. Llama, Qwen2, Mistral, Phi-3).
  - The application automatically disables the NPU option in the UI when a VLM is selected.

---

## License

MIT License. Feel free to use and modify for your own local AI workflows!

