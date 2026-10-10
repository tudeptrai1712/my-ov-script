# OpenVINO GenAI - OpenAI-Compatible API Server & CLI

A high-performance local AI inference suite and OpenAI-compatible serving backend for [**OpenVINO GenAI (`openvinotoolkit/openvino.genai`)**](https://github.com/openvinotoolkit/openvino.genai), featuring dual **LLM** (Text) and **VLM** (Vision-Language) multimodal support, native Intel Arc GPU (XMX) acceleration, an interactive terminal CLI, visual PDF & image inspection, and direct integration with [**Open WebUI**](https://github.com/open-webui/open-webui).

> [!NOTE]
> **Open WebUI Architecture & Intel Guide**:
> In accordance with the official Intel tutorial [Demonstrating integration of Open WebUI with OpenVINO Model Server](https://docs.openvino.ai/2025/model-server/ovms_demos_integration_with_open_webui.html), local inference is served via a dedicated OpenAI-compatible API server (`python run_server.py` on port `8000`) and connected to a full-featured [Open WebUI](https://github.com/open-webui/open-webui) instance (on port `3000`). This gives you instant access to RAG, web search, document indexing, voice TTS, and conversation branching with native Windows Intel Arc acceleration.

> [!WARNING]
> **Intel NPU Advisory (Currently Disabled)**:
> Intel NPU execution is **temporarily disabled / unstable** in the current OpenVINO GenAI release (`NPU compiler does not support platform 'AUTO_DETECT' for architecture 5010`). Please select **`GPU`** (Intel Arc) or **`CPU`** as your target hardware device for all generation and chat workflows until upstream Intel compiler updates are delivered.

> [!WARNING]
> **Performance Notice (Web UI vs. CLI)**:
> While Open WebUI provides a rich graphical interface, browser DOM updates, network streaming, and HTTP serialization introduce measurable overhead. For benchmarking or maximum raw token generation speed (highest tokens/second and minimal latency), **use the interactive CLI instead (`python ovchat.py`)**.

> [!IMPORTANT]
> **Intel Hardware Requirements**:
> This application is specifically tailored for Intel hardware architectures:
> - **GPU**: Requires **Intel Alchemist (1st Gen Arc / Xe-HPG) or newer** (including Battlemage B-series, Meteor Lake Xe-LPG, and Lunar Lake / Arrow Lake Xe2). Older graphics generations (Intel UHD, early Iris Xe) lack the required Matrix/DPAS instructions and will either run slowly or fail compilation.
> - **NPU**: Integrated Intel Neural Processing Unit on Core Ultra Series 1 (Meteor Lake) and newer. Note: NPU is currently **temporarily disabled** due to upstream compiler driver errors.
> - **CPU**: x86_64 CPU with AVX2, AVX512, or AMX instructions.

---

## Key Features

- **Dual LLM & VLM Multimodal Support**:
  - Automatically identifies model architecture via metadata inspection (`model.safetensors.index.json`, `config.json`, or `.xml` topology).
  - Dynamically activates `VLMPipeline` for multimodal vision models (e.g., Gemma-4-E4B) and `LLMPipeline` for text models (Llama, Qwen, Mistral, DeepSeek).
  - Multi-image tensor batching: send one or multiple image/PDF page tensors in a single chat turn.

- **Visual PDF Processing (Native PDF-to-Image for Scanned Exams & Math)**:
  - High-speed visual rendering powered by `pypdfium2` (Chromium PDFium engine, ~25ms per page).
  - Directly renders PDF pages into OpenVINO image tensors (`ov.Tensor`) without requiring manual screenshot or image extraction tools.
  - Ideal for scanned math tests, exam papers, geometry diagrams, chemical formulas, and tables where raw text extraction fails.
  - Flexible page selection: `/image doc.pdf 1`, `/image doc.pdf 1-3`, `/image doc.pdf all`, or `/image doc.pdf:2`.
  - Combine file, page number, and prompt in a single line: `/image "de-toan.pdf" 1 Giải chi tiết câu 1`.

- **Universal Document & Inline Text Extraction**:
  - Supports documents (`.pdf`, `.txt`, `.py`, `.json`, `.csv`, `.md`, `.log`, etc.).
  - Built-in PDF text parser powered by `pypdf` extracts page-by-page text context directly into the prompt.
  - Supports inline document attachments in prompts (e.g. `/file report.pdf <prompt>`).

- **Flexible One-Shot Prompt & Answer Export (`/export`)**:
  - Generate answers and export directly to a text file in one command:  
    `/export "prompt (incl /image doc.pdf:1 or /file doc.pdf)" "output.txt"`
  - Automatically extracts embedded `/file` or `/image` context, streams generation to the terminal, and saves the final response directly to file.
  - Or export the last generated answer: `/export output.txt`.

- **Direct Open WebUI Serving (OpenAI-Compatible FastAPI Backend)**:
  - Serves local models via OpenAI `/v1` endpoints (`/v1/models`, `/v1/chat/completions`) with full streaming support.
  - Defaults to high-speed Intel Arc GPU execution.
  - Interactive startup prompt lets you review and adjust load parameters (device, context length, max tokens, temperature, reasoning) before launching.

- **Dynamic Model Load / Unload & Eject**:
  - Frees GPU VRAM and system memory on demand (`gc.collect()`).
  - Swap models dynamically via CLI `/load` and `/unload` commands without restarting Python.

- **Real-Time Hardware Telemetry**:
  - Continuous sampling of **CPU**, **RAM**, **GPU Compute**, **Intel Arc XMX / Neural Engine**, **GPU 3D Engine**, and **Dedicated / Shared VRAM**.

- **Reasoning / Thinking Mode (`/think`)**:
  - Native support for reasoning models (Gemma 4, DeepSeek, etc.).
  - Live toggle via CLI `/think on` / `/think off` or configuration parameter prompt.

- **Persistent Markdown Chat History**:
  - Automatically saves formatted sessions with YAML frontmatter, timestamps, token counts, throughput, and TTFT directly to:
    `D:\AI models\openvino-genai\chat history`

- **Self-Healing Dependency Verifier**:
  - Automatically checks all required pip packages (`openvino-genai`, `pypdfium2`, `pypdf`, `Pillow`, `fastapi`, `uvicorn`, etc.) on every launch.
  - If any dependency is missing, it auto-installs it via pip before the application initializes.

- **Standalone Single-File Fallback Script**:
  - [`ovchat_fallback.py`](ovchat_fallback.py) is a completely self-contained single file with zero local module dependencies, containing full LLM/VLM, document/image/PDF-to-image upload, `/export` command, reasoning, memory tracking, and chat logging.

---

## Project Architecture

[![Architecture diagram for tudeptrai1712/my-ov-script on a pale lavender background. Five groups show the project components: User Interfaces includes Interactive CLI, API Server, Browser Client, and Fallback Runner; Model Control includes Model Catalog, Model Metadata, Device Selection, Settings Manager, and Model Files; Inference Runtime includes Chat Engine and OpenVINO Runtime; Operations includes Hardware Telemetry, Memory Monitor, and User Settings; Context and State includes Document Processing, History Saver, and Chat History; Other parts includes Terminal User, OpenAI Client, and Open WebUI. The footer reads Architecture diagram · 20 parts · 30 connections, beside an Explore it interactively button. GitDiagram appears at the top right.](https://gitdiagram.com/tudeptrai1712/my-ov-script/diagram.png)](https://gitdiagram.com/tudeptrai1712/my-ov-script?utm_source=readme&utm_medium=picture)

---

## Requirements & Prerequisites

- **Python**: 3.10 – 3.14 (64-bit)
- **Hardware**:
  - Intel Arc GPU (Alchemist or newer, e.g. Arc A-series, B-series B390/B580, Core Ultra Xe-LPG/Xe2)
  - Intel CPU (x86_64 with AVX2/AVX512/AMX)
- **Operating System**: Windows 10/11 or Linux

### Dependencies

Dependencies are checked and installed **automatically on every launch**. Alternatively, install them manually:

```bash
pip install -r requirements.txt
```

---

## Quickstart Guide

### 1. Launching the Interactive Terminal CLI (Peak Speed)

For maximum inference speed without any HTTP/browser serialization overhead:

```bash
python ovchat.py
```

The interactive CLI will:
1. Auto-verify dependencies.
2. Let you select any model from `D:\AI models\openvino-genai`.
3. Filter compatible hardware (Intel Arc GPU, CPU).
4. Run with maximum raw tokens/second throughput.

---

### 2. Processing Images and PDFs Visually (VLMs)

When running a vision-language model like **`gemma-4-E4B-it-int4-ov`**, you can pass images (`.jpg`, `.png`, `.webp`, `.bmp`) or visual PDF pages directly.

#### Using PDF files with `/image`:
PDFs are automatically rendered into crisp image tensors via Chromium's PDFium engine (`pypdfium2`):

```text
# Render specific page (1-based index)
/image "C:\path\to\de-toan.pdf" 1

# Render page range (e.g. pages 1 to 3)
/image "C:\path\to\de-toan.pdf" 1-3

# Render all pages (capped at 10 pages for safety)
/image "C:\path\to\de-toan.pdf" all

# Colon syntax
/image "C:\path\to\de-toan.pdf:2"

# Combine image attachment and prompt in one turn:
/image "C:\path\to\de-toan.pdf" 1 Giải chi tiết câu 1 và câu 2 trong đề thi này

# Using one-shot export:
/export "Giải câu 4 /image \"C:\path\to\de-toan.pdf:1\"" "cau4_solution.txt"
```

> [!TIP]
> **`/image` vs `/file` for PDFs**:
> - Use **`/image`** for scanned exam papers, math formulas, geometry figures, diagrams, and handwriting. The model **sees** the page visually.
> - Use **`/file`** for text-heavy documents (e.g. books, source code, legal agreements) where you only need the raw text extracted.

---

### 3. Launching the OpenAI-Compatible API Server (Port 8000)

Start the high-performance local server:

```bash
python run_server.py --port 8000
# or using the main runner:
python ovchat.py --server

# Optional flags:
python run_server.py --device GPU        # Force Intel Arc GPU inference (default)
python run_server.py --device CPU        # Force CPU inference
python run_server.py -y                  # Skip parameter confirmation prompt
python run_server.py --kill              # Stop running server
```

- **Load Parameter Confirmation**: When launched interactively, the server displays current parameters (Device, Context Length, Max Tokens, Temperature, Top-P, Reasoning) and lets you change them before startup.
- **Default Device**: Intel Arc **GPU** is the recommended default.
- **OpenAI API Base URL**: `http://127.0.0.1:8000/v1`
- **Models List Endpoint**: `http://127.0.0.1:8000/v1/models` (exposes both default, `(GPU)`, and `(CPU)` model entries for selection in Open WebUI)
- **Chat Completions**: `http://127.0.0.1:8000/v1/chat/completions`
- **To stop the server**: Press `q` or `k` in console, type `quit`, or run `python kill_server.py`.

---

### 4. Setting Up & Connecting Open WebUI (Port 3000)

Following the [OpenVINO Open WebUI Integration Guide](https://docs.openvino.ai/2025/model-server/ovms_demos_integration_with_open_webui.html), launch Open WebUI:

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

### 5. Managing Models (Download, Convert & Delete)

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
> Whenever possible, download official pre-converted models from the `OpenVINO/` namespace on Hugging Face (e.g. `OpenVINO/Llama-3.2-3B-Instruct-int4-ov`, `OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov`). Models that are already in OpenVINO INT4 format are downloaded directly without unnecessary re-conversion. When converting raw models, the manager always converts them into optimized **INT4** format for Intel GPU acceleration.

---

### 6. Standalone Fallback Script

If you ever need a 100% self-contained single script requiring no package folders:

```bash
python ovchat_fallback.py
```

---

### 7. Checking Dependency Health

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
| `/image <path> [page(s)] [prompt]` | Attaches an image or renders PDF pages visually (VLMs only) |
| `/file <path>` | Extracts raw text context from a file or PDF document into prompt |
| `/think` | Toggles reasoning mode (`/think on` or `/think off`) |
| `/export [file.txt]` | Exports the **last generated** assistant answer to a `.txt` file |
| `/export ["prompt" "file.txt"]` | Runs prompt (supports inline `/file` & `/image`), streams output, and saves answer directly to `.txt` |
| `/export-chat [file.txt]` | Exports the entire chat conversation to a `.txt` file |
| `/unload` | Ejects the active model from memory and frees GPU VRAM |
| `/load` | Prompts with a list of available models to load without restarting |
| `/quit` | Exits the chat session |

### Advanced `/export` Examples:

```text
# 1. One-shot prompt with visual PDF page attachment:
/export "Giải chi tiết câu 1 /image \"C:\path\to\de-toan.pdf:1\"" "cau1_solution.txt"

# 2. One-shot prompt with document text attachment saved to text file:
/export "Summarize key findings in 3 bullet points /file report.pdf" "summary.txt"

# 3. Quoted file paths with spaces:
/export "Translate to French /file \"my notes/meeting.txt\"" "french_meeting.txt"

# 4. Quick prompt generation to timestamped file:
/export "Write a Python script for quicksort"

# 5. Save the previous answer after reading it in console:
/export final_answer.txt
```

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
