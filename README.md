# OpenVINO GenAI - OpenAI-Compatible API Server & CLI

A high-performance local AI inference suite and OpenAI-compatible serving backend for [**OpenVINO GenAI (`openvinotoolkit/openvino.genai`)**](https://github.com/openvinotoolkit/openvino.genai), featuring dual **LLM** (Text) and **VLM** (Vision-Language) multimodal support, native Intel Arc GPU (XMX) acceleration, **Heterogeneous Pipeline Parallelism**, **Mixture of Experts (MoE) Disk Offloading**, an interactive terminal CLI, visual PDF & image inspection, and direct integration with [**Open WebUI**](https://github.com/open-webui/open-webui).

> [!NOTE]
> **Open WebUI Architecture & Intel Guide**:
> In accordance with the official Intel tutorial [Demonstrating integration of Open WebUI with OpenVINO Model Server](https://docs.openvino.ai/2025/model-server/ovms_demos_integration_with_open_webui.html), local inference is served via a dedicated OpenAI-compatible API server (`python run_server.py` on port `8000`) and connected to a full-featured [Open WebUI](https://github.com/open-webui/open-webui) instance (on port `3000`). This gives you instant access to RAG, web search, document indexing, voice TTS, and conversation branching with native Windows Intel Arc acceleration.

> [!TIP]
> **Heterogeneous Execution with Pipeline Parallelism (Preview)**:
> In accordance with the official [OpenVINO Heterogeneous Execution Guide](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html#pipeline-parallelism-preview), models that exceed single-device memory limits can run across combined hardware devices (`HETERO:GPU,CPU`) using the `PIPELINE_PARALLEL` distribution policy. The pipeline stages are executed sequentially across devices in priority order (GPU compute first, CPU host memory second), enabling large LLMs and MoE models to run smoothly without memory exhaustion.

> [!TIP]
> **Mixture of Experts (MoE) Disk Offloading**:
> As introduced in the OpenVINO release notes, partial expert weight offloading (`OFFLOAD_RATIO` and `ENABLE_MMAP`) enables large MoE models (such as `Gemma-4-26B-A4B` or `Qwen3-30B-A3B`) to run even on devices with 16 GB memory while maintaining high tokens-per-second generation rates. Only active experts and backbone layers are kept in GPU memory while inactive expert weights are paged dynamically from disk.

> [!WARNING]
> **Intel NPU Advisory (Currently Disabled)**:
> Intel NPU execution is **temporarily disabled / unstable** in the current OpenVINO GenAI release (`NPU compiler does not support platform 'AUTO_DETECT' for architecture 5010`). Please select **`GPU`** (Intel Arc), **`HETERO:GPU,CPU`**, or **`CPU`** as your target hardware device for all generation and chat workflows until upstream Intel compiler updates are delivered.

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

- **Heterogeneous Execution with Pipeline Parallelism (Preview)**:
  - Configures `HETERO:GPU,CPU` using `MODEL_DISTRIBUTION_POLICY=PIPELINE_PARALLEL` according to the [Intel OpenVINO Pipeline Parallelism Documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html#pipeline-parallelism-preview).
  - Automatically divides sequential model stages across Intel Arc GPU and host CPU memory based on operation memory requirements.
  - Follows Intel best practices by prioritizing GPU for initial acceleration stages and placing host CPU at the end of the priority chain for memory fallback.
  - Enables models too large for single GPU VRAM to execute seamlessly without crashing.

- **Mixture of Experts (MoE) Offloading to Disk**:
  - Automatically identifies MoE architectures (e.g. `Gemma-4-26B-A4B`, `Qwen3-30B-A3B`) via metadata inspection (`enable_moe_block`, `num_experts`, `top_k_experts`).
  - Configures the GPU plugin's native MoE offload properties (`OFFLOAD_RATIO` and `ENABLE_MMAP`).
  - Allows 26B/30B MoE models to run locally even on 16 GB memory systems by keeping only active experts in VRAM and paging inactive expert weights on-demand from disk.
  - Dynamic user control via CLI (`--moe-offload`, `--moe-ratio`) and chat command (`/moe <ratio%>`).

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
  - Exposes `(GPU)`, `(HETERO)`, and `(CPU)` targets directly in Open WebUI's model dropdown.
  - Defaults to high-speed Intel Arc GPU execution or Hetero Pipeline Parallelism.
  - Interactive startup prompt lets you review and adjust load parameters (device, context length, max tokens, temperature, reasoning, MoE offloading) before launching.

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

- **Standalone Distribution & Automated Bundler**:
  - [`ovchat_fallback.py`](ovchat_fallback.py) provides a lightweight launcher with fallback resolution.
  - Built-in automated bundler ([`scripts/build_standalone.py`](scripts/build_standalone.py)) creates an executable zip archive ([`dist/ovchat.pyz`](dist/ovchat.pyz)) with zero build dependencies using Python's standard `zipapp`.

---

## Project Architecture & Directory Layout

[![Architecture diagram for tudeptrai1712/my-ov-script on a pale lavender background. Five groups show the project components: User Interfaces includes Interactive CLI, API Server, Browser Client, and Fallback Runner; Model Control includes Model Catalog, Model Metadata, Device Selection, Settings Manager, and Model Files; Inference Runtime includes Chat Engine and OpenVINO Runtime; Operations includes Hardware Telemetry, Memory Monitor, and User Settings; Context and State includes Document Processing, History Saver, and Chat History; Other parts includes Terminal User, OpenAI Client, and Open WebUI. The footer reads Architecture diagram · 20 parts · 30 connections, beside an Explore it interactively button. GitDiagram appears at the top right.](https://gitdiagram.com/tudeptrai1712/my-ov-script/diagram.png)](https://gitdiagram.com/tudeptrai1712/my-ov-script?utm_source=readme&utm_medium=picture)

```
my-ov-script/
├── pyproject.toml              # Standard packaging (entrypoint: ovchat)
├── requirements.txt            # Locked dependencies
├── ovchat.py                   # Root unified CLI launcher
├── run_server.py               # Convenience shortcut for API server
├── kill_server.py              # Convenience shortcut to terminate server
├── ovchat_fallback.py          # Standalone launcher
├── scripts/
│   └── build_standalone.py     # Zipapp packager -> dist/ovchat.pyz
├── ovchat/
│   ├── __init__.py             # Package exports & dependency guard
│   ├── cli.py                  # Unified subcommands (chat, serve, stop, check, models)
│   ├── engine.py               # Pure OpenVINO GenAI LLM & VLM runtime wrapper (HETERO & MoE offload)
│   ├── parser.py               # Path sanitization, quotes & inline tag extraction
│   ├── files.py                # Document text parser & PDF-to-image rasterizer
│   ├── metadata.py             # Model topology, MoE detection & metadata inspection
│   ├── history.py              # Chat logging & markdown transcript persistence
│   ├── settings.py             # Persistent user_config.json management
│   ├── ui.py                   # ANSI styling, layout formatting & progress bars
│   ├── models.py               # Model scanner & disk catalog
│   ├── hardware/               # Hardware specs & live telemetry
│   │   ├── __init__.py
│   │   ├── specs.py            # Static discovery (CPU, RAM, Intel Arc GPU EUs, NPU) & HETERO fit
│   │   └── monitor.py          # Background polling (VRAM/RAM usage)
│   ├── commands/               # Modular slash command system
│   │   ├── __init__.py
│   │   ├── base.py             # CommandContext, CommandHandler, CommandResult
│   │   ├── router.py           # Command dispatcher
│   │   ├── attachments.py      # /file, /image, /pdf-image handlers
│   │   ├── export.py           # /export, /save, /export-chat handlers
│   │   └── session.py          # /clear, /info, /think, /moe, /hetero, /unload, /load, /quit
│   └── web/                    # FastAPI OpenAI /v1 API server (GPU / HETERO / CPU)
│       ├── __init__.py
│       └── app.py
```

---

## Requirements & Prerequisites

- **Python**: 3.10 – 3.14 (64-bit)
- **Hardware**:
  - Intel Arc GPU (Alchemist or newer, e.g. Arc A-series, B-series B390/B580, Core Ultra Xe-LPG/Xe2)
  - Intel CPU (x86_64 with AVX2/AVX512/AMX)
- **Operating System**: Windows 10/11 or Linux

### Installation

Install dependencies or install the package in editable mode:

```bash
# Option A: Install dependencies directly
pip install -r requirements.txt

# Option B: Install package globally in your environment (recommended)
pip install -e .
```

---

## Quickstart Guide

### 1. Launching the Interactive Terminal CLI (Peak Speed)

For maximum inference speed without any HTTP/browser serialization overhead:

```bash
# Using installed console command:
ovchat chat

# Or using Python runner directly:
python ovchat.py
```

The interactive CLI will:
1. Auto-verify dependencies.
2. Inspect and display system hardware specifications (CPU cores, RAM total/available, Intel Arc GPU memory & EUs, NPU status).
3. Display available models alongside their hardware fit (whether the model fits into GPU memory, requires MoE disk offloading, or fits via Heterogeneous Pipeline Parallelism).
4. Prompt for execution device:
   - `[1] GPU` (Intel Arc GPU acceleration)
   - `[2] HETERO:GPU,CPU` (Pipeline Parallelism Preview: stages split between GPU and CPU)
   - `[3] CPU` (Reference baseline)
5. If an MoE model is selected (e.g. `Gemma-4-26B-A4B`), prompts to configure MoE expert offload to disk (e.g. 50%).

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
# Using the unified CLI (Default: GPU):
ovchat serve --port 8000

# Launch with Heterogeneous Pipeline Parallelism (GPU + CPU):
ovchat serve --hetero --port 8000

# Force specific device target:
ovchat serve --device HETERO:GPU,CPU
ovchat serve --device GPU
ovchat serve --device CPU

# Configure MoE offloading ratio (e.g. 50% expert weights offloaded to disk):
ovchat serve --moe-ratio 50

# Dedicated runners:
python run_server.py --port 8000
python run_server.py --hetero

# Stop running server:
ovchat stop                      # Or: python kill_server.py
```

- **Load Parameter Confirmation**: When launched interactively, the server displays current parameters (Device, Pipeline Parallelism Policy, Context Length, Max Tokens, Temperature, Top-P, Reasoning, MoE Offloading) and lets you adjust them before startup.
- **OpenAI API Base URL**: `http://127.0.0.1:8000/v1`
- **Models List Endpoint**: `http://127.0.0.1:8000/v1/models` (exposes default, `(GPU)`, `(HETERO)`, and `(CPU)` entries for direct selection in Open WebUI)
- **Chat Completions**: `http://127.0.0.1:8000/v1/chat/completions`
- **To stop the server**: Press `q` or `k` in console, type `quit`, run `ovchat stop`, or run `python kill_server.py`.

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
6. Open WebUI will instantly pull your local models:
   - Select `Gemma-4-E4B (GPU)` for high-speed Arc GPU acceleration.
   - Select `Gemma-4-26B-A4B (HETERO)` for multi-device pipeline parallel execution!

---

### 5. Inspecting Models & Hardware Fit

Inspect local models, MoE architecture indicators, their disk footprints, precision, and GPU memory fit:

```bash
# Using unified CLI:
ovchat models

# Or using Python runner:
python ovchat.py models
```

---

### 6. Standalone Executable & Zipapp Distribution

If you want a portable single-file bundle or lightweight fallback launcher:

```bash
# Option A: Run the standalone zipapp executable archive (dist/ovchat.pyz)
python dist/ovchat.pyz check
python dist/ovchat.pyz chat

# Option B: Build or refresh the standalone zipapp bundle
python scripts/build_standalone.py

# Option C: Use the single-file fallback runner
python ovchat_fallback.py
```

---

### 7. Checking Dependency & Hardware Diagnostic Health

Inspect all packages, versions, and full hardware acceleration status (CPU, RAM, Intel Arc GPU, NPU):

```bash
ovchat check
# or using the main runner:
python ovchat.py check
```

---

## Terminal CLI Commands

While chatting in the terminal (`python ovchat.py` or `python ovchat_fallback.py`), use these slash commands:

| Command | Description |
| :--- | :--- |
| `/clear` | Clears conversation history and starts a fresh transcript |
| `/info` | Displays active model, hardware device, pipeline policy, MoE offload ratio, memory usage, and load time |
| `/image <path> [page(s)] [prompt]` | Attaches an image or renders PDF pages visually (VLMs only) |
| `/file <path>` | Extracts raw text context from a file or PDF document into prompt |
| `/think` | Toggles reasoning mode (`/think on` or `/think off`) |
| `/moe [ratio% \| off]` | Configures MoE expert offload to disk (e.g. `/moe 50` or `/moe off`) |
| `/hetero [devices]` | Switches to Heterogeneous Pipeline Parallelism (e.g. `/hetero GPU,CPU`) |
| `/export [file.txt]` | Exports the **last generated** assistant answer to a `.txt` file |
| `/export ["prompt" "file.txt"]` | Runs prompt (supports inline `/file` & `/image`), streams output, and saves answer directly to `.txt` |
| `/export-chat [file.txt]` | Exports the entire chat conversation to a `.txt` file |
| `/unload` | Ejects the active model from memory and frees GPU VRAM |
| `/load` | Prompts with a list of available models to load without restarting |
| `/quit` | Exits the chat session |

---

## Configuration & Environment Variables

User defaults are saved automatically in [`user_config.json`](user_config.json). You can also configure default paths and runtime modes via environment variables:

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OVCHAT_MODEL_ROOT` | `D:\AI models\openvino-genai` | Root directory where OpenVINO models are stored |
| `OVCHAT_HISTORY_DIR` | `D:\AI models\openvino-genai\chat history` | Directory where chat session markdown transcripts are saved |
| `OVCHAT_HETERO_PRIORITY` | `GPU,CPU` | Device priority list for Heterogeneous Pipeline Parallelism |
| `OVCHAT_MODEL_DISTRIBUTION_POLICY` | `PIPELINE_PARALLEL` | Model distribution policy for multi-device execution |
| `OVCHAT_ENABLE_MOE_OFFLOAD` | `1` | Enable Mixture of Experts weight offloading to disk (`1`/`0`) |
| `OVCHAT_MOE_OFFLOAD_RATIO` | `50` | Percentage of MoE expert weights offloaded to disk (1-99%) |

### Sample `user_config.json`:
```json
{
  "model_root": "D:\\AI models\\openvino-genai",
  "history_dir": "D:\\AI models\\openvino-genai\\chat history",
  "selected_model": "Gemma-4-E4B",
  "selected_device": "GPU",
  "hetero_priorities": "GPU,CPU",
  "model_distribution_policy": "PIPELINE_PARALLEL",
  "enable_moe_offload": true,
  "moe_offload_ratio": 50,
  "context_length": 32768,
  "max_new_tokens": 8192,
  "temperature": 0.7,
  "top_p": 0.95,
  "enable_reasoning": true
}
```

---

## License

MIT License. Feel free to use and modify for your own local AI workflows!
