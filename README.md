# System One AI in Practice: TypeSafe AI’s Jev & Google’s Decisions API (`gemini-decision-1`)

> *"Generative models produce text for people to read. When your code needs a judgment to branch on, generating and parsing text adds latency and brittle parsing logic."*

This repository is a hands-on Python engineering lab (managed with **`uv`**) exploring the architectural shift from autoregressive text generation to **zero-output-token, calibrated decision models**—comparing **Google’s Decisions API (`gemini-decision-1`)** and the paradigm pioneered by **TypeSafe AI’s Jev**.

---

### 1. System 1 vs. System 2 Thinking
In his landmark book *Thinking, Fast and Slow*, psychologist Daniel Kahneman divides human cognition into two modes:
* **System 1 (Fast, Automatic, Instinctive):** Ask someone *"What is 2 × 2?"* and the answer (`4`) appears instantly without deliberate effort.
* **System 2 (Slow, Deliberate, Analytical):** Ask someone *"What is 17 × 24?"* and they must pause, decompose the steps, and compute `408`.

For the past several years, AI engineering has been dominated by **System 2 architectures**: autoregressive Large Language Models that write prose token by token, and reasoning models that generate lengthy chain-of-thought traces. Yet inside production software, most AI calls aren't essays or math proofs—they are split-second **System 1 judgment calls**:
* *Is this customer email a refund request? (Yes / No)*
* *Which department owns this ticket? (Billing / Technical / Shipping)*
* *How urgent is the churn risk? (Low → Critical)*

Forcing an LLM to answer those questions via JSON Structured Outputs still triggers an autoregressive decoding loop (predicting `{`, `"`, keys, colons, and values one token at a time) and returns **uncalibrated confidence**—because models trained with **RLHF** are rewarded for *sounding* confident even when wrong, while **RLVR** models are rewarded only for final answer correctness rather than uncertainty awareness.

### 2. TypeSafe AI’s Jev & RLCD Calibration
Founded in San Francisco in 2024, **TypeSafe AI** introduced **Jev**, a proprietary non-generative "System 1" model built specifically for software control flow. Instead of writing text, Jev takes structured `state` and typed questions (`noul` for yes/no, `choice` for fixed categories, `score` for ordered rubrics) and outputs **calibrated probabilities** in a single forward pass—trained via **RLCD (Reinforcement Learning for Calibrated Decisions)** so that an `80%` predicted probability actually holds true `80%` of the time.

### 3. Google’s Hyperscale Answer: The Decisions API (`gemini-decision-1`)
Google answered this paradigm shift with the **Decisions API** (`v1beta` in the Google GenAI SDK), powered by **`gemini-decision-1`**. It adopts the exact same three typed primitives (`noul`, `choice`, `score`) and **zero-output-token** architecture (`total_output_tokens: 0`), while scaling it across Google's multimodal infrastructure:
* **Parallel, Isolated Evaluation:** Ask up to **1,000 independent questions** against one shared context in a single forward pass with nearly flat latency.
* **Natively Multimodal:** Evaluate text, serialized JSON, images, audio, video, and PDFs (up to a **262,144-token context window**) without separate OCR or transcription pipelines.
* **Calibrated Confidence Gates:** Pair every answer with a mathematically grounded `confidence` float (`0.0` to `1.0`) to decide between full automation and human-in-the-loop escalation.

### 4. Jevons Paradox in Software Engineering
Named after economist William Stanley Jevons—who observed in 1865 that more fuel-efficient steam engines caused Britain's total coal consumption to *skyrocket* as steam spread into new industries—**Jevons Paradox** explains why System 1 models matter. When an AI decision runs in ~150 ms with **0 output tokens**, AI stops being a chatbot bolted onto the UI and becomes an inline primitive running across **every database row, every log line, every webhook, and every video frame**.

---

## Repository Lab Guide

Each script in this repository tests a core pillar of the System 1 decision architecture:

| Script | Primitive(s) Tested | What It Demonstrates |
| :--- | :--- | :--- |
| **`01_basic_noul.py`** | `noul` (Yes/No) | Binary predicate evaluation with explicit `true_description` / `false_description` criteria, `[0.01, 0.99]` probability bounds, and `0` output tokens. |
| **`02_support_router.py`** | `noul` + `choice` + `score` | Multi-dimensional ticket routing in one forward pass plus **three-zone calibrated thresholding** (`> 0.90` auto-route, `< 0.10` ignore, `0.10–0.90` human escalation). |
| **`03_multimodal_qa.py`** | `noul` (Audio & Video) | Direct classification of MP3 audio and low-resolution MP4 video URIs with zero intermediate transcription steps. |
| **`04_guardrails_injection.py`** | `noul` (Dual-Gate) | Running an adversarial prompt-injection check in parallel with business logic at the edge before invoking an expensive System 2 LLM. |
| **`05_benchmark_vs_flash.py`** | All vs. Pydantic JSON | Side-by-side latency, output-token overhead, and calibrated vs. verbalized confidence benchmark comparing `gemini-decision-1` against `gemini-2.5-flash`. |
| **`test_decisions.py`** | `pytest` Contract Suite | Automated assertions verifying zero output tokens, normalized `choice` distributions, and linear `score` rubric ordering. |

---

## Quickstart with `uv`



1. **Set your API Key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY

   ```

2. **Run the Demos via `uv run`:**
   ```bash
   uv run 01_basic_noul.py
   uv run 02_support_router.py
   uv run 03_multimodal_qa.py
   uv run 04_guardrails_injection.py
   ```


3. **Run the Automated `pytest` Suite:**
   ```bash
   uv run pytest -v
   ```



