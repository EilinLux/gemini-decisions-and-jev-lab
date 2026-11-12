# Gemini Decisions API (`gemini-decision-1`) — `uv` Python Lab

A zero-output-token, calibrated-probability demo suite for **Google's Decisions API** (`gemini-decision-1`), managed with **`uv`**.

## Quickstart

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



