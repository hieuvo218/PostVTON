# PostVTON
An agentic framewok with post processing stage enhance realism, robustness, scalability of VTON.
# Introduction
# Key Idea (Agentic Post-Processing)
# Framework Overview (Diagram)
# Installation
# Quick Start
# Run Try-On Server (CatVTON + OOTDiffusion)

On the GPU/server machine:

1) Install deps:

`pip install -r requirements.tryon_server.txt`

2) Start server:

`python scripts/run_tryon_server.py --host 0.0.0.0 --port 8000`

# Run Gradio UI

On the UI machine (can be the same machine):

1) Install deps:

`pip install -r requirements.system.txt`

2) Set env vars:

- `TRYON_SERVER_URL=http://<server-ip>:8000`
- Optional: `GEMINI_API_KEY=...`

3) Launch:

`python scripts/run_gradio_ui.py`

# Pipeline Explanation
# Experiments & Results
# Ablation Studies
# Citation
