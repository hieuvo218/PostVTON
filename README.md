# PostVTON
An agentic framewok with post processing stage enhance realism, robustness, scalability of VTON.

# Introduction
## Framework Overview (Diagram)
<p align="center">
  <img width="1385" height="547"
       src="https://github.com/user-attachments/assets/338acca1-cd25-4fd6-81cd-4b604409272f" />
  <br>
  <em>Figure 1: PostVTON framework: A multi-agent system designed to enhance the results of virtual try-on models, featuring coordinated agents for management, evaluation, problem detection,
planning, and correction.</em>
</p>

## Installation

For development (recommended), install the package in editable mode:

`pip install -e .`

Then install the full dependency stack you need:

- System/UI: `pip install -r requirements.system.txt`
- Try-on server: `pip install -r requirements.tryon_server.txt`

# Quick Start
## Run Try-On Server (CatVTON + OOTDiffusion)

On the GPU/server machine:

1) Install deps:

`pip install -r requirements.tryon_server.txt`

2) Start server:

`python scripts/run_tryon_server.py --host 0.0.0.0 --port 8000`

## Run Gradio UI

On the UI machine (can be the same machine):

1) Install deps:

`pip install -r requirements.system.txt`

2) Set env vars:

- `TRYON_SERVER_URL=http://<server-ip>:8000`
- Optional: `GEMINI_API_KEY=...`

3) Launch:

`python scripts/run_gradio_ui.py`

# Results
## Module-Level Latency Analysis
<img width="1317" height="723" alt="image" src="https://github.com/user-attachments/assets/0b254d8a-49b4-4ed9-9ad2-734590717d6c" />

## Quantitative Results
<img width="1333" height="368" alt="image" src="https://github.com/user-attachments/assets/dd34a9c0-400e-417d-b203-a7a6b20dec96" />

## Qualitative Results
<img width="1333" height="368" alt="image" src="https://github.com/user-attachments/assets/d6c15dee-74ea-4d8f-b181-4c5e5e38be20" />
