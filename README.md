# PostVTON
Virtual Try-on systems have achieved remarkable progress in fitting new garments seamlessly onto human images, offering significant potential for e-commerce and digital fashion applications. However, most existing approaches rely on large diffusion-based generation models that demand extensive training resources and provide little flexibility for local error correction once artifacts occur. We present PostVTON, an agentic postprocessing framework that enhances the realism and reliability of virtual try-on results without retraining the underlying generative model. In PostVTON, post-processing is reformulated as a collaborative multi-agent reasoning process rather than a fixed sequence of image operators. Each component operates as an autonomous agent with explicit perception, decision making, and action capabilities, specializing in distinct refinement objectives such as detecting missing accessories, identifying hand distortions, and executing targeted corrections. Unlike monolithic end-to-end pipelines, our agentic architecture offers modular deployment, parallel processing, interpretability, and flexible extension to new enhancement tools. The coordinated workflow substantially improves visual naturalness and robustness of synthesized try-on images.

To the best of our knowledge, this work is the first to apply agentic reasoning to postprocessing in virtual try-on systems, highlighting the potential of intelligent multi-agent collaboration for controllable image refinement.

# PostVTON
## Framework Overview (Diagram)
<p align="center">
  <img width="1385" height="547"
       src="https://github.com/user-attachments/assets/338acca1-cd25-4fd6-81cd-4b604409272f" />
  <br>
  <em>Figure 1: PostVTON framework: A multi-agent system designed to enhance the results of virtual try-on models, featuring coordinated agents for management, evaluation, problem detection,
planning, and correction.</em>
</p>

PostVTON is an agentic virtual try-on framework designed to improve the realism and consistency of generated try-on images through an iterative multi-agent refinement process. The system is organized into four layers: a Try-on Agent that generates candidate images using multiple state-of-the-art VTON models (e.g., OOTDiffusion and CatVTON), a Perception layer for artifact detection, a Reasoning layer for planning corrective actions, and an Action layer for image editing. These components are orchestrated by an LLM-based Manager Agent that coordinates communication and maintains a feedback-driven quality control loop.

After selecting the best try-on candidate, specialized detection agents identify issues such as deformed hands, distorted anatomy, or missing accessories. Based on these findings, a Planning Agent prioritizes corrective actions, while an Executor Agent invokes dedicated editing tools to refine the image. The corrected result is then re-evaluated and, if necessary, reprocessed through another iteration. This closed-loop workflow enables PostVTON to progressively enhance virtual try-on outputs, producing images with improved realism, consistency, and fidelity to the original inputs.


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
