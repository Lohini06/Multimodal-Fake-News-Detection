# Multimodal Fake News Detection 🔍

A deep learning system that detects fake news by jointly analyzing **news headlines** and **images** using CLIP-based multimodal fusion.

## Overview
Fake news often exploits the mismatch between misleading text and manipulated or unrelated images. This project trains a multimodal classifier that fuses visual and textual representations using OpenAI's CLIP model to flag such mismatches.

## Architecture
- **Vision encoder:** CLIP ViT-B/32
- **Text encoder:** CLIP text transformer
- **Fusion:** Concatenated embeddings → MLP classifier
- **Explainability:** Confusion matrix + training curves

## Tech Stack & Tools

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch, OpenCLIP, HuggingFace Transformers |
| **Vision + Language** | CLIP ViT-B/32, Tokenizer |
| **Data Processing** | Pandas, NumPy, scikit-learn, Pillow |
| **Visualization** | Matplotlib, Seaborn |
| **Demo App** | Streamlit |
| **Experiment Tracking** | Weights & Biases (wandb) |
| **Version Control** | Git, GitHub |
| **Environment** | Python 3.13, venv |
| **IDE** | VS Code |

## Project Structure
```plaintext
├── src/
│   ├── data/           # Dataset loading and preprocessing
│   ├── models/         # CLIP fusion classifier
│   ├── training/       # Training loop with early stopping
│   └── evaluation/     # Metrics, confusion matrix, plots
├── app/
│   └── streamlit_app.py  # Interactive demo
├── outputs/
│   └── results/        # Metrics, confusion matrix, training curves
├── config.yaml         # All hyperparameters
└── train.py            # Main entry point
```
## Quickstart

```bash
git clone https://github.com/Lohini06/Multimodal-Fake-News-Detection.git
cd Multimodal-Fake-News-Detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app/streamlit_app.py
```

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Text-only baseline | - | - |
| Image-only baseline | - | - |
| **Multimodal CLIP (ours)** | **-** | **-** |

*(Results will be updated after training on full FakeNewsNet dataset)*
