# CurvOpt-LLM

> **Curvature & Information Aware Compute Optimizer for LLMs**  
> Smarter Models, Smaller Footprint.

**Authors:** Vishalini S, Girisha Malini N, Syed Ameen G  
**Mentor:** Dr. Vinoth Chakravarthy  
**Institution:** Velammal College of Engineering and Technology, Dept. of CSE

---

## Overview

CurvOpt-LLM is a curvature-guided post-training optimization framework that selectively reduces layer precision based on sensitivity — without retraining. It estimates layer importance using a diagonal Fisher Information approximation and applies hardware-aware precision reduction only to low-impact layers.

Unlike uniform quantization, it preserves accuracy-critical layers, reducing memory usage and inference cost while maintaining controlled performance degradation.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + Vite 5 (port 5173) |
| Backend | Flask 3 + SSE streaming (port 5000) |
| ML Core | PyTorch + HuggingFace Transformers |
| Quantization | bitsandbytes (INT8, CUDA) |
| Calibration Data | HuggingFace Datasets (WikiText-2, C4, PTB) |
| Communication | Vite proxy → Flask, Server-Sent Events |

---

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- (Optional) NVIDIA GPU with CUDA for INT8 quantization

---

### Step 1 — Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

For CUDA INT8 support:
```bash
pip install bitsandbytes --upgrade
```

---

### Step 2 — HuggingFace login *(gated models only)*

Required for LLaMA, Mistral. Not needed for OPT, GPT-2, Pythia, BLOOM.

```bash
huggingface-cli login
```

---

### Step 3 — Start the backend

```bash
cd backend
python app.py
# → http://localhost:5000
```

---

### Step 4 — Start the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

### Step 5 — Open browser

Visit **http://localhost:5173**

---

## Recommended Models to Start With

| Model | Params | RAM Required | Notes |
|-------|--------|-------------|-------|
| `facebook/opt-125m` | 125M | ~1 GB | Best for testing |
| `openai-community/gpt2` | 117M | ~1 GB | Fast, reliable |
| `EleutherAI/pythia-160m` | 160M | ~1 GB | Good baseline |
| `facebook/opt-350m` | 350M | ~2 GB | Medium size |
| `bigscience/bloom-560m` | 560M | ~3 GB | Multilingual |
| `microsoft/phi-2` | 2.7B | ~6 GB | Strong small model |
| `mistralai/Mistral-7B-v0.1` | 7B | ~16 GB | Requires HF login |

---

## Algorithm

```
1. Load pretrained LLM in FP32 (baseline)
2. Sample calibration sequences from WikiText-2 / C4 / PTB
3. Compute Fisher diagonal:  k_i = E[ (dL/dθ_i)^2 ]
   → real loss.backward() over calibration set
4. Aggregate k per transformer layer (mean of param-level scores)
5. Rank layers by curvature sensitivity
6. Assign precision per layer based on PPL tolerance rho:
     High k  → FP32  (sensitive — preserve)
     Mid  k  → FP16 / BF16
     Low  k  → INT8  (CUDA + bitsandbytes only)
7. Apply mixed-precision rewrite in-place
8. Measure perplexity → auto-halt if PPL budget exceeded
9. Benchmark: TPS · memory · PPL · carbon footprint
```

---

## Results (from paper)

### Table I — Estimated Efficiency Gains from FP16 Precision Reduction

| Metric | Reduction | Reason |
|--------|-----------|--------|
| Memory footprint | 40–50% | FP16 uses half the storage |
| Memory bandwidth | 35–45% | Fewer bits fetched |
| Compute (FLOPs) | 25–40% | FP16 tensor ops use less hardware time |
| Electricity | 20–35% | Lower FLOPs + bandwidth reduces power |
| Water footprint | 20–35% | Less cooling required |
| Latency | 10–30% faster | More efficient compute |
| Carbon emissions | 20–40% lower | Directly proportional to kWh saved |

### Table II — Ablation Study: Impact of Precision Allocation Strategy

| Configuration | Memory Reduction | TPS Gain | PPL Increase |
|--------------|-----------------|----------|-------------|
| Uniform FP16 (All Layers) | 50% | 28% | 4.8% |
| Random Allocation | 50% | 23% | 3.9% |
| **CurvOpt-Guided (rho=0.5)** | **45%** | **24%** | **1.3%** |
| **CurvOpt-Guided (rho=0.7)** | **60%** | **32%** | **3.6%** |
| Attention-Only Reduction | 22% | 12% | 0.9% |

> CurvOpt delivers the best accuracy-efficiency trade-off by preserving sensitive layers.

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Check backend status + library versions |
| `POST` | `/api/optimize/stream` | Run optimization — SSE stream |
| `POST` | `/api/report` | Generate downloadable JSON report |

---

### POST `/api/optimize/stream`

**Request body:**
```json
{
  "model": "facebook/opt-125m",
  "hw": "auto",
  "calibSamples": 8,
  "seqLen": 256,
  "pplTolerance": 10,
  "allowFP16": true,
  "allowBF16": true,
  "allowINT8": false,
  "calibDataset": "wikitext"
}
```

**SSE event types:**
```json
{ "type": "progress", "pct": 44, "stage": "Computing Fisher diagonal...",
  "detail": "k_i = E[(dL/dθ_i)^2]", "log": { "msg": "...", "type": "data" } }

{ "type": "result", "layers": [...], "metrics": {...}, "footprint": {...} }

{ "type": "error", "msg": "Model load failed: ..." }
```

---

### GET `/api/health`

```json
{
  "status": "ok",
  "torch": "2.3.0",
  "transformers": "4.41.0",
  "bitsandbytes": "0.43.1",
  "datasets": "2.19.0",
  "cuda_available": true,
  "gpu": "NVIDIA GeForce RTX 3090"
}
```

---

## Project Structure

```
curvopt/
├── README.md
├── backend/
│   ├── app.py               ← Flask server — full real implementation
│   └── requirements.txt     ← torch, transformers, bitsandbytes, datasets
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js        ← proxies /api → :5000
    └── src/
        ├── main.jsx
        ├── index.css         ← full design system
        └── App.jsx           ← React app — all tabs, SSE, charts
```

---

## What's Real

| Component | Status | Detail |
|-----------|--------|--------|
| Fisher diagonal | Real | Actual `loss.backward()` over calibration set |
| Model loading | Real | HuggingFace `AutoModelForCausalLM` |
| FP16/BF16 conversion | Real | In-place `param.data = param.data.half()` |
| INT8 quantization | Real | `bitsandbytes` (CUDA only) |
| Perplexity measurement | Real | NLL cross-entropy |
| TPS benchmark | Real | Wall-clock autoregressive generation |
| Memory measurement | Real | `torch.cuda.memory_allocated()` |
| Auto-halt | Real | PPL budget enforcement with revert |
| Carbon / water / energy | Real | Computed from measured TPS + device TDP |

---

## Notes

- First run downloads the model from HuggingFace (~minutes depending on size)
- Models are cached in `~/.cache/huggingface/` after first download
- INT8 requires CUDA — on CPU/MPS only FP16/BF16 is applied
- For large models (7B+) ensure sufficient RAM/VRAM before running