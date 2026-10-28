# Cool Story — Turn Detection Engine

A production-ready, low-latency turn detection service that determines whether a speaker has completed their conversational turn (**EOT / Turn Complete**) or is merely pausing mid-thought (**HOLD / Incomplete**). Built with a semantic text-based classifier (`all-MiniLM-L6-v2`) deployed via **ONNX Runtime on CPU**, delivering sub-100ms inference latency.

---

## Project Structure

```
Cool-Story/
├── iteration_1/
│   ├── data_generator.py      # Synthetic dataset generation
│   ├── train.py               # Model fine-tuning
│   ├── export_onnx.py         # PyTorch → ONNX export
│   ├── evaluate.py            # Model evaluation
│   ├── main.py                # FastAPI server (port 8000)
│   ├── data/                  # train/val/test JSONL splits
│   └── turn_detector_model/   # Trained model + ONNX export
├── iteration_2/
│   ├── data_generator2.py     # Enhanced ASR-normalized generator
│   ├── train2.py              # Model fine-tuning (data → model)
│   ├── export_onnx2.py        # PyTorch → ONNX export
│   ├── evaluate2.py           # Model evaluation
│   ├── main2.py               # FastAPI server (port 8001)
│   ├── data/                  # train/val/test JSONL splits
│   └── turn_detector_model/   # Trained model + ONNX export
├── stress_test.py             # Async load/stress testing script
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container definition
├── venv/                      # Virtual environment (created locally)
└── README.md
```

---

## System Architecture

```
[Microphone Stream]
       │
       ▼
[Streaming VAD Engine]  (Silero VAD / WebRTC)
       │
  Silence Gap (>200ms)
       │
       ▼
[Streaming ASR Engine]  (Whisper / Sherpa-ONNX)
       │
  Partial Transcripts
       │
       ▼
[Turn Detection Service]  (<20ms per request)
  (DistilBERT / MiniLM via ONNX Runtime CPU)
       │
   ┌───┴───┐
   ▼       ▼
[HOLD]  [EOT]
Keep listening   Yield turn → Fire LLM
```

---

## 1. Data Pipeline (`data_generator.py`)

### Strategy

A turn detection dataset requires pairs of partial utterances labeled as either **Complete** ($y=1$) or **Incomplete** ($y=0$). The generator uses:

1. **Complete templates** — 22 realistic conversational sentences (e.g., bookings, confirmations, questions).
2. **Synthetic incomplete fragments** — Complete templates truncated at random token boundaries (2 to N-1 tokens), optionally appended with filler words ("uh", "um", "like", "you know", "well") with 30% probability.
3. **Hand-crafted incomplete fragments** — 20 mid-thought utterances ending at conjunctions, prepositions, hesitations, and mid-clause positions.

### Split

Stratified 70 / 15 / 15 train / validation / test split, preserving class balance.

**Generated dataset:** Train=3,513 | Val=753 | Test=753

### Data Flow

---

## 2. Model Baseline, Training, and Evaluation

### Model

Fine-tuned `sentence-transformers/all-MiniLM-L6-v2` (a compact 22-layer transformer with 384-dim embeddings) for binary sequence classification. The model has ~11M parameters including the newly initialized classification head.

### Training

- **Epochs:** 3
- **Batch size:** 32 (train), 64 (eval)
- **Learning rate:** 3e-5
- **Max sequence length:** 64 tokens
- **Optimizer:** AdamW with weight decay 0.01
- **Metric for best model:** F1

### Evaluation Results (Test Set)

| Metric | HOLD (Incomplete) | EOT (Complete) |
|--------|:-----------------:|:--------------:|
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| F1-Score | 1.00 | 1.00 |

- **Accuracy:** 1.00 (753/753 correct)
- **Confusion Matrix:** [[378, 0], [0, 375]] — zero errors
- **ROC-AUC:** 1.00


---

## 3. Serving and Performance

### FastAPI Server (`main.py`)

Endpoints:
- `GET /health` — health check
- `POST /predict` — turn detection inference

Request body:
```json
{
  "text": "I think we should proceed",
  "context": "What are your thoughts?",
  "threshold": 0.5
}
```

Response:
```json
{
  "turn_complete": true,
  "confidence": 0.99,
  "decision": "EOT",
  "inference_latency_ms": 38.69
}
```


---

## 4. Production Monitoring and Continuous Improvement

### Key Metrics to Track

| Category | Metric | Target / Alert Threshold |
|----------|--------|-------------------------|
| **Operational** | Inference Latency (P50/P95/P99) | P99 < 100ms |
| **Operational** | QPS & Error Rates (4xx/5xx) | < 1% error rate |
| **Operational** | CPU/Memory Utilization | < 80% |
| **Conversational Quality** | User Barge-In Rate | < 300ms post-EOT |
| **Conversational Quality** | Silence Fallback Rate | < 1200ms timeout |
| **Conversational Quality** | Prediction Confidence Distribution | Monitor drift |

#### Critical Quality Metrics

* **User Barge-In Rate (False Positive EOT):** The user speaks within 300–800ms of the agent beginning its response. Signals premature interruption.
* **Silence Fallback Rate (False Negative EOT):** Model predicts HOLD but user has stopped. Agent response delayed until hard timeout (e.g., 1200ms).
* **Prediction Confidence Distribution Drift:** Track EOT vs. HOLD ratio per rolling hour. Sudden skew indicates demographic shifts, ASR issues, or corrupted audio.

### Continuous Improvement Loop (Active Learning)

1. **Automated Failure Case Logging:**
   - **Barge-in** → extract partial transcript prior to agent speech, label $y=0$ (HOLD)
   - **Hard Silence Timeout** → extract uncommitted transcript, label $y=1$ (EOT)
2. **Confidence-Stratified Triage:** Route utterances where $0.40 \le p \le 0.60$ to human annotation queue.
3. **Shadow Mode / Canary Rollouts:** Deploy updated checkpoints to shadow traffic; compare predicted EOT timestamps without allowing interruption.
4. **Multilingual Expansion:** Fine-tune multilingual backbones (`paraphrase-multilingual-MiniLM-L12-v2`, `xlm-roberta-base`) using the same active learning pipeline.

---

## 5. Architectural Discussion

### Limitations of the Text-Only Approach

| Limitation | Description |
|------------|-------------|
| **Acoustic Blindness** | Identical text ("Yeah, that's fine") can have rising intonation (more to follow) or falling intonation (final). Text-only cannot disambiguate. |
| **ASR Dependence** | Upstream ASR errors or omitted words due to noise cause garbled input to the classifier. |
| **Punctuation Sensitivity** | Models rely on terminal punctuation added by ASR. Without streamed punctuation, accuracy drops unless trained on unpunctuated text. |

### Audio vs. Text vs. Multimodal vs. Pure VAD

| Approach | Latency | Semantic Awareness | Prosodic Awareness | Failure Modes |
|----------|---------|-------------------|-------------------|---------------|
| **Traditional VAD** (Silero, WebRTC) | Ultra Low (<10ms) | None | Pure energy/voice | Cuts off during pauses, thinking breaks, hesitations |
| **Text-Only Model** (DistilBERT/MiniLM) | Low (5–20ms) | Very High | None | Cannot detect pitch rise/drop, trailing audio |
| **Acoustic-Only Model** | Medium (30–60ms) | Low | Very High | Cannot detect grammatical incompleteness on flat tone |
| **Multimodal Model** (Audio + Text) | Medium (40–90ms) | Very High | Very High | Highest complexity, requires synchronized alignment |

**Why VAD is Insufficient Alone:** VAD answers *"Is sound being produced?"* rather than *"Is the thought finished?"* Humans routinely pause 400–700ms mid-sentence. Standard VAD either interrupts prematurely or requires unnatural 1200ms silence timeouts, making conversation feel sluggish.

### Adapting Transcribers (Whisper / Parakeet) for Turn Detection

1. **Special End-of-Turn Token Injection:** Fine-tune Whisper's decoder with an explicit `<|eot_turn|>` token placed at true speaker transitions. During streaming decoding, if the next autoregressive token is `<|eot_turn|>`, trigger an external interrupt immediately.

2. **Encoder Head Attachment:** Attach a lightweight binary classification head directly to Whisper's/Parakeet's acoustic encoder output states. Evaluates frame sequences every 80ms, predicting turn completion from learned latent representations of prosody and semantics — without waiting for the slow autoregressive decoder.

### End-to-End Voice Agent Integration

```
[User Mic] ──(WebSockets/WebRTC)──> [Audio Buffer]
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
          [Streaming Silero VAD]                   [Fast ASR (Sherpa/Whisper)]
                   │                                           │
         Silence detected (>250ms)                             │ Partial text stream
                   │                                           │
                   └─────────────────┬─────────────────────────┘
                                     ▼
                        [Turn Detection Engine]
                        (POST /predict via ONNX)
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
             p(EOT) >= 0.5                      p(EOT) < 0.5
                    │                                 |
                  [EOT]                             [HOLD]      
```
---

## 6. How to Run

### Prerequisites

Python 3.10+

### Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 1. Generate Dataset

```powershell
python data_generator.py
```
Output: `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`

### 2. Train Model

```powershell
python train.py
```
Output: `turn_detector_model/` (PyTorch weights + tokenizer)

### 3. Export to ONNX

```powershell
python export_onnx.py
```
Output: `turn_detector_model/model.onnx`

### 4. Evaluate

```powershell
python evaluate.py
```

### 5. Start Server

```powershell
python main.py
# or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 6. Run Stress Test

```powershell
python stress_test.py
```

### Docker

```bash
docker build -t turn-detection-engine .
docker run -p 8000:8000 turn-detection-engine
```

---

## 7. Results Summary

| Component | Result |
|-----------|--------|
| Dataset | 3,513 train / 753 val / 753 test (stratified) |
| Model | all-MiniLM-L6-v2 fine-tuned for binary classification |
| Test Accuracy | 100% (753/753) |
| Test F1 (EOT) | 1.00 |
| Single-item P99 Latency | 36.15 ms |
| Target Budget | < 100ms ✅ |
| Stress Test Throughput | 129.53 QPS (25 concurrent, Iteration 1 on port 8000) |
| Stress Test Success Rate | 100% (500/500, Iteration 1) |


---

## 9. Iteration 2 — Enhanced Data Generator (`data_generator2.py`)

### Motivation

Iteration 1 achieved perfect test metrics (100% accuracy) but suffered from **synthetic overfitting** and **punctuation leakage**:
* The model could simply learn regex patterns (e.g., trailing `.` or `?` → EOT) rather than true syntactic completeness.
* Filler words only appeared on incomplete samples, teaching the model "filler = HOLD" — a false heuristic.
* Random token truncation created accidentally complete slices labeled as incomplete.
* Only ~40 unique seed sentences meant the model memorized n-grams instead of learning generalization.

### Fixes in Iteration 2

| Issue | Iteration 1 | Iteration 2 |
|-------|-------------|-------------|
| **Casing** | Mixed case preserved | All lowercased (raw ASR form) |
| **Punctuation** | Periods/question marks leaked | All punctuation stripped |
| **Numbers/symbols** | Left as-is (`$50`, `2`) | Normalized to words (`fifty dollars`, `two`) |
| **Seed diversity** | ~40 templates | 130+ diverse seeds across 10 conversation categories |
| **Truncation** | Random token cut (2 to N-1) | spaCy dependency-aware slicing at dangling POS (ADP, CCONJ, SCONJ, DET, PART, AUX) |
| **Disfluencies** | Fillers only on HOLD class | Fillers, stutters, self-corrections injected into **both** classes |
| **Output path** | `iteration_1/data/` | `iteration_2/data/` |

### Dataset (Iteration 2)

* **Train:** 4,199 samples | **Val:** 901 | **Test:** 900 | **Total:** 6,000
* All text lowercased, unpunctuated, numbers/symbols converted to words
* Syntactic truncation via spaCy (with keyword fallback)
* Disfluency injection (~35% filler, ~20% stutter, ~10% self-correction) on both classes

### Training & Evaluation Results (Iteration 2)

| Metric | HOLD (Incomplete) | EOT (Complete) |
|--------|:-----------------:|:--------------:|
| Precision | 1.00 | 0.9956 |
| Recall | 0.9956 | 1.00 |
| F1-Score | 0.9978 | 0.9978 |

> **Note on metrics:** The confusion matrix `[[448, 2], [0, 450]]` shows 2 false positives (HOLD utterances misclassified as EOT). Per-class precision and recall are derived directly from this matrix: HOLD precision = 448/448 = 1.00, HOLD recall = 448/450 ≈ 0.9956, EOT precision = 450/452 ≈ 0.9956, EOT recall = 450/450 = 1.00.

* **Accuracy:** 0.998 (898/900 correct)
* **Confusion Matrix:** [[448, 2], [0, 450]]
* **ROC-AUC:** 1.00

### Inference Latency (Single Item, CPU) — Iteration 2

| Metric | Latency |
|--------|---------|
| P50 | 9.72 ms |
| P95 | 13.56 ms |
| P99 | 15.52 ms |

Compared to Iteration 1 (P99: 36.15 ms), iteration 2 is **2.3× faster** at the tail, driven by optimized sequence length handling and reduced token variance in ASR-normalized strings.

### Stress Testing (HTTP Server)

The `stress_test.py` script now accepts a port argument and uses ASR-normalized (lowercase, unpunctuated) input phrases appropriate for both models:

```powershell
# Start both servers
python iteration_1/main.py   # port 8000
python iteration_2/main2.py  # port 8001

# Stress test Iteration 1 (port 8000)
python stress_test.py 8000

# Stress test Iteration 2 (port 8001)
python stress_test.py 8001
```

The Section 7 stress test results (129.53 QPS, P99: 563.78 ms) were collected against **Iteration 1** on port 8000. These numbers include HTTP round-trip, JSON serialization, and Python/ONNX overhead under 25 concurrent connections — they are not representative of the ONNX model's raw inference latency (P99: 36.15 ms for Iteration 1, 15.52 ms for Iteration 2, measured directly in `evaluate.py`/`evaluate2.py`).

### Running Iteration 2

```powershell
# 1. Generate enhanced dataset
python iteration_2/data_generator2.py

# 2. Train model on iteration_2/data/
python iteration_2/train2.py

# 3. Export to ONNX
python iteration_2/export_onnx2.py

# 4. Evaluate
python iteration_2/evaluate2.py

# 5. Start v2 server (port 8001)
python iteration_2/main2.py
```

---

## 10. Future Work

- **Multimodal extension:** Add acoustic/prosodic features alongside text for disambiguating intonation.
- **Streaming Whisper integration:** Replace standalone ASR + turn detection with an encoder-head architecture for end-to-end audio turn detection.
- **Multilingual support:** Fine-tune `xlm-roberta-base` or `paraphrase-multilingual-MiniLM-L12-v2` for Spanish, French, German.
- **Online learning:** Deploy shadow-mode canary rollouts with automated failure-case relabeling for continuous improvement.
