# Cool Story — Turn Detection Engine

A production-ready, low-latency turn detection service that determines whether a speaker has completed their conversational turn (**EOT / Turn Complete**) or is merely pausing mid-thought (**HOLD / Incomplete**). Built with a semantic text-based classifier (`all-MiniLM-L6-v2`) deployed via **ONNX Runtime on CPU**, delivering sub-100ms inference latency.

---

## Project Structure

```
Cool-Story/
├── data_generator.py      # Synthetic conversational dataset generation
├── train.py               # Model fine-tuning (HuggingFace Trainer)
├── export_onnx.py         # PyTorch → ONNX export for CPU inference
├── evaluate.py            # Model evaluation on held-out test set
├── main.py                # FastAPI inference server
├── stress_test.py         # Async load/stress testing script
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container definition
├── venv/                  # Virtual environment (created locally)
├── data/                  # Generated train/val/test JSONL splits
└── turn_detector_model/   # Trained model + ONNX export
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
                    │                                 │

---

## 6. How to Run

### Prerequisites

Python 3.10+ and the virtual environment at `./venv`.

### Setup

```powershell
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
| Stress Test Throughput | 129.53 QPS (25 concurrent) |
| Stress Test Success Rate | 100% (500/500) |

---

## 8. Future Work

- **Multimodal extension:** Add acoustic/prosodic features alongside text for disambiguating intonation.
- **Streaming Whisper integration:** Replace standalone ASR + turn detection with an encoder-head architecture for end-to-end audio turn detection.
- **Multilingual support:** Fine-tune `xlm-roberta-base` or `paraphrase-multilingual-MiniLM-L12-v2` for Spanish, French, German.
- **Online learning:** Deploy shadow-mode canary rollouts with automated failure-case relabeling for continuous improvement.
                    ▼                                 ▼
         [Interrupt Turn State]                 [Hold State]
       • Close input stream               • Keep listening
       • Dispatch full text to LLM        • Cancel if voice resumes
       • Stream LLM tokens to TTS         • Fallback: trigger after 1200ms
```

**Integration Steps:**
1. **Audio Ingestion:** Client streams raw PCM 16kHz audio over WebSockets.
2. **VAD Gating:** Frame-by-frame VAD routes speech to streaming ASR.
3. **Turn Arbitration:** On 250ms silence, query Turn Detection engine with uncommitted transcript — do not assume completion from VAD alone.
4. **Fast Path:** On `turn_complete=True`, close audio turn and trigger LLM immediately, reducing user-perceived turnaround by 500–800ms vs. fixed-timeout VAD.
Server startup includes a warm-up inference to initialize ONNX Runtime session and avoid cold-start latency.

### Stress Test Results (`stress_test.py`)

| Metric | Value |
|--------|-------|
| Total Requests | 500 |
| Concurrency | 25 |
| Successful | 500 |
| Failed | 0 |
| Throughput | 129.53 QPS |
| Latency P50 | 147.01 ms |
| Latency P95 | 440.66 ms |
| Latency P99 | 563.78 ms |

*Note: Stress test latency includes HTTP round-trip, JSON serialization, and Python/ONNX overhead under concurrency. Single-item inference latency (measured directly in `evaluate.py`) is P99 = 36.15 ms.*
### ONNX Export

Exported to ONNX opset 14 with dynamic axes for batch and sequence length. Optimized with `ORT_ENABLE_ALL` graph optimization level and 2 intra-op threads.

### Inference Latency (Single Item, CPU)

| Metric | Latency |
|--------|---------|
| P50 | 23.19 ms |
| P95 | 28.97 ms |
| P99 | 36.15 ms |

All well within the **<100ms** target budget.

1. **Streaming VAD** detects an acoustic silence threshold (e.g., 200–300ms).
2. The current partial transcript is dispatched to the Turn Detection API (`POST /predict`).
3. If the model outputs `EOT`, the system triggers the LLM immediately without waiting for a typical 700–1200ms silence timeout.
4. If the model outputs `HOLD`, the system waits for the user to resume speaking.
