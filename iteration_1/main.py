import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

app = FastAPI(title="Turn Detection Engine", version="1.0.0")

MODEL_DIR = "./turn_detector_model"
tokenizer = None
ort_session = None


class PredictionRequest(BaseModel):
    text: str = Field(..., example="I think we should proceed with the deployment")
    context: str = Field(default="", example="What are your thoughts on the release?")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    turn_complete: bool
    confidence: float
    decision: str
    inference_latency_ms: float


@app.on_event("startup")
def load_artifacts():
    global tokenizer, ort_session
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(
        f"{MODEL_DIR}/model.onnx",
        session_options,
        providers=["CPUExecutionProvider"],
    )
    # Warm-up inference
    warmup_tokens = tokenizer(
        "Warmup utterance", max_length=64, padding="max_length", return_tensors="np"
    )
    ort_session.run(
        None,
        {
            "input_ids": warmup_tokens["input_ids"].astype(np.int64),
            "attention_mask": warmup_tokens["attention_mask"].astype(np.int64),
        },
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "engine": "onnxruntime-cpu"}


@app.post("/predict", response_model=PredictionResponse)
def predict_turn(req: PredictionRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    t0 = time.perf_counter()

    # Prepend context if provided
    full_prompt = f"{req.context} [SEP] {req.text}" if req.context else req.text
    tokens = tokenizer(
        full_prompt, max_length=64, truncation=True, padding="max_length", return_tensors="np"
    )

    inputs = {
        "input_ids": tokens["input_ids"].astype(np.int64),
        "attention_mask": tokens["attention_mask"].astype(np.int64),
    }
    logits = ort_session.run(None, inputs)[0][0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    eot_prob = float(probs[1])

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    turn_complete = eot_prob >= req.threshold

    return PredictionResponse(
        turn_complete=turn_complete,
        confidence=round(eot_prob if turn_complete else (1.0 - eot_prob), 4),
        decision="EOT" if turn_complete else "HOLD",
        inference_latency_ms=round(elapsed_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)