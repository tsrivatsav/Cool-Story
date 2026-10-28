import json
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix


def run_evaluation():
    """Evaluate the ONNX turn detection model on the held-out test set."""
    tokenizer = AutoTokenizer.from_pretrained("./iteration_1/turn_detector_model")
    session = ort.InferenceSession("./iteration_1/turn_detector_model/model.onnx", providers=["CPUExecutionProvider"])

    texts, true_labels = [], []
    with open("iteration_1/data/test.jsonl") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            true_labels.append(r["label"])

    preds = []
    latencies = []

    for text in texts:
        t0 = time.perf_counter()
        encoded = tokenizer(
            text, max_length=64, truncation=True, padding="max_length", return_tensors="np"
        )
        ort_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        logits = session.run(None, ort_inputs)[0]
        prob_eot = np.exp(logits[0][1]) / np.sum(np.exp(logits[0]))
        latencies.append((time.perf_counter() - t0) * 1000)
        preds.append(1 if prob_eot >= 0.5 else 0)

    print("=== Model Performance on Test Set ===")
    print(classification_report(true_labels, preds, target_names=["HOLD (Incomplete)", "EOT (Complete)"]))
    print("Confusion Matrix:\n", confusion_matrix(true_labels, preds))
    print(f"Latency P50: {np.percentile(latencies, 50):.2f} ms")
    print(f"Latency P95: {np.percentile(latencies, 95):.2f} ms")
    print(f"Latency P99: {np.percentile(latencies, 99):.2f} ms")


if __name__ == "__main__":
    run_evaluation()