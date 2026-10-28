import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./iteration_1/turn_detector_model"
ONNX_PATH = "./iteration_1/turn_detector_model/model.onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

dummy_text = "Is there anything else I can help you with today"
inputs = tokenizer(dummy_text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")

torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    },
    opset_version=14,
)
print(f"ONNX model successfully exported to {ONNX_PATH}")