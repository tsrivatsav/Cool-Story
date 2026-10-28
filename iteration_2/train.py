"""
Iteration 2 — Model Training (ASR-normalized data)
Fine-tunes all-MiniLM-L6-v2 on iteration_2/data/ and saves to iteration_2/turn_detector_model/
"""

import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "./iteration_2/turn_detector_model"
DATA_DIR = "./iteration_2/data"


def load_jsonl(path):
    texts, labels = [], []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["label"])
    return Dataset.from_dict({"text": texts, "label": labels})


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=1)[:, 1].numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = load_jsonl(f"{DATA_DIR}/train.jsonl")
    val_ds = load_jsonl(f"{DATA_DIR}/val.jsonl")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=64)

    train_tok = train_ds.map(tokenize_fn, batched=True)
    val_tok = val_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Iteration-2 model fine-tuning complete and saved.")


if __name__ == "__main__":
    main()