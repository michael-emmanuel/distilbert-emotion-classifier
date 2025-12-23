from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch

# 1. Load Niche Dataset (Emotion classification: 6 classes)
dataset = load_dataset("dair-ai/emotion")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 2. Define Metrics (Accuracy & F1)
metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": acc, "f1": f1}

# 3. Model Architecture (Transfer Learning with DistilBERT)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=1, # Increased for better results
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    # Save predictions for the confusion matrix in model_eval.py
    predictions = trainer.predict(tokenized_datasets["test"])
    np.save("results/preds.npy", np.argmax(predictions.predictions, axis=-1))
    np.save("results/labels.npy", predictions.label_ids)
