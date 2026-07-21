import os
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification, 
    TrainingArguments, 
    Trainer
)
from transformers import DefaultDataCollator

# --- 1. Configuration ---
# Build an absolute path to the root of your project so it never gets lost
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw", "training_images")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "models", "custom_vit_fire")

# We start with the base Google model that only knows general shapes, not fires
BASE_MODEL = "google/vit-base-patch16-224"

def train_model():
    print("🔥 Starting Vision Transformer Training Pipeline 🔥")
    
    # --- 2. Load and Split the Dataset ---
    print(f"Loading images from {DATA_DIR}...")
    # Hugging Face automatically reads the folder names ('fire', 'no_fire') as the labels!
    dataset = load_dataset("imagefolder", data_dir=DATA_DIR)
    
    # Split the data: 80% for training, 20% for testing to see if it actually learned
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_ds = dataset["train"]
    test_ds = dataset["test"]
    
    labels = train_ds.features["label"].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}
    print(f"Detected Labels: {labels}")

    # --- 3. Preprocessing (The Image Processor) ---
    print("Initializing Image Processor...")
    processor = ViTImageProcessor.from_pretrained(BASE_MODEL)

    def transform(example_batch):
        # Convert images to RGB (fixes issues if you downloaded PNGs with transparency)
        inputs = processor([x.convert("RGB") for x in example_batch["image"]], return_tensors="pt")
        inputs["labels"] = example_batch["label"]
        return inputs

    # Apply the transformations to our datasets
    prepared_train = train_ds.with_transform(transform)
    prepared_test = test_ds.with_transform(transform)

    # --- 4. Load the Base Model ---
    print("Loading Base ViT Model...")
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True # Important: We are changing the output layer to just 2 classes
    )

    # --- 5. Define Accuracy Metric ---
    metric = evaluate.load("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    # --- 6. Set Training Arguments ---
    # WARNING: If your computer freezes, change per_device_train_batch_size from 8 to 4 or 2
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        per_device_train_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3, # 3 passes over the dataset. Increase to 5 or 10 later for better accuracy!
        learning_rate=2e-5,
        remove_unused_columns=False,
        push_to_hub=False,
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # --- 7. The Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=prepared_train,
        eval_dataset=prepared_test,
        compute_metrics=compute_metrics,
    )

    # --- 8. Execute Training ---
    print("🚀 Commencing Training Phase... (This might take a while depending on your hardware)")
    trainer.train()

    # --- 9. Save the Final Custom Model ---
    print(f"✅ Training Complete! Saving your custom brain to {MODEL_SAVE_PATH}")
    trainer.save_model(MODEL_SAVE_PATH)
    processor.save_pretrained(MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_model()