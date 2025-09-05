"""
Configuration settings for DistilBERT sentiment analysis fine-tuning
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "num_labels": 3,  # negative, neutral, positive
    "max_length": 512,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 100,
    "gradient_accumulation_steps": 2,
    "max_grad_norm": 1.0,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.01,
}

# Data configuration
DATA_CONFIG = {
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42,
    "stratify": True,  # Maintain class distribution
}

# Label mapping for sentiment analysis
LABEL_MAPPING = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Reverse mapping
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# Evaluation metrics
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix"
]

# Device configuration
DEVICE_CONFIG = {
    "use_gpu": True,
    "gpu_id": 0,
    "mixed_precision": True,  # Use FP16 if available
}

# Logging configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "save_logs": True,
    "log_file": LOGS_DIR / "training.log",
}

# Model saving configuration
MODEL_SAVE_CONFIG = {
    "save_best_model": True,
    "save_last_model": True,
    "save_optimizer": False,
    "save_scheduler": False,
}

# Inference configuration
INFERENCE_CONFIG = {
    "batch_size": 32,
    "return_probabilities": True,
    "return_attention": False,
}
