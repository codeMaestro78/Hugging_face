"""
Fine-tune DistilBERT for sentiment analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from config import (
    MODEL_CONFIG, TRAINING_CONFIG, DEVICE_CONFIG,
    LOGGING_CONFIG, MODEL_SAVE_CONFIG, RESULTS_DIR,
    MODELS_DIR, LABEL_MAPPING
)
from data_preparation import SentimentDataProcessor

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["log_level"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up device
device = torch.device(
    f"cuda:{DEVICE_CONFIG['gpu_id']}" if torch.cuda.is_available() and DEVICE_CONFIG["use_gpu"]
    else "cpu"
)
logger.info(f"Using device: {device}")


class SentimentTrainer:
    """
    Handles the complete fine-tuning process for DistilBERT sentiment analysis
    """

    def __init__(self, model_name: str = None, output_dir: str = None):
        self.model_name = model_name or MODEL_CONFIG["model_name"]
        self.output_dir = Path(output_dir) if output_dir else MODELS_DIR / f"distilbert_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tokenizer = None
        self.trainer = None

        # Initialize model and tokenizer
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer"""
        logger.info(f"Loading {self.model_name}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=MODEL_CONFIG["num_labels"]
            )

            logger.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics

        Args:
            eval_pred: Evaluation predictions

        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def get_training_arguments(self) -> TrainingArguments:
        """Get training arguments for the Trainer"""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=TRAINING_CONFIG["num_epochs"],
            per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
            per_device_eval_batch_size=TRAINING_CONFIG["batch_size"],
            learning_rate=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
            warmup_steps=TRAINING_CONFIG["warmup_steps"],
            logging_steps=TRAINING_CONFIG["logging_steps"],
            eval_steps=TRAINING_CONFIG["eval_steps"],
            save_steps=TRAINING_CONFIG["save_steps"],
            save_total_limit=3,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
            max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
            fp16=DEVICE_CONFIG["mixed_precision"] and torch.cuda.is_available(),
            dataloader_pin_memory=False,
            report_to=[],  # Disable wandb/tensorboard for simplicity
        )

    def train(self, train_dataset, eval_dataset, data_collator=None):
        """
        Train the model

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator for batching
        """
        logger.info("Starting training...")

        # Set up trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.get_training_arguments(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"],
                early_stopping_threshold=TRAINING_CONFIG["early_stopping_threshold"]
            )]
        )

        # Train the model
        train_result = self.trainer.train()

        # Save the model
        self.save_model()

        logger.info("Training completed!")
        return train_result

    def evaluate(self, test_dataset):
        """
        Evaluate the model on test dataset

        Args:
            test_dataset: Test dataset

        Returns:
            Evaluation results
        """
        logger.info("Evaluating model on test dataset...")

        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids

        # Calculate detailed metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None
        )

        # Confusion matrix
        cm = confusion_matrix(labels, preds)

        results = {
            'accuracy': accuracy,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'predictions': preds.tolist(),
            'true_labels': labels.tolist()
        }

        # Save results
        self.save_evaluation_results(results)

        # Print classification report
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(labels, preds, target_names=list(LABEL_MAPPING.values())))

        return results

    def save_model(self):
        """Save the trained model and tokenizer"""
        logger.info(f"Saving model to {self.output_dir}")

        if MODEL_SAVE_CONFIG["save_best_model"]:
            self.trainer.save_model(str(self.output_dir / "best_model"))

        if MODEL_SAVE_CONFIG["save_last_model"]:
            self.model.save_pretrained(str(self.output_dir / "final_model"))
            self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))

        # Save training config
        config = {
            "model_name": self.model_name,
            "training_config": TRAINING_CONFIG,
            "model_config": MODEL_CONFIG,
            "timestamp": datetime.now().isoformat()
        }

        with open(self.output_dir / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Model saved successfully")

    def save_evaluation_results(self, results):
        """Save evaluation results and create visualizations"""
        results_dir = self.output_dir / "evaluation"
        results_dir.mkdir(exist_ok=True)

        # Save results as JSON
        with open(results_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)

        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(LABEL_MAPPING.values()),
                   yticklabels=list(LABEL_MAPPING.values()))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create metrics bar plot
        plt.figure(figsize=(10, 6))
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        values = [
            results['accuracy'],
            np.mean(results['precision']),
            np.mean(results['recall']),
            np.mean(results['f1'])
        ]

        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(results_dir / "metrics.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Evaluation results saved to {results_dir}")

    def predict(self, texts: list) -> dict:
        """
        Make predictions on new texts

        Args:
            texts: List of texts to classify

        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.trainer:
            raise ValueError("Model not trained yet. Call train() first.")

        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MODEL_CONFIG["max_length"],
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Make predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(predictions, dim=-1)

        # Convert to readable format
        results = []
        for i, text in enumerate(texts):
            pred_class = predicted_classes[i].item()
            probs = predictions[i].cpu().numpy()

            result = {
                'text': text,
                'prediction': LABEL_MAPPING[pred_class],
                'confidence': float(probs[pred_class]),
                'probabilities': {
                    label: float(prob) for label, prob in zip(LABEL_MAPPING.values(), probs)
                }
            }
            results.append(result)

        return {'predictions': results}


def main():
    """Main training function"""
    print("🚀 Starting DistilBERT Sentiment Analysis Fine-tuning")
    print("=" * 60)

    # Initialize data processor
    processor = SentimentDataProcessor()

    # Create or load dataset
    if (Path("sample_data.csv")).exists():
        print("📂 Loading existing dataset...")
        df = processor.load_data_from_csv("sample_data.csv")
    else:
        print("🎯 Creating sample dataset...")
        df = processor.create_sample_data(2000)
        df.to_csv("sample_data.csv", index=False)

    # Prepare dataset
    print("🔧 Preparing dataset...")
    dataset = processor.prepare_dataset(df)

    # Initialize trainer
    print("🤖 Initializing trainer...")
    trainer = SentimentTrainer()

    # Train the model
    print("🏃 Starting training...")
    train_result = trainer.train(
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=processor.get_data_collator()
    )

    # Evaluate on test set
    print("📊 Evaluating model...")
    test_results = trainer.evaluate(dataset["test"])

    # Test predictions
    print("🧪 Testing predictions...")
    test_texts = [
        "I absolutely love this product! It's amazing!",
        "This is okay, nothing special.",
        "This is terrible, I hate it so much."
    ]

    predictions = trainer.predict(test_texts)

    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    for pred in predictions['predictions']:
        print(f"Text: {pred['text'][:50]}...")
        print(f"Prediction: {pred['prediction']} (confidence: {pred['confidence']:.3f})")
        print("-" * 30)

    print("✅ Training completed successfully!" )   
    print(f"📁 Model saved to: {trainer.output_dir}")
    print(f"📊 Results saved to: {trainer.output_dir}/evaluation")


if __name__ == "__main__":
    main()
