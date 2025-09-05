"""
Model evaluation and analysis for DistilBERT sentiment analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import json
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings("ignore")

from config import LABEL_MAPPING, RESULTS_DIR
from train_distilbert import SentimentTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation of sentiment analysis model
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize evaluator

        Args:
            model_path: Path to saved model
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.label_mapping = LABEL_MAPPING

        self._load_model()

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _load_model(self):
        """Load the trained model and tokenizer"""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def evaluate_on_dataset(self, dataset, batch_size: int = 32) -> Dict[str, Any]:
        """
        Evaluate model on a dataset

        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating model on dataset...")

        all_predictions = []
        all_labels = []
        all_probabilities = []

        # Create data loader
        from torch.utils.data import DataLoader
        from transformers import DataCollatorWithPadding

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=False
        )

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get model outputs
                outputs = self.model(**batch)
                logits = outputs.logits

                # Get predictions and probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)

        # Calculate metrics
        results = self._calculate_metrics(predictions, labels, probabilities)

        return results

    def _calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray,
                          probabilities: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics

        Args:
            predictions: Model predictions
            labels: True labels
            probabilities: Prediction probabilities

        Returns:
            Dictionary with all metrics
        """
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        # Per-class metrics
        class_metrics = {}
        for i, label_name in self.label_mapping.items():
            class_metrics[label_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }

        # Overall metrics
        overall_metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted)
        }

        results = {
            'overall': overall_metrics,
            'per_class': class_metrics,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'true_labels': labels.tolist(),
            'probabilities': probabilities.tolist()
        }

        return results

    def create_evaluation_report(self, results: Dict[str, Any],
                               save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive evaluation report

        Args:
            results: Evaluation results
            save_path: Path to save the report

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("SENTIMENT ANALYSIS MODEL EVALUATION REPORT")
        report.append("=" * 60)

        # Overall metrics
        report.append("\n📊 OVERALL PERFORMANCE")
        report.append("-" * 30)
        overall = results['overall']
        report.append(".4f")
        report.append(".4f")
        report.append(".4f")
        report.append(".4f")

        # Per-class metrics
        report.append("\n🎯 PER-CLASS PERFORMANCE")
        report.append("-" * 30)
        report.append("<12")
        report.append("-" * 50)

        for class_name, metrics in results['per_class'].items():
            report.append("<12")

        # Confusion matrix
        report.append("\n📈 CONFUSION MATRIX")
        report.append("-" * 30)
        cm = np.array(results['confusion_matrix'])
        report.append("True\\Pred | " + " | ".join(f"{self.label_mapping[i]:>8}" for i in range(len(self.label_mapping))))
        report.append("-" * (12 + 11 * len(self.label_mapping)))

        for i in range(len(cm)):
            row = f"{self.label_mapping[i]:>10} | " + " | ".join(f"{cm[i,j]:>8}" for j in range(len(cm[i])))
            report.append(row)

        # Classification report
        report.append("\n📋 CLASSIFICATION REPORT")
        report.append("-" * 30)
        from sklearn.metrics import classification_report as sk_report
        report.append(sk_report(
            results['true_labels'],
            results['predictions'],
            target_names=list(self.label_mapping.values())
        ))

        report_str = "\n".join(report)

        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
            logger.info(f"Report saved to {save_path}")

        return report_str

    def create_visualizations(self, results: Dict[str, Any],
                            save_dir: Optional[str] = None):
        """
        Create comprehensive visualizations

        Args:
            results: Evaluation results
            save_dir: Directory to save plots
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(10, 8))
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(self.label_mapping.values()),
                   yticklabels=list(self.label_mapping.values()),
                   cbar_kws={'label': 'Number of Samples'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Metrics Comparison
        plt.figure(figsize=(12, 6))
        metrics = ['Precision', 'Recall', 'F1-Score']
        classes = list(results['per_class'].keys())
        values = np.array([[results['per_class'][cls][metric.lower()]
                          for metric in metrics] for cls in classes])

        x = np.arange(len(metrics))
        width = 0.25

        for i, (cls, color) in enumerate(zip(classes, ['skyblue', 'lightgreen', 'lightcoral'])):
            plt.bar(x + i*width, values[i], width, label=cls, color=color, alpha=0.8)

        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
        plt.xticks(x + width, metrics)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)

        # Add value labels
        for i in range(len(classes)):
            for j in range(len(metrics)):
                plt.text(x[j] + i*width, values[i][j] + 0.01,
                        '.3f', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Overall Metrics Radar Chart
        plt.figure(figsize=(8, 8))
        overall = results['overall']

        # Metrics for radar chart
        radar_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        radar_values = [overall[metric] for metric in radar_metrics]
        radar_metrics = [metric.replace('_macro', '').replace('_weighted', '').title()
                        for metric in radar_metrics]

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        radar_values += radar_values[:1]  # Close the circle
        angles += angles[:1]

        ax = plt.subplot(111, polar=True)
        ax.fill(angles, radar_values, 'skyblue', alpha=0.3)
        ax.plot(angles, radar_values, 'skyblue', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True)

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / 'radar_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_errors(self, results: Dict[str, Any], dataset,
                      num_samples: int = 10) -> pd.DataFrame:
        """
        Analyze prediction errors

        Args:
            results: Evaluation results
            dataset: Original dataset
            num_samples: Number of error samples to analyze

        Returns:
            DataFrame with error analysis
        """
        predictions = np.array(results['predictions'])
        true_labels = np.array(results['true_labels'])

        # Find misclassifications
        errors_mask = predictions != true_labels
        error_indices = np.where(errors_mask)[0]

        if len(error_indices) == 0:
            logger.info("No prediction errors found!")
            return pd.DataFrame()

        # Get error samples
        error_samples = []
        for idx in error_indices[:num_samples]:
            sample = dataset[idx]
            error_samples.append({
                'index': idx,
                'true_label': self.label_mapping[true_labels[idx]],
                'predicted_label': self.label_mapping[predictions[idx]],
                'text': sample.get('text', 'N/A')  # May not be available in tokenized dataset
            })

        error_df = pd.DataFrame(error_samples)
        logger.info(f"Found {len(error_indices)} errors. Showing {len(error_samples)} samples.")

        return error_df

    def save_results(self, results: Dict[str, Any], save_dir: str = None):
        """
        Save all evaluation results

        Args:
            results: Evaluation results
            save_dir: Directory to save results
        """
        if not save_dir:
            save_dir = RESULTS_DIR / f"evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        with open(save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Save report
        report = self.create_evaluation_report(results, save_dir / 'evaluation_report.txt')

        # Create visualizations
        self.create_visualizations(results, save_dir)

        logger.info(f"All evaluation results saved to {save_dir}")


def main():
    """Example usage of the evaluator"""
    print("📊 Starting Model Evaluation")
    print("=" * 50)

    # Path to your trained model
    model_path = input("Enter path to trained model directory: ").strip()

    if not model_path:
        print("❌ No model path provided. Using default path...")
        # Try to find the latest model
        import glob
        model_dirs = glob.glob("models/distilbert_sentiment_*")
        if model_dirs:
            model_path = sorted(model_dirs)[-1]  # Get latest
            print(f"📁 Using latest model: {model_path}")
        else:
            print("❌ No trained models found. Please train a model first.")
            return

    # Initialize evaluator
    evaluator = ModelEvaluator(model_path)

    # Load test data (you would typically load this from your prepared dataset)
    from data_preparation import SentimentDataProcessor
    processor = SentimentDataProcessor()

    if Path("sample_data.csv").exists():
        df = processor.load_data_from_csv("sample_data.csv")
        dataset = processor.prepare_dataset(df)
        test_dataset = dataset["test"]
    else:
        print("❌ No test data found. Please run data preparation first.")
        return

    # Evaluate model
    results = evaluator.evaluate_on_dataset(test_dataset)

    # Create and display report
    report = evaluator.create_evaluation_report(results)
    print(report)

    # Create visualizations
    evaluator.create_visualizations(results, "evaluation_plots")

    # Analyze errors
    error_analysis = evaluator.analyze_errors(results, test_dataset)
    if not error_analysis.empty:
        print("\n🔍 ERROR ANALYSIS")
        print("-" * 30)
        print(error_analysis.head())

    # Save all results
    evaluator.save_results(results)

    print("✅ Evaluation completed!")
    print(f"📊 Results saved to: evaluation_plots/")


if __name__ == "__main__":
    main()
