"""
Inference script for DistilBERT sentiment analysis model
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import List, Dict, Union, Optional
import argparse
from tqdm import tqdm

from config import LABEL_MAPPING, MODEL_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentPredictor:
    """
    Production-ready sentiment predictor using fine-tuned DistilBERT
    """

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the predictor

        Args:
            model_path: Path to the saved model directory
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.label_mapping = LABEL_MAPPING
        self.max_length = MODEL_CONFIG["max_length"]

        self._load_model()
        logger.info(f"Predictor initialized with model from {model_path}")

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

            logger.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing for inference

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        import re

        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        return text

    def predict_single(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Predict sentiment for a single text

        Args:
            text: Input text

        Returns:
            Dictionary with prediction results
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        # Format results
        result = {
            'text': text,
            'processed_text': processed_text,
            'prediction': self.label_mapping[predicted_class],
            'confidence': round(confidence, 4),
            'probabilities': {
                label: round(float(prob), 4)
                for label, prob in zip(self.label_mapping.values(), probabilities[0].cpu().numpy())
            }
        }

        return result

    def predict_batch(self, texts: List[str], batch_size: int = 32,
                     show_progress: bool = True) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Predict sentiment for multiple texts

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of prediction results
        """
        results = []

        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), disable=not show_progress):
            batch_texts = texts[i:i + batch_size]

            # Preprocess batch
            processed_texts = [self.preprocess_text(text) for text in batch_texts]

            # Tokenize batch
            inputs = self.tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Make predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)

            # Format results for this batch
            for j, text in enumerate(batch_texts):
                pred_class = predicted_classes[j].item()
                probs = probabilities[j].cpu().numpy()

                result = {
                    'text': text,
                    'processed_text': processed_texts[j],
                    'prediction': self.label_mapping[pred_class],
                    'confidence': round(float(probs[pred_class]), 4),
                    'probabilities': {
                        label: round(float(prob), 4)
                        for label, prob in zip(self.label_mapping.values(), probs)
                    }
                }
                results.append(result)

        return results

    def predict_from_csv(self, csv_path: str, text_column: str = "text",
                        output_path: Optional[str] = None,
                        batch_size: int = 32) -> pd.DataFrame:
        """
        Predict sentiment for texts in a CSV file

        Args:
            csv_path: Path to input CSV file
            text_column: Name of text column
            output_path: Path to save results (optional)
            batch_size: Batch size for processing

        Returns:
            DataFrame with predictions
        """
        # Load data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} texts from {csv_path}")

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")

        # Make predictions
        texts = df[text_column].tolist()
        predictions = self.predict_batch(texts, batch_size=batch_size)

        # Create results DataFrame
        results_df = pd.DataFrame(predictions)

        # Combine with original data
        result_df = pd.concat([df, results_df.drop('text', axis=1)], axis=1)

        # Save results if path provided
        if output_path:
            result_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

        return result_df

    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get information about the loaded model"""
        return {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'num_labels': len(self.label_mapping),
            'max_length': self.max_length,
            'label_mapping': self.label_mapping
        }


def interactive_prediction(predictor: SentimentPredictor):
    """
    Interactive prediction mode

    Args:
        predictor: Initialized predictor
    """
    print("\n🤖 Interactive Sentiment Analysis")
    print("=" * 40)
    print("Enter text to analyze sentiment (or 'quit' to exit)")
    print("-" * 40)

    while True:
        text = input("\nEnter text: ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not text:
            print("❌ Please enter some text")
            continue

        # Make prediction
        result = predictor.predict_single(text)

        # Display results
        print(f"\n📝 Text: {result['text']}")
        print(f"🎯 Prediction: {result['prediction']} (confidence: {result['confidence']:.1%})")
        print("📊 Probabilities:")

        for label, prob in result['probabilities'].items():
            bar = "█" * int(prob * 20)  # Simple progress bar
            print("8")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="DistilBERT Sentiment Analysis Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--csv_file", type=str, help="CSV file to process")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Text column name in CSV")
    parser.add_argument("--output_file", type=str, help="Output file for CSV results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")

    args = parser.parse_args()

    # Initialize predictor
    predictor = SentimentPredictor(args.model_path)

    # Display model info
    model_info = predictor.get_model_info()
    print("🤖 Model Information:")
    print(f"   Path: {model_info['model_path']}")
    print(f"   Device: {model_info['device']}")
    print(f"   Labels: {list(model_info['label_mapping'].values())}")

    if args.interactive:
        # Interactive mode
        interactive_prediction(predictor)

    elif args.text:
        # Single text prediction
        result = predictor.predict_single(args.text)
        print(f"\n📝 Text: {result['text']}")
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print("📈 Probabilities:")
        for label, prob in result['probabilities'].items():
            print("8")

    elif args.csv_file:
        # CSV file processing
        results_df = predictor.predict_from_csv(
            args.csv_file,
            args.text_column,
            args.output_file,
            args.batch_size
        )
        print(f"\n✅ Processed {len(results_df)} texts")
        if args.output_file:
            print(f"💾 Results saved to {args.output_file}")

        # Show sample results
        print("\n📊 Sample Results:")
        print(results_df[['text', 'prediction', 'confidence']].head())

    else:
        print("❌ Please specify --text, --csv_file, or --interactive")
        parser.print_help()


if __name__ == "__main__":
    # For testing without command line arguments
    import sys

    if len(sys.argv) == 1:
        # No arguments provided, run interactive mode
        print("🚀 Starting Sentiment Analysis Inference")
        print("Usage: python inference.py --model_path /path/to/model [--interactive]")

        # Try to find a trained model automatically
        import glob
        model_dirs = glob.glob("models/distilbert_sentiment_*")
        if model_dirs:
            model_path = sorted(model_dirs)[-1]
            print(f"📁 Found model: {model_path}")
            predictor = SentimentPredictor(model_path)
            interactive_prediction(predictor)
        else:
            print("❌ No trained models found. Please train a model first or specify --model_path")
    else:
        main()
