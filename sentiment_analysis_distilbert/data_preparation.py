"""
Data preparation and preprocessing for DistilBERT sentiment analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import re
from typing import Tuple, List, Dict, Any
import logging
from pathlib import Path
import json

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import torch

from config import (
    MODEL_CONFIG, DATA_CONFIG, LABEL_MAPPING,
    REVERSE_LABEL_MAPPING, PROJECT_ROOT
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentDataProcessor:
    """
    Handles data loading, preprocessing, and preparation for sentiment analysis
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_CONFIG["model_name"]
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.max_length = MODEL_CONFIG["max_length"]

        # Initialize tokenizer
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load the tokenizer for the specified model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Loaded tokenizer for {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def load_data_from_csv(self, file_path: str, text_column: str = "text",
                          label_column: str = "sentiment") -> pd.DataFrame:
        """
        Load data from CSV file

        Args:
            file_path: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} samples from {file_path}")

            # Validate columns exist
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in dataset")
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in dataset")

            # Basic data validation
            df = df.dropna(subset=[text_column, label_column])
            df = df[df[text_column].str.len() > 0]

            logger.info(f"After cleaning: {len(df)} samples")
            return df

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def create_sample_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Create sample sentiment analysis data for demonstration

        Args:
            num_samples: Number of samples to generate

        Returns:
            DataFrame with sample data
        """
        np.random.seed(DATA_CONFIG["random_seed"])

        # Sample texts for different sentiments
        positive_texts = [
            "I absolutely love this product! It's amazing and works perfectly.",
            "This is the best experience I've ever had. Highly recommended!",
            "Outstanding quality and excellent customer service. Five stars!",
            "I'm so happy with this purchase. It exceeded my expectations.",
            "Fantastic work! This is exactly what I was looking for.",
            "Brilliant design and superb functionality. Couldn't be happier!",
            "This is incredible! The quality is top-notch and very reliable.",
            "Wonderful experience! I would definitely buy this again.",
            "Perfect in every way. I'm thoroughly impressed.",
            "Exceptional value for money. Highly satisfied with the purchase."
        ]

        neutral_texts = [
            "This product is okay. It does what it's supposed to do.",
            "It's neither good nor bad. Just average performance.",
            "The item arrived on time and works as described.",
            "Standard quality for the price. Nothing special.",
            "It's functional and meets basic requirements.",
            "Decent product with average features.",
            "Works fine for everyday use. No complaints.",
            "Basic functionality without any extras.",
            "It's acceptable but not impressive.",
            "Standard item that does the job adequately."
        ]

        negative_texts = [
            "This is terrible! I'm very disappointed with the quality.",
            "Worst purchase ever. Complete waste of money.",
            "Poor quality and terrible customer service. Avoid!",
            "I'm extremely unhappy with this product. It doesn't work properly.",
            "This is awful! Don't buy this if you value your money.",
            "Horrible experience. The product broke after one use.",
            "Very dissatisfied. This is not worth the price.",
            "This product is junk. Stay away from it.",
            "Complete disappointment. Nothing works as advertised.",
            "Terrible quality and poor design. Regret buying this."
        ]

        # Generate balanced dataset
        samples_per_class = num_samples // 3
        texts = []
        labels = []

        for _ in range(samples_per_class):
            texts.extend([
                np.random.choice(positive_texts),
                np.random.choice(neutral_texts),
                np.random.choice(negative_texts)
            ])
            labels.extend(["positive", "neutral", "negative"])

        # Add remaining samples if any
        remaining = num_samples % 3
        for _ in range(remaining):
            texts.append(np.random.choice(positive_texts))
            labels.append("positive")

        df = pd.DataFrame({
            "text": texts,
            "sentiment": labels
        })

        logger.info(f"Created sample dataset with {len(df)} samples")
        return df

    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove mentions and hashtags (optional)
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)

        return text

    def encode_labels(self, labels: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Encode string labels to integers

        Args:
            labels: List of string labels

        Returns:
            Tuple of encoded labels and label mapping
        """
        encoded_labels = self.label_encoder.fit_transform(labels)
        label_mapping = dict(zip(self.label_encoder.classes_,
                                self.label_encoder.transform(self.label_encoder.classes_)))

        logger.info(f"Label mapping: {label_mapping}")
        return encoded_labels, label_mapping

    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize function for Hugging Face datasets

        Args:
            examples: Batch of examples

        Returns:
            Tokenized examples
        """
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    def prepare_dataset(self, df: pd.DataFrame, text_column: str = "text",
                       label_column: str = "sentiment") -> DatasetDict:
        """
        Prepare dataset for training

        Args:
            df: Input DataFrame
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            DatasetDict with train/val/test splits
        """
        # Preprocess text
        logger.info("Preprocessing text data...")
        df[text_column] = df[text_column].apply(self.preprocess_text)

        # Encode labels
        labels, label_mapping = self.encode_labels(df[label_column].tolist())
        df["label"] = labels

        # Save label mapping
        with open(PROJECT_ROOT / "label_mapping.json", "w") as f:
            # Convert numpy types to regular Python types for JSON serialization
            json_label_mapping = {str(k): int(v) for k, v in label_mapping.items()}
            json.dump(json_label_mapping, f, indent=2)

        # Create Hugging Face dataset
        from datasets import ClassLabel
        class_label = ClassLabel(names=["negative", "neutral", "positive"])
        dataset = Dataset.from_pandas(df[[text_column, "label"]])
        dataset = dataset.cast_column("label", class_label)

        # Split dataset
        train_val_test = dataset.train_test_split(
            test_size=DATA_CONFIG["val_split"] + DATA_CONFIG["test_split"],
            stratify_by_column="label",
            seed=DATA_CONFIG["random_seed"]
        )

        val_test = train_val_test["test"].train_test_split(
            test_size=DATA_CONFIG["test_split"] / (DATA_CONFIG["val_split"] + DATA_CONFIG["test_split"]),
            stratify_by_column="label",
            seed=DATA_CONFIG["random_seed"]
        )

        dataset_dict = DatasetDict({
            "train": train_val_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"]
        })

        # Tokenize datasets
        logger.info("Tokenizing datasets...")
        tokenized_datasets = dataset_dict.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        # Set format for PyTorch
        tokenized_datasets.set_format("torch")

        logger.info("Dataset preparation complete!")
        logger.info(f"Train: {len(tokenized_datasets['train'])} samples")
        logger.info(f"Validation: {len(tokenized_datasets['validation'])} samples")
        logger.info(f"Test: {len(tokenized_datasets['test'])} samples")

        return tokenized_datasets

    def get_data_collator(self):
        """
        Get data collator for training

        Returns:
            Data collator function
        """
        from transformers import DataCollatorWithPadding
        return DataCollatorWithPadding(tokenizer=self.tokenizer)


def main():
    """Example usage of the data processor"""
    # Create sample data
    processor = SentimentDataProcessor()
    df = processor.create_sample_data(1000)

    # Save sample data
    df.to_csv(PROJECT_ROOT / "sample_data.csv", index=False)
    logger.info("Saved sample data to sample_data.csv")

    # Prepare dataset
    dataset = processor.prepare_dataset(df)

    # Show some examples
    print("\nSample training examples:")
    for i in range(3):
        example = dataset["train"][i]
        print(f"Text: {df.iloc[i]['text'][:50]}...")
        print(f"Label: {LABEL_MAPPING[example['label'].item()]}")
        print("-" * 50)


if __name__ == "__main__":
    main()
