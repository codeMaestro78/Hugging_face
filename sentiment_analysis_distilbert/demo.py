"""
Quick Demo Script for DistilBERT Sentiment Analysis
"""

from inference import SentimentPredictor
import glob

# Find the latest trained model
model_dirs = glob.glob("models/distilbert_sentiment_*")
if not model_dirs:
    print("No trained models found. Please train a model first.")
    exit(1)

model_path = sorted(model_dirs)[-1]
print(f"Using model: {model_path}")

# Initialize predictor
predictor = SentimentPredictor(model_path)

# Test predictions
test_texts = [
    "I absolutely love this product! It's amazing!",
    "This is okay, nothing special.",
    "This is terrible, I hate it so much.",
    "The customer service was excellent and very helpful.",
    "The product arrived damaged and late.",
    "It's decent for the price, but not great."
]

print("\nSentiment Analysis Demo")
print("=" * 40)

for text in test_texts:
    result = predictor.predict_single(text)
    print(f"\nText: {result['text']}")
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.1%})")

print("\nDemo completed!")
