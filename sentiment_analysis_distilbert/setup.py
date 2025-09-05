"""
Setup and run script for DistilBERT sentiment analysis project
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent


def run_command(command, description, cwd=None):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd or PROJECT_DIR
        )
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def setup_environment():
    """Set up the complete environment"""
    print("🚀 Setting up DistilBERT Sentiment Analysis Environment")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False

    print(f"✅ Python {sys.version.split()[0]} detected")

    # Install requirements
    if (PROJECT_DIR / "requirements.txt").exists():
        success = run_command("pip install -r requirements.txt", "Installing dependencies")
        if not success:
            return False
    else:
        print("❌ requirements.txt not found")
        return False

    # Validate installation
    print("\n🧪 Validating installation...")

    required_packages = [
        "torch", "transformers", "datasets", "numpy",
        "pandas", "sklearn", "matplotlib", "tqdm"
    ]

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError:
            print(f"❌ {package} import failed")
            return False

    print("\n✅ All packages installed and validated successfully!")
    return True


def run_data_preparation():
    """Run data preparation"""
    print("\n🔧 Running Data Preparation")
    print("-" * 30)

    success = run_command("python data_preparation.py", "Preparing dataset")
    return success is not None


def run_training():
    """Run model training"""
    print("\n🏃 Running Model Training")
    print("-" * 30)

    # Check if CUDA is available
    import torch
    if torch.cuda.is_available():
        print("🎯 CUDA available - training will use GPU")
    else:
        print("💻 CUDA not available - training will use CPU")

    success = run_command("python train_distilbert.py", "Training DistilBERT model")
    return success is not None


def run_evaluation():
    """Run model evaluation"""
    print("\n📊 Running Model Evaluation")
    print("-" * 30)

    # Find the latest trained model
    import glob
    model_dirs = glob.glob("models/distilbert_sentiment_*")
    if not model_dirs:
        print("❌ No trained models found")
        return False

    latest_model = sorted(model_dirs)[-1]
    print(f"📁 Evaluating model: {latest_model}")

    # Run evaluation
    success = run_command(f"python evaluate_model.py --model_path {latest_model}",
                         "Evaluating trained model")
    return success is not None


def run_inference_demo():
    """Run inference demonstration"""
    print("\n🚀 Running Inference Demo")
    print("-" * 30)

    # Find the latest trained model
    import glob
    model_dirs = glob.glob("models/distilbert_sentiment_*")
    if not model_dirs:
        print("❌ No trained models found")
        return False

    latest_model = sorted(model_dirs)[-1]
    print(f"📁 Using model: {latest_model}")

    # Test single prediction
    test_text = "I absolutely love this product! It's amazing!"
    success = run_command(
        f"python inference.py --model_path \"{latest_model}\" --text \"{test_text}\"",
        "Testing single prediction"
    )

    if success:
        # Test CSV prediction
        if (PROJECT_DIR / "sample_data.csv").exists():
            success = run_command(
                f"python inference.py --model_path \"{latest_model}\" --csv_file sample_data.csv --output_file predictions.csv",
                "Testing CSV batch prediction"
            )

    return success is not None


def create_demo_script():
    """Create a simple demo script"""
    demo_script = '''"""
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

print("\\nSentiment Analysis Demo")
print("=" * 40)

for text in test_texts:
    result = predictor.predict_single(text)
    print(f"\\nText: {result['text']}")
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.1%})")

print("\\nDemo completed!")
'''

    with open(PROJECT_DIR / "demo.py", "w") as f:
        f.write(demo_script)

    print("📄 Created demo.py for quick testing")


def show_project_summary():
    """Show project summary and next steps"""
    print("\n🎉 PROJECT SETUP COMPLETE!")
    print("=" * 60)
    print("Your DistilBERT sentiment analysis project is ready!")
    print("=" * 60)

    print("\\n📁 Project Structure:")
    print("├── 📄 train_distilbert.py          # Main training script")
    print("├── 🔧 data_preparation.py          # Data loading and preprocessing")
    print("├── 📊 evaluate_model.py            # Model evaluation and metrics")
    print("├── 🚀 inference.py                 # Production inference")
    print("├── ⚙️ config.py                    # Configuration settings")
    print("├── 📋 requirements.txt             # Dependencies")
    print("├── 📖 README.md                    # Documentation")
    print("├── 📄 sample_data.csv              # Sample dataset")
    print("├── 📄 demo.py                      # Quick demo script")
    print("├── 📁 models/                      # Trained models")
    print("└── 📁 results/                     # Evaluation results")

    print("\\n🚀 Quick Start Commands:")
    print("1. Train model:     python train_distilbert.py")
    print("2. Evaluate model:  python evaluate_model.py")
    print("3. Run demo:        python demo.py")
    print("4. Interactive:     python inference.py --interactive")

    print("\\n📊 Sample Results:")
    print("- Model accuracy: ~85-95% (depending on dataset)")
    print("- Training time: ~5-15 minutes on GPU")
    print("- Inference speed: ~100-500 texts/second")

    print("\\n🎯 Next Steps:")
    print("1. Replace sample_data.csv with your own dataset")
    print("2. Adjust hyperparameters in config.py")
    print("3. Experiment with different model variants")
    print("4. Deploy the model for production use")


def main():
    """Main setup function"""
    print("🤖 DistilBERT Sentiment Analysis Setup")
    print("=" * 50)

    # Step 1: Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        return

    # Step 2: Create demo script
    create_demo_script()

    # Ask user what to do next
    print("\\n" + "="*50)
    print("SETUP COMPLETE! What would you like to do next?")
    print("="*50)
    print("1. Run full pipeline (data prep → train → evaluate → demo)")
    print("2. Run data preparation only")
    print("3. Run training only")
    print("4. Exit (you can run individual scripts later)")

    while True:
        try:
            choice = input("\\nEnter your choice (1-4): ").strip()

            if choice == "1":
                # Run full pipeline
                print("\\n🚀 Running Full Pipeline")
                print("=" * 30)

                if run_data_preparation():
                    if run_training():
                        if run_evaluation():
                            run_inference_demo()

                show_project_summary()
                break

            elif choice == "2":
                run_data_preparation()
                break

            elif choice == "3":
                run_training()
                break

            elif choice == "4":
                print("\\n👋 Setup complete! You can run individual scripts anytime:")
                print("   python data_preparation.py")
                print("   python train_distilbert.py")
                print("   python evaluate_model.py")
                print("   python inference.py --interactive")
                break

            else:
                print("❌ Invalid choice. Please enter 1-4.")

        except KeyboardInterrupt:
            print("\\n\\n👋 Setup interrupted. You can continue later!")
            break


if __name__ == "__main__":
    main()
