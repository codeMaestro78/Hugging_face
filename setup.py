"""
Setup and Installation Script for Semantic Search Engine

This script handles the complete setup process including:
- Virtual environment creation
- Dependency installation
- Basic validation
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def setup_environment():
    """Set up the complete environment"""
    print("🚀 Setting up Semantic Search Engine Environment")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  No virtual environment detected")
        print("Recommendation: Create a virtual environment first")
        print("Run: python -m venv venv && venv\\Scripts\\activate")
    
    # Install requirements
    if Path("requirements.txt").exists():
        run_command("pip install -r requirements.txt", "Installing dependencies")
    else:
        print("❌ requirements.txt not found")
        return False
    
    # Validate installation
    print("\n🧪 Validating installation...")
    
    required_packages = [
        "torch",
        "transformers", 
        "faiss",
        "numpy",
        "tqdm"
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


def run_basic_test():
    """Run a basic test of the semantic search engine"""
    print("\n🧪 Running basic functionality test...")
    
    test_code = '''
from semantic_search_engine import SemanticSearchEngine

# Create a simple test
engine = SemanticSearchEngine(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Add test documents
docs = [
    "Machine learning is amazing",
    "Deep learning uses neural networks", 
    "AI will change the world"
]

engine.add_documents(docs)

# Test search
results = engine.search("artificial intelligence", k=2)
print(f"Found {len(results)} results")
for r in results:
    print(f"Score: {r.score:.4f}, Text: {r.text}")

print("✅ Basic test completed successfully!")
'''
    
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


def main():
    """Main setup function"""
    if setup_environment():
        print("\n🎯 Setup complete! You can now:")
        print("1. Run the basic example: python semantic_search_engine.py")
        print("2. Run the advanced example: python advanced_search_example.py")
        print("3. Import the SemanticSearchEngine class in your own code")
        
        # Optionally run basic test
        response = input("\nWould you like to run a basic test? (y/N): ")
        if response.lower() in ['y', 'yes']:
            run_basic_test()
    else:
        print("\n❌ Setup failed. Please check the errors above.")


if __name__ == "__main__":
    main()
