# 🧠 Semantic Search Engine with FAISS and Hugging Face

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗-transformers-yellow.svg)](https://huggingface.co/transformers/)
[![FAISS](https://img.shields.io/badge/Meta-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready semantic search engine** that combines the power of Hugging Face transformers with FAISS for ultra-fast similarity search. Unlike traditional keyword search, this engine understands **meaning and context**, finding relevant documents even when they don't contain exact keyword matches.

## ✨ **Key Features**

- 🧠 **Semantic Understanding**: Finds "car" when you search for "automobile"
- ⚡ **Lightning Fast**: FAISS-powered similarity search with multiple index types
- 🎯 **Multi-Domain Support**: Works across technology, science, business, and more
- 🔧 **Production Ready**: Persistence, batch processing, and metadata support
- 🤖 **Multiple Models**: Support for sentence-transformers, SciBERT, and custom models
- 📊 **Smart Filtering**: Configurable similarity thresholds and domain filtering
- 💾 **Persistent Storage**: Save and load engines with all data intact

## 🏗️ **Project Structure**

```
HuggingFace/
├── 📄 semantic_search_engine.py     # Core engine implementation
├── 🔬 advanced_search_example.py    # Research paper search demo
├── 🎯 sample_searches.py            # Comprehensive search examples
├── ⚙️ setup.py                      # Environment setup script
├── 📋 requirements.txt              # Python dependencies
├── 📊 enhanced_sample_search_results.json  # Sample search outputs
├── 🗂️ search_engine_data/          # Basic engine saved data
├── 🔬 research_paper_engine/       # Research paper engine data
├── 🚫 .gitignore                   # Git ignore rules
└── 📖 README.md                    # This file
```

## � **Quick Start**

### Prerequisites
- Python 3.11+ 
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/codeMaestro78/Hugging_face.git
cd Hugging_face
```

2. **Set up virtual environment**
```powershell
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```powershell
pip install -r requirements.txt
```

### 🎯 **Running Examples**

#### 1. Basic Semantic Search Demo
```powershell
python semantic_search_engine.py
```
**What it does**: Demonstrates basic semantic search with technology documents

#### 2. Advanced Research Paper Search
```powershell
python advanced_search_example.py
```
**What it does**: Shows sophisticated research paper search with SciBERT model and metadata filtering

#### 3. Comprehensive Sample Searches
```powershell
python sample_searches.py
```
**What it does**: Runs multiple search scenarios including cross-domain search, similarity comparisons, and threshold filtering

## � **Why Semantic Search?**

| Traditional Keyword Search | Semantic Search |
|---------------------------|-----------------|
| 🔍 Query: "car" | 🔍 Query: "car" |
| ❌ Misses: "automobile", "vehicle" | ✅ Finds: "automobile", "vehicle", "sedan" |
| 📝 Exact word matching only | 🧠 Understands meaning and context |
| 🎯 "Apple company" = "Apple fruit" | 🎯 Distinguishes concepts correctly |

### **Real Example from Our Demo:**
```python
# Query: "artificial intelligence"
# Traditional search: Only finds documents with "artificial intelligence"
# Our semantic search finds:
# ✅ "AI systems can learn from data automatically" (score: 0.62)
# ✅ "Machine learning algorithms process large datasets" (score: 0.34)
```

## 🛠️ **Core API Usage**

### Basic Usage
```python
from semantic_search_engine import SemanticSearchEngine

# Create engine
engine = SemanticSearchEngine(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast model
    index_type="flat"  # Exact search
)

# Add your documents
documents = [
    "Your first document text here",
    "Your second document text here",
    # ... more documents
]

engine.add_documents(documents)

# Search
results = engine.search("your search query", k=5)
for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Text: {result.text}")
```

### Advanced Usage with Metadata and Filtering
```python
# Add documents with rich metadata
texts = [
    "Machine learning transforms healthcare diagnostics",
    "Quantum computing breakthrough in cryptography"
]
metadata = [
    {"domain": "healthcare", "year": 2024, "confidence": "high"},
    {"domain": "quantum", "year": 2024, "confidence": "medium"}
]

engine.add_documents(texts, metadata=metadata)

# Search with similarity threshold
results = engine.search("AI in medical field", k=5, threshold=0.4)
for result in results:
    print(f"[{result.score:.3f}] {result.metadata['domain']}: {result.text}")
```

### Cross-Domain Search (From sample_searches.py)
```python
# Search across technology, science, and business domains
results = engine.search("innovation and data analysis", k=10)

# Group results by domain
domain_results = {}
for result in results:
    domain = result.metadata["domain"]
    if domain not in domain_results:
        domain_results[domain] = []
    domain_results[domain].append(result)
```

## 🎯 **Available Embedding Models**

### 🚀 **Fast Models** (Good for prototyping & real-time apps)
| Model | Dimensions | Speed | Use Case |
|-------|------------|-------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | General purpose, fast |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | ⚡⚡ | Better quality, still fast |

### 🎯 **High-Quality Models** (Best accuracy)
| Model | Dimensions | Speed | Use Case |
|-------|------------|-------|----------|
| `sentence-transformers/all-mpnet-base-v2` | 768 | ⚡ | Best general-purpose model |
| `sentence-transformers/all-roberta-large-v1` | 1024 | 🐌 | Highest quality |

### 🔬 **Domain-Specific Models** (Used in our examples)
| Model | Domain | Use Case |
|-------|--------|----------|
| `allenai/scibert_scivocab_uncased` | Scientific | Research papers (advanced_search_example.py) |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | Biomedical | Medical literature |
| `sentence-transformers/msmarco-distilbert-base-v4` | Web search | Information retrieval |

## 🏗️ **FAISS Index Types**

Choose the right index for your dataset size:

| Index Type | Dataset Size | Search Speed | Memory Usage | Accuracy |
|------------|--------------|--------------|--------------|----------|
| `flat` | < 100K docs | ⚡⚡⚡ | High | 100% (Exact) |
| `ivf` | 100K - 1M docs | ⚡⚡ | Medium | ~99% (Approx) |
| `hnsw` | > 1M docs | ⚡ | Low | ~95% (Approx) |

```python
# Small dataset - exact search
engine = SemanticSearchEngine(index_type="flat")

# Medium dataset - fast approximate search  
engine = SemanticSearchEngine(index_type="ivf")

# Large dataset - very fast approximate search
engine = SemanticSearchEngine(index_type="hnsw")
```

## � **Sample Search Results** 

Our `enhanced_sample_search_results.json` contains real search outputs:

```json
{
  "machine learning and AI": [
    {
      "text": "Artificial intelligence aims to create machines...",
      "score": 0.5419,
      "domain": "technology"
    },
    {
      "text": "Machine learning algorithms can automatically...",
      "score": 0.5334,
      "domain": "technology"
    }
  ]
}
```

**Try these queries** (from our sample_searches.py):
- 🤖 "artificial intelligence and machine learning"
- 🌐 "web development and frontend frameworks" 
- 🔒 "security and protection systems"
- 🔬 "innovation and new technologies"
- 📊 "analysis and data patterns"

## �💾 **Persistence & Data Management**

### Save Engine
```python
engine.save("./my_search_engine")
```

### Load Engine
```python
engine = SemanticSearchEngine.load("./my_search_engine")
```

### Save & Load Engine
```python
# Save complete engine (FAISS index + texts + metadata + config)
engine.save("./my_search_engine")

# Load saved engine
engine = SemanticSearchEngine.load("./my_search_engine")
```

### Data Directories in Project
- **`search_engine_data/`**: Basic engine demo data
- **`research_paper_engine/`**: SciBERT research paper search data
- **Contains**: FAISS indices, original texts, metadata, and model config

## 🎯 **Advanced Features Demonstrated**

### 1. **Cross-Domain Search** (sample_searches.py)
Search across technology, science, and business domains simultaneously:
```python
# Returns results grouped by domain with metadata
results = engine.search("innovation and data analysis")
```

### 2. **Similarity Threshold Filtering** 
Control result quality vs quantity:
```python
# High precision: only very relevant results
results = engine.search("AI", threshold=0.7)

# High recall: more results, lower relevance
results = engine.search("AI", threshold=0.3)
```

### 3. **Research Paper Search** (advanced_search_example.py)
Specialized search for academic papers using SciBERT:
```python
# Uses domain-specific model for better scientific text understanding
engine = SemanticSearchEngine(model_name="allenai/scibert_scivocab_uncased")
```

### 4. **Semantic vs Keyword Comparison**
See the difference between semantic and traditional search:
```python
# Query: "dog playing"
# Semantic finds: "dog" AND "canine" documents
# Keyword finds: only "dog" documents
```

## 🔧 **Performance Optimization**

### 🚀 **Speed Optimizations**
1. **Batch Processing**: Add documents in batches (`batch_size=32-128`)
2. **GPU Support**: Install `faiss-gpu` for CUDA acceleration
3. **Model Choice**: Use MiniLM for speed, MPNet for quality
4. **Index Type**: Match index to dataset size (see table above)

### 💾 **Memory Management**
```python
# For large datasets, use approximate indices
engine = SemanticSearchEngine(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model
    index_type="hnsw"  # Memory-efficient index
)
```

### ⚡ **Real Performance Example**
From our demos:
- **Basic search**: ~50-60 queries/second
- **Cross-domain search**: 45 documents in 2.51 it/s
- **Research papers**: 8 papers processed in 1.07s

## 🗂️ **File Descriptions**

| File | Purpose | Key Features |
|------|---------|--------------|
| `semantic_search_engine.py` | Core engine | HuggingFaceEmbedder, SemanticSearchEngine classes |
| `advanced_search_example.py` | Research demo | SciBERT model, research papers, domain filtering |
| `sample_searches.py` | Comprehensive examples | Cross-domain, thresholds, semantic vs keyword |
| `enhanced_sample_search_results.json` | Sample outputs | Real search results with scores and metadata |
| `setup.py` | Environment setup | Dependency installation and validation |

## 🧪 **Example Use Cases**

### 🔬 **Academic Research** 
```python
# Search scientific papers (advanced_search_example.py)
results = engine.search("attention mechanisms in neural networks")
# Returns papers about Transformers, BERT, attention models
```

### 💼 **Business Intelligence**
```python
# Search business documents (sample_searches.py) 
results = engine.search("scalable solutions for growing businesses")
# Finds CRM systems, supply chain, management strategies
```

### 🤖 **Technical Documentation**
```python
# Search code documentation
results = engine.search("container deployment and orchestration")  
# Finds Docker, Kubernetes, deployment guides
```

### � **Multi-Domain Knowledge Base**
```python
# Search across multiple domains simultaneously
results = engine.search("analysis and data patterns")
# Returns relevant content from tech, science, business domains
```

## 🎓 **Learning Path**

1. **Start**: Run `python semantic_search_engine.py` (basic demo)
2. **Explore**: Run `python sample_searches.py` (comprehensive examples)
3. **Advanced**: Run `python advanced_search_example.py` (research papers)
4. **Build**: Use the API to create your own search applications

## � **Dependencies**

Core packages (see `requirements.txt`):
```
torch>=1.9.0                    # PyTorch for neural networks
transformers>=4.20.0            # Hugging Face transformers
faiss-cpu>=1.7.2               # Facebook AI Similarity Search
sentence-transformers>=2.2.0    # Sentence embedding models
numpy>=1.21.0                  # Numerical computing
tqdm>=4.62.0                   # Progress bars
```

## 🤝 **Contributing**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## � **Acknowledgments**

- **Hugging Face** for transformer models and tokenizers
- **Facebook AI Research** for FAISS similarity search
- **Sentence Transformers** for pre-trained embedding models
- **Allen Institute** for SciBERT scientific domain model

## 📞 **Support & Questions**

- 📧 **Issues**: Open a GitHub issue for bugs or feature requests
- 📖 **Documentation**: Check the code comments and docstrings
- 🔍 **Examples**: All usage patterns are demonstrated in the sample files

---

**Happy Searching!** 🔍✨ 

*Built with ❤️ using Hugging Face Transformers and FAISS*
