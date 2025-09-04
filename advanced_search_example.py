"""
Advanced Example: Building a Semantic Search Engine for Research Papers

This example demonstrates how to build a more sophisticated semantic search engine
that can handle research papers, scientific documents, or any domain-specific corpus.
"""

from semantic_search_engine import SemanticSearchEngine, SearchResult
from typing import List, Dict
import json
from pathlib import Path


def load_sample_papers() -> List[Dict]:
    """
    Load sample research paper abstracts for demonstration
    In practice, you would load from your actual data source
    """
    papers = [
        {
            "title": "Attention Is All You Need",
            "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            "authors": ["Vaswani et al."],
            "year": 2017,
            "venue": "NeurIPS",
            "domain": "Natural Language Processing"
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.",
            "authors": ["Devlin et al."],
            "year": 2018,
            "venue": "NAACL",
            "domain": "Natural Language Processing"
        },
        {
            "title": "ResNet: Deep Residual Learning for Image Recognition",
            "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.",
            "authors": ["He et al."],
            "year": 2016,
            "venue": "CVPR",
            "domain": "Computer Vision"
        },
        {
            "title": "Generative Adversarial Networks",
            "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
            "authors": ["Goodfellow et al."],
            "year": 2014,
            "venue": "NeurIPS",
            "domain": "Machine Learning"
        },
        {
            "title": "Adam: A Method for Stochastic Optimization",
            "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters.",
            "authors": ["Kingma & Ba"],
            "year": 2014,
            "venue": "ICLR",
            "domain": "Optimization"
        },
        {
            "title": "You Only Look Once: Unified, Real-Time Object Detection",
            "abstract": "We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation.",
            "authors": ["Redmon et al."],
            "year": 2016,
            "venue": "CVPR",
            "domain": "Computer Vision"
        },
        {
            "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
            "abstract": "Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use, making it difficult to deal with overfitting by combining the predictions of many different large neural nets at test time. Dropout is a technique for addressing this problem.",
            "authors": ["Srivastava et al."],
            "year": 2014,
            "venue": "JMLR",
            "domain": "Machine Learning"
        },
        {
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions.",
            "authors": ["Brown et al."],
            "year": 2020,
            "venue": "NeurIPS",
            "domain": "Natural Language Processing"
        }
    ]
    return papers


def build_research_paper_search_engine():
    """
    Build a semantic search engine for research papers with enhanced features
    """
    print("🔬 Building Research Paper Semantic Search Engine")
    print("=" * 60)
    
    # Load sample papers
    papers = load_sample_papers()
    
    # Extract texts and metadata
    texts = []
    metadata = []
    
    for paper in papers:
        # Combine title and abstract for better search
        combined_text = f"{paper['title']}. {paper['abstract']}"
        texts.append(combined_text)
        
        # Store metadata
        metadata.append({
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "venue": paper["venue"],
            "domain": paper["domain"]
        })
    
    # Create search engine with scientific text optimized model
    print("🤖 Initializing search engine with SciBERT model...")
    engine = SemanticSearchEngine(
        model_name="allenai/scibert_scivocab_uncased",  # Scientific domain model
        index_type="flat",  # Use exact search for small dataset
        pooling_strategy="mean"
    )
    
    # Add documents
    print("📚 Adding research papers to index...")
    engine.add_documents(texts, metadata=metadata)
    
    return engine


def demonstrate_advanced_search(engine: SemanticSearchEngine):
    """
    Demonstrate advanced search capabilities
    """
    print("\n🔍 Advanced Search Demonstrations")
    print("=" * 60)
    
    # Example queries
    queries = [
        {
            "query": "attention mechanisms in neural networks",
            "description": "Looking for papers about attention mechanisms"
        },
        {
            "query": "image recognition and computer vision",
            "description": "Searching for computer vision papers"
        },
        {
            "query": "optimization algorithms for machine learning",
            "description": "Finding optimization-related research"
        },
        {
            "query": "generative models and adversarial training",
            "description": "Looking for generative AI papers"
        },
        {
            "query": "preventing overfitting in deep networks",
            "description": "Finding regularization techniques"
        }
    ]
    
    for query_info in queries:
        print(f"\n📝 {query_info['description']}")
        print(f"Query: '{query_info['query']}'")
        print("-" * 50)
        
        results = engine.search(query_info["query"], k=3, threshold=0.3)
        
        if not results:
            print("No results found above threshold.")
            continue
        
        for i, result in enumerate(results, 1):
            metadata = result.metadata
            print(f"{i}. [{result.score:.4f}] {metadata['title']}")
            print(f"   Authors: {', '.join(metadata['authors'])}")
            print(f"   Year: {metadata['year']} | Venue: {metadata['venue']} | Domain: {metadata['domain']}")
            print()


def domain_specific_search(engine: SemanticSearchEngine):
    """
    Demonstrate domain-specific filtering and search
    """
    print("\n🎯 Domain-Specific Search")
    print("=" * 60)
    
    query = "neural network architectures"
    print(f"Query: '{query}'")
    print("Filtering by domain...")
    
    # Get all results
    all_results = engine.search(query, k=10)
    
    # Group by domain
    domains = {}
    for result in all_results:
        domain = result.metadata["domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(result)
    
    # Display results by domain
    for domain, results in domains.items():
        print(f"\n📂 {domain} ({len(results)} results):")
        for i, result in enumerate(results[:2], 1):  # Show top 2 per domain
            metadata = result.metadata
            print(f"  {i}. [{result.score:.4f}] {metadata['title']}")


def save_and_load_demo(engine: SemanticSearchEngine):
    """
    Demonstrate saving and loading the search engine
    """
    print("\n💾 Save and Load Demonstration")
    print("=" * 60)
    
    # Save the engine
    save_path = "./research_paper_engine"
    print(f"Saving engine to {save_path}...")
    engine.save(save_path)
    
    # Load the engine
    print("Loading engine from disk...")
    loaded_engine = SemanticSearchEngine.load(save_path)
    
    # Test the loaded engine
    print("Testing loaded engine...")
    results = loaded_engine.search("transformer models", k=2)
    
    print("✅ Successfully loaded and tested engine!")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result.score:.4f}] {result.metadata['title']}")


def main():
    """
    Main function demonstrating the complete workflow
    """
    try:
        # Build the search engine
        engine = build_research_paper_search_engine()
        
        # Run demonstrations
        demonstrate_advanced_search(engine)
        domain_specific_search(engine)
        save_and_load_demo(engine)
        
        print("\n🎉 Semantic Search Engine Demo Complete!")
        print("\nNext steps:")
        print("1. Replace sample data with your actual corpus")
        print("2. Experiment with different embedding models")
        print("3. Try different FAISS index types for larger datasets")
        print("4. Add more sophisticated metadata filtering")
        print("5. Implement semantic clustering and visualization")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have installed all requirements:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
