"""
Sample Search Demonstrations for Semantic Search Engine

This script provides various examples of how to use the semantic search engine
with different types of queries and datasets.
"""

from semantic_search_engine import SemanticSearchEngine
import json
from typing import List, Dict


def create_sample_datasets():
    """Create different sample datasets for demonstration"""
    
    # Technology & Programming Dataset
    tech_docs = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "JavaScript is the programming language of the web, used for both frontend and backend development.",
        "Machine learning algorithms can automatically learn patterns from data without explicit programming.",
        "Deep learning neural networks have revolutionized computer vision and natural language processing.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
        "Docker containers package applications with their dependencies for consistent deployment.",
        "Kubernetes orchestrates containerized applications across distributed systems.",
        "API (Application Programming Interface) allows different software systems to communicate.",
        "Database management systems store and retrieve data efficiently using SQL queries.",
        "Version control systems like Git track changes in source code during software development.",
        "Cybersecurity protects digital systems from threats and unauthorized access.",
        "Artificial intelligence aims to create machines that can perform tasks requiring human intelligence.",
        "Blockchain technology creates immutable ledgers for secure and transparent transactions.",
        "React is a JavaScript library for building user interfaces with reusable components.",
        "TensorFlow is an open-source framework for machine learning and deep learning applications."
    ]
    
    # Science & Research Dataset
    science_docs = [
        "Quantum mechanics describes the behavior of matter and energy at the atomic scale.",
        "The theory of relativity revolutionized our understanding of space, time, and gravity.",
        "DNA contains the genetic instructions for the development of all living organisms.",
        "Climate change refers to long-term shifts in global temperature and weather patterns.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "The periodic table organizes chemical elements by their atomic structure and properties.",
        "Evolution explains how species change over time through natural selection.",
        "The Big Bang theory describes the origin and expansion of the universe.",
        "Vaccines stimulate the immune system to protect against infectious diseases.",
        "Renewable energy sources like solar and wind power reduce greenhouse gas emissions.",
        "Stem cells have the potential to develop into many different cell types.",
        "The human brain contains billions of neurons that process information and control behavior.",
        "Genetic engineering allows scientists to modify the DNA of living organisms.",
        "Ocean currents play a crucial role in regulating Earth's climate and weather patterns.",
        "Antibiotics are medications that kill or inhibit the growth of bacteria."
    ]
    
    # Business & Economics Dataset
    business_docs = [
        "Supply chain management coordinates the flow of goods from suppliers to customers.",
        "Digital marketing uses online channels to promote products and engage with customers.",
        "Financial markets facilitate the trading of stocks, bonds, and other securities.",
        "Entrepreneurship involves starting and running new business ventures.",
        "Corporate strategy defines how companies compete and create value in their markets.",
        "Human resources management focuses on recruiting, training, and retaining employees.",
        "Data analytics helps businesses make informed decisions by analyzing patterns in data.",
        "Customer relationship management systems track interactions with clients and prospects.",
        "E-commerce platforms enable businesses to sell products and services online.",
        "Project management methodologies help teams deliver projects on time and within budget.",
        "Venture capital provides funding to startups and early-stage companies.",
        "Lean manufacturing eliminates waste and improves efficiency in production processes.",
        "Brand management builds and maintains the reputation and image of products or companies.",
        "Market research analyzes consumer behavior and competitive landscapes.",
        "Corporate governance ensures companies operate ethically and transparently."
    ]
    
    return {
        "technology": tech_docs,
        "science": science_docs,
        "business": business_docs
    }


def demo_basic_semantic_search():
    """Demonstrate basic semantic search capabilities"""
    print("🔍 BASIC SEMANTIC SEARCH DEMO")
    print("=" * 60)
    
    # Create engine with general-purpose model
    engine = SemanticSearchEngine(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_type="flat"
    )
    
    # Load technology dataset
    datasets = create_sample_datasets()
    tech_docs = datasets["technology"]
    
    # Add documents with metadata
    metadata = [{"category": "technology", "doc_id": i} for i in range(len(tech_docs))]
    engine.add_documents(tech_docs, metadata=metadata)
    
    # Sample queries with explanations
    queries = [
        {
            "query": "artificial intelligence and machine learning",
            "explanation": "Looking for AI/ML related content"
        },
        {
            "query": "web development and frontend frameworks",
            "explanation": "Searching for web development topics"
        },
        {
            "query": "software deployment and containers",
            "explanation": "Finding deployment and containerization info"
        },
        {
            "query": "data storage and databases",
            "explanation": "Looking for database-related content"
        },
        {
            "query": "security and protection systems",
            "explanation": "Searching for cybersecurity topics"
        }
    ]
    
    for query_info in queries:
        print(f"\n🎯 {query_info['explanation']}")
        print(f"Query: '{query_info['query']}'")
        print("-" * 50)
        
        results = engine.search(query_info["query"], k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result.score:.4f}] {result.text[:80]}...")
    
    return engine


def demo_cross_domain_search():
    """Demonstrate search across multiple domains"""
    print("\n\n🌐 CROSS-DOMAIN SEARCH DEMO")
    print("=" * 60)
    
    # Create engine
    engine = SemanticSearchEngine(
        model_name="sentence-transformers/all-mpnet-base-v2",  # Higher quality model
        index_type="flat"
    )
    
    # Load all datasets
    datasets = create_sample_datasets()
    all_docs = []
    all_metadata = []
    
    for domain, docs in datasets.items():
        all_docs.extend(docs)
        all_metadata.extend([{"domain": domain, "doc_id": i} for i in range(len(docs))])
    
    # Add all documents
    engine.add_documents(all_docs, metadata=all_metadata)
    
    # Cross-domain queries
    queries = [
        {
            "query": "innovation and new technologies",
            "description": "Innovation across all domains"
        },
        {
            "query": "analysis and data patterns",
            "description": "Data analysis in different fields"
        },
        {
            "query": "systems and processes",
            "description": "Systems thinking across domains"
        },
        {
            "query": "energy and power",
            "description": "Energy concepts in various contexts"
        }
    ]
    
    for query_info in queries:
        print(f"\n🔍 {query_info['description']}")
        print(f"Query: '{query_info['query']}'")
        print("-" * 50)
        
        results = engine.search(query_info["query"], k=5)
        
        # Group results by domain
        domain_results = {}
        for result in results:
            domain = result.metadata["domain"]
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(result)
        
        # Display results by domain
        for domain, domain_res in domain_results.items():
            print(f"\n📂 {domain.upper()}:")
            for result in domain_res[:2]:  # Show top 2 per domain
                print(f"   [{result.score:.4f}] {result.text[:70]}...")


def demo_similarity_comparison():
    """Demonstrate how semantic similarity works vs keyword matching"""
    print("\n\n🧠 SEMANTIC vs KEYWORD COMPARISON")
    print("=" * 60)
    
    # Create simple engine
    engine = SemanticSearchEngine(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_type="flat"
    )
    
    # Sample documents
    docs = [
        "The dog is playing in the park with a ball.",
        "A canine is running around the garden with a sphere.",
        "Python programming language is excellent for data science.",
        "The snake slithered through the grass quietly.",
        "Machine learning algorithms process large datasets.",
        "AI systems can learn from data automatically.",
        "The car drove down the highway at high speed.",
        "The automobile traveled on the freeway rapidly."
    ]
    
    engine.add_documents(docs)
    
    # Test queries
    test_cases = [
        {
            "query": "dog playing",
            "expected_semantic": "Both 'dog' and 'canine' documents should match",
            "expected_keyword": "Only 'dog' document would match"
        },
        {
            "query": "artificial intelligence",
            "expected_semantic": "Should find ML and AI documents",
            "expected_keyword": "Would miss 'machine learning' documents"
        },
        {
            "query": "vehicle transportation",
            "expected_semantic": "Should find both 'car' and 'automobile'",
            "expected_keyword": "Would miss both (no exact matches)"
        }
    ]
    
    for test in test_cases:
        print(f"\n🔍 Query: '{test['query']}'")
        print(f"Semantic Expected: {test['expected_semantic']}")
        print(f"Keyword Expected: {test['expected_keyword']}")
        print("Results:")
        
        results = engine.search(test["query"], k=3)
        for i, result in enumerate(results, 1):
            if result.score > 0.3:  # Only show relevant results
                print(f"  {i}. [{result.score:.4f}] {result.text}")


def demo_threshold_filtering():
    """Demonstrate how similarity thresholds work"""
    print("\n\n🎚️  SIMILARITY THRESHOLD DEMO")
    print("=" * 60)
    
    # Create engine
    engine = SemanticSearchEngine(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_type="flat"
    )
    
    # Mixed relevance documents
    docs = [
        "Deep learning neural networks for image recognition",  # Highly relevant
        "Machine learning algorithms for data analysis",       # Relevant
        "Statistical methods for scientific research",         # Somewhat relevant
        "Database design and optimization techniques",         # Less relevant
        "Cooking recipes for Italian pasta dishes",           # Not relevant
        "Travel destinations in Southeast Asia",               # Not relevant
    ]
    
    engine.add_documents(docs)
    
    query = "artificial intelligence and deep learning"
    thresholds = [0.0, 0.3, 0.5, 0.7]
    
    print(f"Query: '{query}'")
    print("\nResults with different similarity thresholds:")
    
    for threshold in thresholds:
        print(f"\n🎯 Threshold: {threshold}")
        print("-" * 30)
        
        results = engine.search(query, k=10, threshold=threshold)
        
        if not results:
            print("   No results above threshold")
        else:
            for i, result in enumerate(results, 1):
                print(f"   {i}. [{result.score:.4f}] {result.text[:50]}...")


def demo_advanced_queries():
    """Demonstrate complex and nuanced queries"""
    print("\n\n🎯 ADVANCED QUERY EXAMPLES")
    print("=" * 60)
    
    # Create comprehensive dataset
    datasets = create_sample_datasets()
    all_docs = []
    all_metadata = []
    
    for domain, docs in datasets.items():
        all_docs.extend(docs)
        all_metadata.extend([{"domain": domain} for _ in docs])
    
    # Use advanced model
    engine = SemanticSearchEngine(
        model_name="sentence-transformers/all-mpnet-base-v2",
        index_type="flat"
    )
    
    engine.add_documents(all_docs, metadata=all_metadata)
    
    # Advanced query scenarios
    advanced_queries = [
        {
            "query": "How to automate repetitive tasks efficiently?",
            "description": "Question-based query about automation"
        },
        {
            "query": "scalable solutions for growing businesses",
            "description": "Business growth and scalability"
        },
        {
            "query": "environmental impact of technology",
            "description": "Intersection of tech and environment"
        },
        {
            "query": "learning from data without human supervision",
            "description": "Unsupervised learning concepts"
        },
        {
            "query": "protecting sensitive information from hackers",
            "description": "Cybersecurity and data protection"
        }
    ]
    
    for query_info in advanced_queries:
        print(f"\n💡 {query_info['description']}")
        print(f"Query: '{query_info['query']}'")
        print("-" * 50)
        
        results = engine.search(query_info["query"], k=3, threshold=0.4)
        
        if not results:
            print("   No highly relevant results found")
        else:
            for i, result in enumerate(results, 1):
                domain = result.metadata["domain"]
                print(f"   {i}. [{result.score:.4f}] ({domain}) {result.text[:60]}...")


def save_sample_search_results():
    """Save sample search results for analysis"""
    print("\n\n💾 SAVING SAMPLE RESULTS")
    print("=" * 60)
    
    # Create engine and add data
    engine = SemanticSearchEngine(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_type="flat"
    )
    
    datasets = create_sample_datasets()
    tech_docs = datasets["technology"]
    metadata = [{"category": "technology", "doc_id": i} for i in range(len(tech_docs))]
    engine.add_documents(tech_docs, metadata=metadata)
    
    # Sample queries and save results
    sample_queries = [
        "machine learning and AI",
        "web development frameworks",
        "cloud computing platforms",
        "database management",
        "cybersecurity protection"
    ]
    
    all_results = {}
    
    for query in sample_queries:
        results = engine.search(query, k=5)
        all_results[query] = [
            {
                "text": r.text,
                "score": float(r.score),
                "index": r.index,
                "metadata": r.metadata
            }
            for r in results
        ]
    
    # Save to file
    with open("sample_search_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("✅ Sample search results saved to 'sample_search_results.json'")
    print("📊 Contains results for 5 different queries with similarity scores")


def main():
    """Run all sample search demonstrations"""
    print("🚀 SEMANTIC SEARCH ENGINE - SAMPLE SEARCHES")
    print("=" * 80)
    print("This demo shows various ways to use semantic search effectively")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        demo_basic_semantic_search()
        demo_cross_domain_search()
        demo_similarity_comparison()
        demo_threshold_filtering()
        demo_advanced_queries()
        save_sample_search_results()
        
        print("\n\n🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("\nKey Takeaways:")
        print("1. 🧠 Semantic search understands meaning, not just keywords")
        print("2. 🎯 Similarity scores help filter relevant results")
        print("3. 🌐 Works across different domains and topics")
        print("4. 💡 Handles complex, question-based queries")
        print("5. ⚙️  Thresholds control result quality vs quantity")
        
        print("\n📝 Next Steps:")
        print("• Try your own queries with the search engine")
        print("• Experiment with different embedding models")
        print("• Add your own document collections")
        print("• Build domain-specific search applications")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure all dependencies are installed and models are accessible")


if __name__ == "__main__":
    main()
