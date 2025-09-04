"""
Sample Search Demonstrations for Semantic Search Engine

This script provides various examples of how to use the semantic search engine
with different types of queries and datasets.
"""

from semantic_search_engine import SemanticSearchEngine
import json
from typing import List, Dict

def create_sample_datasets():
    """Create expanded sample datasets for demonstration"""

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
        "TensorFlow is an open-source framework for machine learning and deep learning applications.",
        "Edge computing brings computation and data storage closer to the location where it is needed.",
        "Internet of Things (IoT) connects physical devices to the internet for remote monitoring and control.",
        "Quantum computing uses qubits to perform computations far faster than classical computers.",
        "Rust is a systems programming language focused on safety and performance.",
        "DevOps practices integrate software development and IT operations for faster delivery."
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
        "Antibiotics are medications that kill or inhibit the growth of bacteria.",
        "Neuroscience studies the structure and function of the nervous system.",
        "Astrophysics explores the physics of celestial objects and phenomena.",
        "Microbiology examines microscopic organisms such as bacteria and viruses.",
        "Synthetic biology combines biology and engineering to design new biological systems.",
        "Geology studies the Earth's physical structure, composition, and processes."
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
        "Corporate governance ensures companies operate ethically and transparently.",
        "Behavioral economics studies how psychological factors influence economic decisions.",
        "Financial planning helps individuals and organizations manage their assets and expenses.",
        "International trade involves the exchange of goods and services between countries.",
        "Innovation management fosters creativity and the development of new products and services.",
        "Risk management identifies, assesses, and mitigates potential business risks."
    ]

    # Additional Domain: Health & Medicine
    health_docs = [
        "Nutrition plays a vital role in maintaining overall health and well-being.",
        "Cardiovascular diseases affect the heart and blood vessels, often linked to lifestyle.",
        "Mental health includes emotional, psychological, and social well-being.",
        "Exercise improves physical fitness, strengthens muscles, and enhances mood.",
        "Immunology studies the immune system and how it protects the body from diseases.",
        "Pharmacology focuses on the effects and uses of drugs in medicine.",
        "Epidemiology analyzes patterns, causes, and effects of health and disease conditions.",
        "Medical imaging techniques like MRI and CT scans help diagnose health issues.",
        "Genomics studies the complete set of DNA in an organism and its functions.",
        "Telemedicine allows healthcare delivery remotely using digital communication technology."
    ]

    # Additional Domain: Arts & Literature
    arts_docs = [
        "Shakespeare is considered one of the greatest playwrights in English literature.",
        "Renaissance art emphasized humanism, perspective, and classical themes.",
        "Modernist literature broke traditional forms and explored new narrative techniques.",
        "Music theory analyzes the structure, harmony, and composition of music.",
        "Photography captures moments through light and composition.",
        "Cinema combines storytelling, visuals, and sound to create immersive experiences.",
        "Sculpture is a three-dimensional art form using materials like stone, metal, or clay.",
        "Poetry expresses emotions and ideas through rhythmic and aesthetic language.",
        "Graphic design communicates ideas visually using typography, imagery, and color.",
        "Architecture combines art and engineering to design functional and aesthetic structures."
    ]

    return {
        "technology": tech_docs,
        "science": science_docs,
        "business": business_docs,
        "health": health_docs,
        "arts": arts_docs
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
    },
    {
        "query": "cloud computing platforms and services",
        "explanation": "Exploring cloud infrastructure and solutions"
    },
    {
        "query": "deep learning and neural networks",
        "explanation": "Finding content on advanced machine learning techniques"
    },
    {
        "query": "version control with Git and GitHub",
        "explanation": "Searching for resources on source code management"
    },
    {
        "query": "APIs for software integration",
        "explanation": "Looking for information on connecting software systems"
    },
    {
        "query": "blockchain technology and cryptocurrency",
        "explanation": "Exploring blockchain use cases and crypto applications"
    },
    {
        "query": "React and frontend libraries",
        "explanation": "Learning about building interactive web interfaces"
    },
    {
        "query": "TensorFlow and machine learning frameworks",
        "explanation": "Searching for ML frameworks and tools"
    },
    {
        "query": "Kubernetes orchestration of containers",
        "explanation": "Understanding container orchestration concepts"
    },
    {
        "query": "edge computing and IoT devices",
        "explanation": "Exploring computing close to data sources"
    },
    {
        "query": "quantum computing principles and applications",
        "explanation": "Learning about emerging computational paradigms"
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
            "description": "Exploring innovation across all domains"
        },
        {
            "query": "analysis and data patterns",
            "description": "Finding data analysis insights in various fields"
        },
        {
            "query": "systems and processes",
            "description": "Understanding systems thinking across domains"
        },
        {
            "query": "energy and power",
            "description": "Exploring energy concepts in multiple contexts"
        },
        {
            "query": "healthcare and medical advancements",
            "description": "Discovering breakthroughs in health and medicine"
        },
        {
            "query": "artistic expression and creativity",
            "description": "Finding topics on arts, literature, and creativity"
        },
        {
            "query": "financial planning and investment strategies",
            "description": "Insights on finance, economics, and investments"
        },
        {
            "query": "climate change and environmental sustainability",
            "description": "Searching for information on environmental impact and sustainability"
        },
        {
            "query": "artificial intelligence in business",
            "description": "Applications of AI in corporate and business contexts"
        },
        {
            "query": "emerging trends in technology",
            "description": "Cutting-edge technology developments across domains"
        },
        {
            "query": "nutrition, exercise, and wellness",
            "description": "Health, fitness, and lifestyle topics"
        },
        {
            "query": "digital media and design",
            "description": "Topics in creative arts, media, and graphic design"
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


def demo_similarity_comparison_enhanced():
    """Demonstrate semantic similarity vs keyword matching with more documents and queries"""
    print("\n\n🧠 SEMANTIC vs KEYWORD COMPARISON - ENHANCED")
    print("=" * 70)
    
    # Create engine
    engine = SemanticSearchEngine(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_type="flat"
    )
    
    # Sample documents across multiple domains
    docs = [
        # Animals
        "The dog is playing in the park with a ball.",
        "A canine is running around the garden with a sphere.",
        "The snake slithered through the grass quietly.",
        "Cats are agile and love climbing trees.",
        "A wolf howled at the moon in the forest.",
        
        # Vehicles
        "The car drove down the highway at high speed.",
        "The automobile traveled on the freeway rapidly.",
        "Bicycles are eco-friendly and promote fitness.",
        "Trains connect cities efficiently and quickly.",
        
        # Technology & AI
        "Python programming language is excellent for data science.",
        "Machine learning algorithms process large datasets.",
        "AI systems can learn from data automatically.",
        "Deep learning neural networks improve computer vision.",
        "Blockchain technology ensures secure transactions.",
        
        # Health & Science
        "Exercise improves physical fitness and strengthens muscles.",
        "Nutrition is essential for maintaining good health.",
        "Vaccines protect the body from infectious diseases.",
        "Climate change impacts global weather patterns.",
        "Photosynthesis converts sunlight into chemical energy in plants."
    ]
    
    engine.add_documents(docs)
    
    # Test queries with expected semantic vs keyword matching
    test_cases = [
        {
            "query": "dog and canine activity",
            "expected_semantic": "Both 'dog' and 'canine' documents should match",
            "expected_keyword": "Only 'dog' document would match"
        },
        {
            "query": "machine learning and AI",
            "expected_semantic": "Should find ML and AI related documents",
            "expected_keyword": "Might miss 'machine learning' or 'deep learning' documents"
        },
        {
            "query": "automobile or car transport",
            "expected_semantic": "Should match both 'car' and 'automobile' documents",
            "expected_keyword": "Would miss 'automobile' if only 'car' keyword searched"
        },
        {
            "query": "exercise and fitness",
            "expected_semantic": "Should find documents on health and exercise",
            "expected_keyword": "Only exact matches like 'exercise' would appear"
        },
        {
            "query": "photosynthesis and plant energy",
            "expected_semantic": "Should find plant biology and photosynthesis docs",
            "expected_keyword": "May fail if keywords do not exactly match"
        },
        {
            "query": "secure online transactions",
            "expected_semantic": "Should find blockchain and cybersecurity related docs",
            "expected_keyword": "Would fail if 'blockchain' keyword not used"
        },
        {
            "query": "bicycle and eco transport",
            "expected_semantic": "Should match documents about bicycles and green transport",
            "expected_keyword": "May not match unless exact word 'bicycle' used"
        }
    ]
    
    # Execute search and compare
    for test in test_cases:
        print(f"\n🔍 Query: '{test['query']}'")
        print(f"Semantic Expected: {test['expected_semantic']}")
        print(f"Keyword Expected: {test['expected_keyword']}")
        print("Results:")
        
        results = engine.search(test["query"], k=5, threshold=0.3)
        if not results:
            print("  No results above threshold")
        else:
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result.score:.4f}] {result.text}")
    
    print("\n✅ Semantic similarity allows finding meaningfully related documents even without exact keyword matches.")



def demo_threshold_filtering_enhanced():
    """Demonstrate how similarity thresholds work with multiple queries and domains"""
    print("\n\n🎚️ SIMILARITY THRESHOLD DEMO - ENHANCED")
    print("=" * 70)
    
    # Create engine
    engine = SemanticSearchEngine(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_type="flat"
    )
    
    # Mixed relevance documents across domains
    docs = [
        # AI / Tech
        "Deep learning neural networks for image recognition",  # Highly relevant
        "Machine learning algorithms for data analysis",        # Relevant
        "AI systems learn patterns automatically from data",    # Relevant
        "TensorFlow framework for building AI applications",    # Somewhat relevant
        "Database design and optimization techniques",          # Less relevant
        "Cloud computing platforms and services",               # Less relevant
        
        # Health / Life
        "Exercise improves physical fitness and overall health",  # Somewhat relevant
        "Nutrition tips for maintaining a healthy lifestyle",    # Less relevant
        "Vaccines help prevent infectious diseases",             # Less relevant
        "Cooking recipes for Italian pasta dishes",              # Not relevant
        
        # Travel / Lifestyle
        "Travel destinations in Southeast Asia",                 # Not relevant
        "Bicycling is a sustainable form of transportation",     # Less relevant
        "Photography tips for landscape photography",            # Not relevant
        "Meditation techniques for mental well-being"            # Less relevant
    ]
    
    engine.add_documents(docs)
    
    # Queries to test threshold effects
    queries = [
        "artificial intelligence and deep learning",
        "health and fitness tips",
        "sustainable transportation methods"
    ]
    
    thresholds = [0.0, 0.3, 0.5, 0.7]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("Results with different similarity thresholds:")
        
        for threshold in thresholds:
            print(f"\n🎯 Threshold: {threshold}")
            print("-" * 40)
            
            results = engine.search(query, k=10, threshold=threshold)
            
            if not results:
                print("   No results above threshold")
            else:
                for i, result in enumerate(results, 1):
                    # Show a bit more text for clarity
                    snippet = result.text if len(result.text) <= 80 else result.text[:77] + "..."
                    print(f"   {i}. [{result.score:.4f}] {snippet}")
    
    print("\n✅ Observation:")
    print("• Lower thresholds return more results including less relevant ones.")
    print("• Higher thresholds return fewer results, focusing on highly relevant documents.")


def demo_advanced_queries_enhanced():
    """Demonstrate complex and nuanced queries with enhanced coverage"""
    print("\n\n🎯 ADVANCED QUERY EXAMPLES - ENHANCED")
    print("=" * 70)
    
    # Create comprehensive dataset
    datasets = create_sample_datasets()
    all_docs = []
    all_metadata = []
    
    for domain, docs in datasets.items():
        all_docs.extend(docs)
        all_metadata.extend([{"domain": domain} for _ in docs])
    
    # Use high-quality embedding model
    engine = SemanticSearchEngine(
        model_name="sentence-transformers/all-mpnet-base-v2",
        index_type="flat"
    )
    
    engine.add_documents(all_docs, metadata=all_metadata)
    
    # Advanced query scenarios with explanations
    advanced_queries = [
        {
            "query": "How to automate repetitive tasks efficiently?",
            "description": "Automation and workflow optimization"
        },
        {
            "query": "scalable solutions for growing businesses",
            "description": "Business growth, scalability, and efficiency"
        },
        {
            "query": "environmental impact of technology",
            "description": "Sustainable technology and climate effects"
        },
        {
            "query": "learning from data without human supervision",
            "description": "Unsupervised learning and AI"
        },
        {
            "query": "protecting sensitive information from hackers",
            "description": "Cybersecurity, data protection, and encryption"
        },
        {
            "query": "latest trends in renewable energy and climate change",
            "description": "Energy sector and environmental impact"
        },
        {
            "query": "creative ways to express emotions in art",
            "description": "Arts, literature, and emotional expression"
        },
        {
            "query": "emerging programming languages for system-level development",
            "description": "Tech innovation and programming trends"
        },
        {
            "query": "methods to improve mental health and well-being",
            "description": "Healthcare and psychological wellness"
        },
        {
            "query": "strategies for effective project management",
            "description": "Business productivity and project planning"
        }
    ]
    
    for query_info in advanced_queries:
        print(f"\n💡 {query_info['description']}")
        print(f"Query: '{query_info['query']}'")
        print("-" * 60)
        
        results = engine.search(query_info["query"], k=5, threshold=0.4)  # Top 5 results
        
        if not results:
            print("   No highly relevant results found")
        else:
            for i, result in enumerate(results, 1):
                domain = result.metadata.get("domain", "N/A")
                snippet = result.text if len(result.text) <= 80 else result.text[:77] + "..."
                print(f"   {i}. [{result.score:.4f}] ({domain}) {snippet}")


def save_sample_search_results_enhanced():
    """Save enhanced sample search results across domains for analysis"""
    print("\n\n💾 SAVING ENHANCED SAMPLE RESULTS")
    print("=" * 70)
    
    # Create engine and add all domain data
    engine = SemanticSearchEngine(
        model_name="sentence-transformers/all-mpnet-base-v2",
        index_type="flat"
    )
    
    datasets = create_sample_datasets()
    all_docs = []
    all_metadata = []
    
    for domain, docs in datasets.items():
        all_docs.extend(docs)
        all_metadata.extend([{"domain": domain, "doc_id": i} for i, _ in enumerate(docs)])
    
    engine.add_documents(all_docs, metadata=all_metadata)
    
    # Enhanced sample queries with explanations
    sample_queries = [
        {"query": "machine learning and AI", "description": "Artificial intelligence and ML concepts"},
        {"query": "web development frameworks", "description": "Frontend and backend technologies"},
        {"query": "cloud computing platforms", "description": "Cloud infrastructure and services"},
        {"query": "database management", "description": "Data storage and retrieval techniques"},
        {"query": "cybersecurity protection", "description": "Security measures for digital systems"},
        {"query": "renewable energy and climate change", "description": "Sustainability and energy"},
        {"query": "mental health and well-being", "description": "Healthcare and psychology"},
        {"query": "creative art and expression", "description": "Arts, literature, and creativity"},
        {"query": "business growth and scalability", "description": "Business development strategies"},
        {"query": "emerging programming languages", "description": "New technologies in software development"}
    ]
    
    all_results = {}
    
    for query_info in sample_queries:
        query = query_info["query"]
        print(f"\nProcessing query: '{query}' ({query_info['description']})")
        
        results = engine.search(query, k=5)
        all_results[query] = [
            {
                "text": r.text,
                "score": float(r.score),
                "index": r.index,
                "domain": r.metadata.get("domain", r.metadata.get("category", "N/A"))
            }
            for r in results
        ]
    
    # Save results to JSON file
    with open("enhanced_sample_search_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Enhanced sample search results saved to 'enhanced_sample_search_results.json'")
    print("📊 Contains results for 10 different queries with domain info and similarity scores")



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
        demo_similarity_comparison_enhanced()
        demo_threshold_filtering_enhanced()
        demo_advanced_queries_enhanced()
        save_sample_search_results_enhanced()
        
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
