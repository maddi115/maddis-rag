#!/usr/bin/env python3
"""
Quick test of Probability-Stasis RAG
"""
import sys
sys.path.append('./src')

from rag_pipeline import ProbabilityStasisRAG

# Initialize RAG
print("Initializing Probability-Stasis RAG...")
rag = ProbabilityStasisRAG(
    collection_name="test_docs",
    persist_directory="./test_chroma_db",
    stasis_threshold=0.15,
    top_k=3
)

# Add some test documents
test_docs = [
    "Python is a high-level programming language with simple syntax.",
    "Machine learning models require large datasets for training.",
    "Probability-stasis filtering improves RAG system accuracy.",
    "ChromaDB is a vector database for semantic search.",
    "Neural networks are used in deep learning applications.",
    "Filtering unstable results reduces hallucination in AI.",
]

rag.add_documents(test_docs)

# Query it
query = "How does filtering improve AI systems?"
print(f"\nQuerying: {query}\n")

results = rag.query(query, use_cross_reference=True)
rag.display_results(results, query)

print("\n✓ Test complete!")
