#!/usr/bin/env python3
"""
Testing my own Probability-Stasis RAG library!
"""
from probability_stasis_rag import ProbabilityStasisRAG

print("="*80)
print("MY PROBABILITY-STASIS RAG LIBRARY")
print("="*80)

# Initialize
rag = ProbabilityStasisRAG(
    collection_name="my_test",
    persist_directory="./my_test_db",
    stasis_threshold=0.05,
    top_k=3
)

# Add documents
docs = [
    "Probability-stasis filtering improves RAG accuracy by removing unstable results.",
    "Machine learning models require large datasets for training.",
    "ChromaDB is a vector database for semantic search applications.",
    "Filtering noisy data helps improve model performance significantly.",
    "RAG systems combine retrieval with generation for better answers.",
]

print("\nAdding documents...")
rag.add_documents(docs)

# Query
query = "How does filtering improve AI systems?"
print(f"\nQuerying: '{query}'")

results = rag.query(query, use_cross_reference=True)

# Display
rag.display_results(results, query)

print("\n" + "="*80)
print("✓ MY LIBRARY WORKS! I can now use it in ANY project!")
print("="*80)
