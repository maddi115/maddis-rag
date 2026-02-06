#!/usr/bin/env python3
"""
Quick command-line query tool for Probability-Stasis RAG
Usage: python3 query.py "your question here"
"""
import sys
from maddi_retrieval import ProbabilityStasisRAG

if len(sys.argv) < 2:
    print("Usage: python3 query.py \"your question\"")
    print("Example: python3 query.py \"What is Python?\"")
    sys.exit(1)

# Get query from command line
query = " ".join(sys.argv[1:])

# Initialize RAG (using existing database)
rag = ProbabilityStasisRAG(
    collection_name="agentmaddi_history",  # Your 9000-line database
    persist_directory="databases/agentmaddi_chroma_db",
    stasis_threshold=0.05,
    top_k=3
)

print(f"\nQuerying: '{query}'\n")

# Query
results = rag.query(query, use_cross_reference=True)

# Display
if results:
    for i, result in enumerate(results, 1):
        print(f"{i}. [Stasis: {result['stasis_score']:.3f}]")
        print(f"   {result['document'][:200]}...")
        print()
else:
    print("No stable results found.")
