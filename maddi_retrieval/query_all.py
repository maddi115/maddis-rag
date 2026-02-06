#!/usr/bin/env python3
"""
Multi-Method Query Tool - The Trinity
"""
import sys
from maddi_retrieval import ProbabilityStasisRAG, VectorSearch
import importlib.util
spec = importlib.util.spec_from_file_location("gradient", "maddi_retrieval/newv2Gradient_Proximity_Search_Rarity_Based_Chaining.py")
gradient_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gradient_module)
GradientProximitySearch = gradient_module.GradientProximitySearch

if len(sys.argv) < 2:
    print("Usage: python3 query_all.py \"your question\"")
    sys.exit(1)

query = " ".join(sys.argv[1:])

# Initialize
rag = ProbabilityStasisRAG(
    collection_name="agentmaddi_history",
    persist_directory="databases/agentmaddi_chroma_db",
    stasis_threshold=0.05,
    top_k=3
)

print(f"\n{'='*78}")
print(f"QUERYING: '{query}'")
print(f"{'='*78}\n")

collection = rag.collection
all_docs = collection.get()
documents = all_docs['documents'] if all_docs['documents'] else []

# === GRADIENT PROXIMITY (FULL BEAUTIFUL OUTPUT) ===
print("━" * 78)
print("🔥 GRADIENT PROXIMITY (Advanced Rarity-Based Chaining)")
print("━" * 78)
gradient = GradientProximitySearch(documents, initial_window=20, base_strength_boost=0.7)
gradient_results, anchor_word = gradient.search(query)

# Use the BEAUTIFUL formatting from your advanced version!
formatted_gradient = gradient.format_results(gradient_results, query, anchor_word)
print(formatted_gradient)
print("\n")

# === VECTOR SEARCH ===
print("━" * 78)
print("🎯 VECTOR SEARCH (Pure Semantic Similarity)")
print("━" * 78)
vector_search = VectorSearch(rag.collection, rag.embedder)
vector_results = vector_search.search(query, top_k=5)

if vector_results:
    for i, result in enumerate(vector_results, 1):
        print(f"[{i}] Similarity: {result['score']:.3f}")
        print(f"{result['document'][:150]}...")
        print()
else:
    print("No results found.\n")

# === STASIS ===
print("━" * 78)
print("🔬 STASIS (Your Original Probability Algorithm)")
print("━" * 78)
stasis_results = rag.query(query, use_cross_reference=True)

if stasis_results:
    for i, result in enumerate(stasis_results, 1):
        print(f"[{i}] Stasis: {result['stasis_score']:.3f}")
        print(f"{result['document'][:150]}...")
        print()
else:
    print("No stable results found.\n")

print("━" * 78)
