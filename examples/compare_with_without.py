#!/usr/bin/env python3
"""
Compare RAG results WITH vs WITHOUT Probability-Stasis filtering
"""
import sys
sys.path.append('./src')

from rag_pipeline import ProbabilityStasisRAG

print("="*80)
print("COMPARISON: WITH vs WITHOUT Probability-Stasis")
print("="*80)

# Initialize RAG
rag = ProbabilityStasisRAG(
    collection_name="test_docs",
    persist_directory="./test_chroma_db",
    stasis_threshold=0.15,
    top_k=3
)

query = "How does filtering improve AI systems?"

print(f"\nQuery: {query}\n")

# ============================================================================
# WITHOUT Probability-Stasis (just raw similarity)
# ============================================================================
print("\n" + "="*80)
print("WITHOUT PROBABILITY-STASIS (Raw Similarity Only)")
print("="*80)

# Get raw results from ChromaDB
query_embedding = rag.embedder.encode(query)
raw_results = rag.collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=6
)

print(f"\nShowing ALL {len(raw_results['documents'][0])} results (no filtering):\n")

for i, (doc, dist) in enumerate(zip(raw_results['documents'][0], raw_results['distances'][0]), 1):
    similarity = 1 - dist
    print(f"Rank {i}:")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Document:   {doc[:200]}...")
    print()

# ============================================================================
# WITH Probability-Stasis
# ============================================================================
print("\n" + "="*80)
print("WITH PROBABILITY-STASIS (Cross-Verified + Filtered)")
print("="*80)

results = rag.query(query, use_cross_reference=True)

print(f"\nShowing {len(results)} stable results (filtered from 6):\n")

for i, result in enumerate(results, 1):
    print(f"Rank {i}:")
    print(f"  StasisScore: {result['stasis_score']:.4f} ⭐")
    print(f"  Similarity:  {result['similarity']:.4f}")
    print(f"  Variance:    {result['prob_variance']:.4f} (lower = more stable)")
    print(f"  Document:    {result['document'][:200]}...")
    print()

# ============================================================================
# Summary
# ============================================================================
print("="*80)
print("WHAT'S THE DIFFERENCE?")
print("="*80)
print("""
WITHOUT Probability-Stasis:
  ❌ Shows ALL results, including noisy/irrelevant ones
  ❌ Based on similarity score alone (can be misleading)
  ❌ No cross-verification
  ❌ Unstable results included

WITH Probability-Stasis:
  ✅ Filters out unstable/contradictory results
  ✅ Cross-checks: similarity + keywords + length
  ✅ Only keeps results where ALL signals agree
  ✅ Lower variance = more reliable
  ✅ Reduces hallucination risk for LLM
""")
print("="*80)
