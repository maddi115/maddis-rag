#!/usr/bin/env python3
"""
Test Probability-Stasis RAG with real Twitch chat logs
"""
import sys
sys.path.append('./src')
from rag_pipeline import ProbabilityStasisRAG

# Read your Twitch chat logs
print("Loading Twitch chat logs...")
with open('data/twitch_chat.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Clean and chunk the data (combine every 3 lines into a chunk)
chunks = []
for i in range(0, len(lines), 3):
    chunk = ' '.join(lines[i:i+3]).strip()
    if chunk:  # Only add non-empty chunks
        chunks.append(chunk)

print(f"Created {len(chunks)} chunks from {len(lines)} lines")

# Initialize RAG
rag = ProbabilityStasisRAG(
    collection_name="twitch_chat",
    persist_directory="./twitch_chroma_db",
    stasis_threshold=0.05,
    top_k=5
)

# Add chunks to RAG
rag.add_documents(
    texts=chunks,
    metadata=[{'chunk_id': i, 'source': 'twitch_chat.txt'} for i in range(len(chunks))]
)

# Test queries on your real data
test_queries = [
    "What were people talking about?",
    "Any funny moments?",
    "What did the streamer say?",
]

print("\n" + "="*80)
print("TESTING WITH YOUR REAL TWITCH CHAT DATA")
print("="*80)

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print("="*80)
    
    # Get raw results (no filter)
    query_embedding = rag.embedder.encode(query)
    raw_results = rag.collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10
    )
    
    print("\nWITHOUT Filter (Top 3 by similarity):")
    print("-" * 80)
    for i in range(min(3, len(raw_results['documents'][0]))):
        doc = raw_results['documents'][0][i]
        dist = raw_results['distances'][0][i]
        similarity = 1 - dist
        print(f"{i+1}. [Sim: {similarity:.3f}]")
        print(f"   {doc[:150]}...")
        print()
    
    # With probability-stasis filter
    results = rag.query(query, use_cross_reference=True)
    
    print("\nWITH Probability-Stasis (Top 3 stable results):")
    print("-" * 80)
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. [Stasis: {result['stasis_score']:.3f} | Var: {result['prob_variance']:.4f}]")
        print(f"   {result['document'][:150]}...")
        print()

print("\n" + "="*80)
print("✓ Real data test complete!")
print("="*80)

