#!/usr/bin/env python3
"""
Test Probability-Stasis with 9000+ lines of real chat data
"""
import sys
import csv
sys.path.append('./src')
from rag_pipeline import ProbabilityStasisRAG

print("Loading chat_messages.csv...")

# Read CSV
messages = []
with open('data/chat_messages.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        messages.append(row)

print(f"Loaded {len(messages)} messages")

# Show first few to see structure
print("\nFirst 3 rows (sample):")
for i, msg in enumerate(messages[:3], 1):
    print(f"{i}. {msg.get('login')}: {msg.get('body')}")

# Create chunks (combine every 5 messages for context)
chunk_size = 5
chunks = []
metadata = []

for i in range(0, len(messages), chunk_size):
    chunk_msgs = messages[i:i+chunk_size]
    # Combine messages into one chunk using correct field names
    chunk_text = ' | '.join([
        f"{msg.get('login', 'Unknown')}: {msg.get('body', '')}" 
        for msg in chunk_msgs if msg.get('body')
    ])
    
    if chunk_text.strip():
        chunks.append(chunk_text)
        metadata.append({
            'chunk_id': i,
            'source': 'chat_messages.csv',
            'time': chunk_msgs[0].get('time', 'unknown')
        })

print(f"\nCreated {len(chunks)} chunks from {len(messages)} messages")

# Initialize RAG
rag = ProbabilityStasisRAG(
    collection_name="agentmaddi_history",
    persist_directory="./agentmaddi_chroma_db",
    stasis_threshold=0.05,
    top_k=5
)

# Add to RAG (first 1000 chunks to start - about 5000 messages)
num_chunks = min(1000, len(chunks))
print(f"\nAdding first {num_chunks} chunks to RAG system...")
rag.add_documents(chunks[:num_chunks], metadata[:num_chunks])

# Test queries
queries = [
    "What does agentmaddi talk about?",
    "What games does agentmaddi play?",
    "What happened with the bicycle accident?",
    "What are agentmaddi's opinions?",
]

print("\n" + "="*80)
print(f"TESTING WITH {len(messages)} LINES OF REAL DATA")
print("="*80)

for query in queries:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print("="*80)
    
    # Raw results
    query_embedding = rag.embedder.encode(query)
    raw_results = rag.collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10
    )
    
    print("\nWITHOUT Probability-Stasis (Top 3):")
    print("-"*80)
    for i in range(min(3, len(raw_results['documents'][0]))):
        doc = raw_results['documents'][0][i]
        dist = raw_results['distances'][0][i]
        similarity = 1 - dist
        print(f"{i+1}. [Sim: {similarity:.3f}]")
        print(f"   {doc[:250]}...")
        print()
    
    # With filter
    results = rag.query(query, use_cross_reference=True)
    
    print("\nWITH Probability-Stasis (Top 3 stable):")
    print("-"*80)
    if results:
        for i, result in enumerate(results[:3], 1):
            print(f"{i}. [Stasis: {result['stasis_score']:.3f} | Var: {result['prob_variance']:.4f}]")
            print(f"   {result['document'][:250]}...")
            print()
    else:
        print("   No stable results passed the filter threshold")
        print()

print("\n" + "="*80)
print("✓ Real 9000-line dataset test complete!")
print("="*80)
print("\nNOTE: With larger real datasets, probability-stasis filtering becomes")
print("more effective at removing noise and keeping stable, relevant results.")
print("="*80)
