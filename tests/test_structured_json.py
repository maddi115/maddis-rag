#!/usr/bin/env python3
"""
Test Probability-Stasis with YOUR structured JSON data
"""
import sys
import json
sys.path.append('./src')
from rag_pipeline import ProbabilityStasisRAG

# Your structured data (paste your JSON here)
data = [
  {
    "category": "Person/Entity Fact",
    "twitch_user": "Turboagentmaddi",
    "detail": "Currently weighs 253 pounds and is fasting to lose weight.",
    "source": "Turboagentmaddi",
    "source_id": 8
  },
  {
    "category": "Person/Entity Fact",
    "twitch_user": "Turboagentmaddi",
    "detail": "Is trying to reach Plax's or Helga's weight.",
    "source": "Turboagentmaddi",
    "source_id": 1
  },
  {
    "category": "Person/Entity Fact",
    "twitch_user": "Turboagentmaddi",
    "detail": "Is losing double chin and fat around the face.",
    "source": "Turboagentmaddi",
    "source_id": 7
  },
  {
    "category": "Person/Entity Fact",
    "twitch_user": "Plax",
    "detail": "Weight is estimated at 160 pounds wet.",
    "source": "cheer 1astrolegumes",
    "source_id": 16
  },
  {
    "category": "Topic: Sexuality",
    "twitch_user": "Plax",
    "detail": "Does not discriminate; he 'swings both ways' (likes both men and women).",
    "source": "Clips Leader 3work_dayy",
    "source_id": 6
  },
  {
    "category": "Topic: Sexuality",
    "twitch_user": "Plax",
    "detail": "Is described as 'gayer than a rainbow after a rainstorm'.",
    "source": "sosuke___aizen",
    "source_id": 6
  },
  {
    "category": "Person/Entity Fact",
    "subject": "Plax's Sister",
    "detail": "A request was put into HR to get her a seat cushion.",
    "source": "Clips Leader 3work_dayy",
    "source_id": 2
  },
  {
    "category": "Topic: Food Preparation",
    "subject": "Plax's Sister",
    "detail": "Is making pancakes with vegetables (scallion pancakes).",
    "source": "babysquirrellchan",
    "source_id": 20
  },
  {
    "category": "Topic: Diet/Equipment",
    "twitch_user": "astrolegumes",
    "detail": "Notes that Plax has a 5090 (GPU) but not a food scale.",
    "source": "cheer 1astrolegumes",
    "source_id": 4
  },
  {
    "category": "Topic: General Chat",
    "subject": "Chat",
    "detail": "Is described as 'toxic' and 'the biggest bully'.",
    "source": "babysquirrellchan",
    "source_id": 9
  },
  {
    "category": "Topic: Relationships/Dating",
    "twitch_user": "sattamxSAM",
    "detail": "Asks Plax if his sister knows that 'we are dating'.",
    "source": "Gifter Leader 1sattamxSAM",
    "source_id": 10
  }
]

# Convert to clean text chunks
texts = []
metadata = []

for item in data:
    # Create searchable text from the detail
    text = f"{item['category']}: {item['detail']}"
    texts.append(text)
    
    # Keep original data as metadata
    metadata.append({
        'category': item['category'],
        'user': item.get('twitch_user', item.get('subject', 'unknown')),
        'source': item['source'],
        'source_id': item['source_id']
    })

print(f"Loaded {len(texts)} structured facts")

# Initialize RAG
rag = ProbabilityStasisRAG(
    collection_name="structured_facts",
    persist_directory="./structured_chroma_db",
    stasis_threshold=0.05,  # Higher threshold now - structured data!
    top_k=3
)

# Add to RAG
rag.add_documents(texts, metadata)

# Test queries
queries = [
    "What is Plax's weight?",
    "Tell me about agentmaddi's diet",
    "What was Plax's sister doing?",
    "What did people say about Plax's sexuality?",
]

print("\n" + "="*80)
print("TESTING WITH STRUCTURED JSON DATA")
print("="*80)

for query in queries:
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print("="*80)
    
    # Raw results
    query_embedding = rag.embedder.encode(query)
    raw_results = rag.collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )
    
    print("\nWITHOUT Probability-Stasis:")
    print("-" * 80)
    for i in range(min(3, len(raw_results['documents'][0]))):
        doc = raw_results['documents'][0][i]
        dist = raw_results['distances'][0][i]
        similarity = 1 - dist
        print(f"{i+1}. [Sim: {similarity:.3f}] {doc}")
    
    # With filter
    results = rag.query(query, use_cross_reference=True)
    
    print("\nWITH Probability-Stasis Filter:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i}. [Stasis: {result['stasis_score']:.3f} | Var: {result['prob_variance']:.4f}]")
        print(f"   {result['document']}")
        print(f"   User: {result['metadata']['user']} | Category: {result['metadata']['category']}")
    
    print()

print("\n" + "="*80)
print("✓ NOW you'll see the REAL difference!")
print("="*80)
