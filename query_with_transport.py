#!/usr/bin/env python3
"""
Trinity + Transportation Model Integration
Runs all 3 retrieval methods then transports results into vessel-based answers
"""
import sys
from retrieval_methods import ProbabilityStasisRAG, VectorSearch
import importlib.util

# Import Gradient module
spec = importlib.util.spec_from_file_location("gradient", "retrieval_methods/newv2Gradient_Proximity_Search_Rarity_Based_Chaining.py")
gradient_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gradient_module)
GradientProximitySearch = gradient_module.GradientProximitySearch

# Import Transportation Model
from transportation_model import TransportationModel, format_output

if len(sys.argv) < 2:
    print("Usage: python3 query_with_transport.py \"your question\"")
    sys.exit(1)

query = " ".join(sys.argv[1:])

# Initialize RAG system
rag = ProbabilityStasisRAG(
    collection_name="agentmaddi_history",
    persist_directory="databases/agentmaddi_chroma_db",
    stasis_threshold=0.05,
    top_k=3
)

print(f"\n{'='*78}")
print(f"QUERYING: '{query}'")
print(f"{'='*78}\n")

# Get all documents
collection = rag.collection
all_docs = collection.get()
documents = all_docs['documents'] if all_docs['documents'] else []

# === RUN TRINITY ===
print("Running Trinity retrieval methods...\n")

# 1. GRADIENT PROXIMITY
print("🔥 Gradient Proximity Search...")
gradient = GradientProximitySearch(documents, initial_window=20, base_strength_boost=0.7)
gradient_results, anchor_word = gradient.search(query)
print(f"   Found {len(gradient_results)} results\n")

# 2. VECTOR SEARCH
print("🎯 Vector Search...")
vector_search = VectorSearch(rag.collection, rag.embedder)
vector_results = vector_search.search(query, top_k=5)
print(f"   Found {len(vector_results)} results\n")

# 3. STASIS
print("🔬 Stasis Search...")
stasis_results = rag.query(query, use_cross_reference=True)
print(f"   Found {len(stasis_results)} results\n")

print("="*78)
print("Trinity retrieval complete. Starting transportation...\n")

# === TRANSPORTATION MODEL ===
transport = TransportationModel(
    high_confidence_threshold=0.7,
    medium_confidence_threshold=0.3,
    low_confidence_threshold=0.1
)

# Transport the results
result = transport.transport(
    query=query,
    gradient_results=gradient_results,
    vector_results=vector_results,
    stasis_results=stasis_results
)

# Display the transported answer
output = format_output(result)
print(output)

# Optional: Show supporting vessels
if result['vessels']['supporting']:
    print("\n" + "="*78)
    print("SUPPORTING VESSELS:")
    print("="*78)
    for i, vessel in enumerate(result['vessels']['supporting'][:3], 1):
        print(f"\n[{i}] Confidence: {vessel['confidence']:.3f} | Source: {vessel['source_method']}")
        print(f"    Content: '{vessel['content'][:150]}...'")

# Metadata summary
print("\n" + "="*78)
print("METADATA:")
print("="*78)
print(f"Query Type: {result['query_classification']['type']}")
print(f"Total Vessels: {result['vessels']['metadata']['total_sources']}")
print(f"High Confidence: {result['vessels']['metadata']['high_confidence_count']}")
print(f"Medium Confidence: {result['vessels']['metadata']['medium_confidence_count']}")
print(f"Low Confidence: {result['vessels']['metadata']['low_confidence_count']}")
print(f"Needs LLM Polish: {result['needs_llm_polish']}")
print("="*78 + "\n")
