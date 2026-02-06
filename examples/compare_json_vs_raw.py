#!/usr/bin/env python3
"""
Compare: Structured JSON vs Raw Text
"""
import sys
sys.path.append('./src')
from rag_pipeline import ProbabilityStasisRAG

# Same data, two formats
json_data = [
  {"category": "Person/Entity Fact", "twitch_user": "Turboagentmaddi", "detail": "Currently weighs 253 pounds and is fasting to lose weight."},
  {"category": "Person/Entity Fact", "twitch_user": "Turboagentmaddi", "detail": "Is trying to reach Plax's or Helga's weight."},
  {"category": "Person/Entity Fact", "twitch_user": "Turboagentmaddi", "detail": "Is losing double chin and fat around the face."},
  {"category": "Person/Entity Fact", "twitch_user": "Plax", "detail": "Weight is estimated at 160 pounds wet."},
  {"category": "Topic: Sexuality", "twitch_user": "Plax", "detail": "Does not discriminate; he 'swings both ways' (likes both men and women)."},
  {"category": "Topic: Sexuality", "twitch_user": "Plax", "detail": "Is described as 'gayer than a rainbow after a rainstorm'."},
  {"category": "Person/Entity Fact", "subject": "Plax's Sister", "detail": "A request was put into HR to get her a seat cushion."},
  {"category": "Topic: Food Preparation", "subject": "Plax's Sister", "detail": "Is making pancakes with vegetables (scallion pancakes)."},
  {"category": "Topic: Diet/Equipment", "twitch_user": "astrolegumes", "detail": "Notes that Plax has a 5090 (GPU) but not a food scale."},
  {"category": "Topic: General Chat", "subject": "Chat", "detail": "Is described as 'toxic' and 'the biggest bully'."},
  {"category": "Topic: Relationships/Dating", "twitch_user": "sattamxSAM", "detail": "Asks Plax if his sister knows that 'we are dating'."}
]

print("="*80)
print("TEST 1: WITH JSON STRUCTURE (Category: Detail)")
print("="*80)

# Format with category prefix
json_texts = [f"{item['category']}: {item['detail']}" for item in json_data]

rag_json = ProbabilityStasisRAG(
    collection_name="test_json",
    persist_directory="./test_json_db",
    stasis_threshold=0.05,
    top_k=3
)
rag_json.add_documents(json_texts)

query = "What is Plax's weight?"
results_json = rag_json.query(query, use_cross_reference=True)

print(f"\nQuery: {query}")
print("-"*80)
for i, r in enumerate(results_json, 1):
    print(f"{i}. [Stasis: {r['stasis_score']:.3f}] {r['document']}")

# ============================================================================

print("\n" + "="*80)
print("TEST 2: WITHOUT JSON STRUCTURE (Just raw details)")
print("="*80)

# Just the raw details
raw_texts = [item['detail'] for item in json_data]

rag_raw = ProbabilityStasisRAG(
    collection_name="test_raw",
    persist_directory="./test_raw_db",
    stasis_threshold=0.05,
    top_k=3
)
rag_raw.add_documents(raw_texts)

results_raw = rag_raw.query(query, use_cross_reference=True)

print(f"\nQuery: {query}")
print("-"*80)
for i, r in enumerate(results_raw, 1):
    print(f"{i}. [Stasis: {r['stasis_score']:.3f}] {r['document']}")

# ============================================================================

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print("\nWITH category prefix:")
for i, r in enumerate(results_json, 1):
    print(f"  {i}. Score: {r['stasis_score']:.3f}")
    
print("\nWITHOUT category prefix (raw):")
for i, r in enumerate(results_raw, 1):
    print(f"  {i}. Score: {r['stasis_score']:.3f}")

print("\n✓ Which format works better?")
