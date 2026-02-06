"""
Pure Vector Search - Semantic similarity using embeddings
"""
from typing import List, Dict, Any

class VectorSearch:
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.encode(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        formatted_results = []
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                formatted_results.append({
                    'score': similarity,
                    'document': results['documents'][0][i],
                    'doc_idx': i,
                    'distance': distance
                })
        
        return formatted_results
