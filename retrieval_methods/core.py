"""
Complete RAG Pipeline with Probability-Stasis Filtering
Integrates ChromaDB + Sentence Transformers + Probability-Stasis Filter
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import os
from pathlib import Path


class ProbabilityStasisRAG:
    """
    RAG system with probability-stasis filtering for stable, reliable results.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        stasis_threshold: float = 0.5,
        top_k: int = 5
    ):
        """
        Initialize RAG pipeline with probability-stasis filtering.
        
        Args:
            collection_name: Name for ChromaDB collection
            embedding_model: Sentence-transformers model name
            persist_directory: Where to store ChromaDB data
            stasis_threshold: Minimum StasisScore to keep results
            top_k: Maximum number of chunks to return
        """
        self.collection_name = collection_name
        self.stasis_threshold = stasis_threshold
        self.top_k = top_k
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        print(f"Initializing ChromaDB at: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✓ RAG system initialized with {self.collection.count()} documents")
    
    def add_documents(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add documents to the RAG system.
        
        Args:
            texts: List of document texts/chunks
            metadata: Optional metadata for each document
        """
        if not texts:
            return
        
        print(f"Adding {len(texts)} documents...")
        
        # Generate embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Prepare IDs and metadata
        ids = [f"doc_{i}" for i in range(len(texts))]
        if metadata is None:
            metadata = [{'index': i} for i in range(len(texts))]
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata,
            ids=ids
        )
        
        print(f"✓ Added {len(texts)} documents (Total: {self.collection.count()})")
    
    def _calculate_keyword_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Calculate simple keyword overlap scores for cross-referencing.
        """
        query_words = set(query.lower().split())
        scores = []
        
        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / len(query_words) if query_words else 0.0
            scores.append(score)
        
        return scores
    
    def _calculate_length_scores(self, documents: List[str]) -> List[float]:
        """
        Calculate length-based relevance scores (prefer medium-length chunks).
        """
        lengths = [len(doc.split()) for doc in documents]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths) if len(lengths) > 1 else 1.0
        
        scores = []
        for length in lengths:
            # Score based on distance from mean (prefer average length)
            distance = abs(length - mean_length)
            score = max(0, 1 - (distance / (2 * std_length)))
            scores.append(score)
        
        return scores
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Softmax normalization for probability distribution."""
        scores_array = np.array(scores)
        exp_scores = np.exp(scores_array - np.max(scores_array))
        return (exp_scores / np.sum(exp_scores)).tolist()
    
    def _calculate_stasis_score(self, probabilities: List[float]) -> float:
        """
        Calculate probability-stasis score.
        
        Formula: StasisScore = mean(p_i) / (1 + variance(p_i))
        High mean + low variance = stable, reliable result
        """
        if not probabilities or len(probabilities) < 2:
            return 0.0
        
        p_array = np.array(probabilities)
        mean_p = np.mean(p_array)
        var_p = np.var(p_array)
        
        return mean_p / (1 + var_p)
    
    def query(
        self,
        query_text: str,
        n_results: int = 20,
        use_cross_reference: bool = True
    ) -> List[Dict]:
        """
        Query the RAG system with probability-stasis filtering.
        
        Args:
            query_text: User query
            n_results: Number of initial candidates to retrieve
            use_cross_reference: Enable cross-reference scoring
        
        Returns:
            List of filtered documents with stasis scores
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query_text)
        
        # Retrieve from ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, self.collection.count())
        )
        
        if not results['documents'][0]:
            print("No results found")
            return []
        
        documents = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        # Convert distances to similarity scores (1 - cosine_distance)
        similarity_scores = [1 - d for d in distances]
        
        # Apply probability-stasis filtering
        filtered_results = []
        
        for i, doc in enumerate(documents):
            # Collect probability measures
            prob_measures = [self._normalize_scores(similarity_scores)[i]]
            
            if use_cross_reference:
                # Keyword overlap score
                keyword_scores = self._calculate_keyword_scores(query_text, documents)
                prob_measures.append(self._normalize_scores(keyword_scores)[i])
                
                # Length-based relevance
                length_scores = self._calculate_length_scores(documents)
                prob_measures.append(self._normalize_scores(length_scores)[i])
            
            # Calculate stasis score
            stasis_score = self._calculate_stasis_score(prob_measures)
            
            if stasis_score >= self.stasis_threshold:
                filtered_results.append({
                    'document': doc,
                    'metadata': metadatas[i],
                    'stasis_score': stasis_score,
                    'similarity': similarity_scores[i],
                    'prob_variance': np.var(prob_measures) if len(prob_measures) > 1 else 0.0
                })
        
        # Sort by stasis score and take top_k
        filtered_results.sort(key=lambda x: x['stasis_score'], reverse=True)
        return filtered_results[:self.top_k]
    
    def display_results(self, results: List[Dict], query: str):
        """Pretty print query results."""
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        print(f"Found {len(results)} stable results after probability-stasis filtering\n")
        
        for i, result in enumerate(results, 1):
            print(f"Rank {i}:")
            print(f"  StasisScore: {result['stasis_score']:.4f}")
            print(f"  Similarity:  {result['similarity']:.4f}")
            print(f"  Variance:    {result['prob_variance']:.4f}")
            print(f"  Document:    {result['document'][:200]}...")
            if result['metadata']:
                print(f"  Metadata:    {result['metadata']}")
            print()
