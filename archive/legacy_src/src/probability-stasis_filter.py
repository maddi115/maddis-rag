import numpy as np
from typing import List, Dict, Tuple

class ProbabilityStasisFilter:
    """
    Filters RAG retrieval results based on probability stasis:
    Keeps only chunks with high, stable relevance scores.
    """
    
    def __init__(self, top_k: int = 5, stasis_threshold: float = 0.5):
        """
        Args:
            top_k: Maximum number of chunks to return
            stasis_threshold: Minimum StasisScore to keep (0-1 scale)
        """
        self.top_k = top_k
        self.stasis_threshold = stasis_threshold
    
    def calculate_stasis_score(self, probabilities: List[float]) -> float:
        """
        Calculate StasisScore for a set of probability values.
        
        Formula: StasisScore = mean(p_i) / (1 + variance(p_i))
        
        High mean + low variance = high score (stable & probable)
        """
        if not probabilities or len(probabilities) < 2:
            return 0.0
        
        p_array = np.array(probabilities)
        mean_p = np.mean(p_array)
        var_p = np.var(p_array)
        
        stasis_score = mean_p / (1 + var_p)
        return stasis_score
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Convert similarity scores to probability distribution via softmax."""
        scores_array = np.array(scores)
        exp_scores = np.exp(scores_array - np.max(scores_array))  # numerical stability
        probabilities = exp_scores / np.sum(exp_scores)
        return probabilities.tolist()
    
    def filter_chunks(self, 
                     chunks: List[Dict], 
                     similarity_scores: List[float],
                     cross_check_scores: List[List[float]] = None) -> List[Dict]:
        """
        Filter RAG chunks using probability-stasis.
        
        Args:
            chunks: Retrieved document chunks
            similarity_scores: Primary similarity scores (e.g., cosine similarity)
            cross_check_scores: Optional list of additional score types for cross-referencing
                               e.g., [keyword_scores, recency_scores, bm25_scores]
        
        Returns:
            Filtered list of chunks with StasisScore metadata
        """
        if not chunks:
            return []
        
        # Normalize primary scores to probabilities
        primary_probs = self.normalize_scores(similarity_scores)
        
        # Calculate StasisScore for each chunk
        chunk_stasis_scores = []
        
        for i, chunk in enumerate(chunks):
            # Collect all probability measures for this chunk
            prob_measures = [primary_probs[i]]
            
            # Add cross-check scores if provided
            if cross_check_scores:
                for score_list in cross_check_scores:
                    if i < len(score_list):
                        normalized = self.normalize_scores(score_list)
                        prob_measures.append(normalized[i])
            
            # Calculate stasis score across all measures
            stasis_score = self.calculate_stasis_score(prob_measures)
            
            chunk_stasis_scores.append({
                'chunk': chunk,
                'stasis_score': stasis_score,
                'primary_prob': primary_probs[i],
                'prob_variance': np.var(prob_measures) if len(prob_measures) > 1 else 0.0
            })
        
        # Filter by threshold
        filtered = [
            item for item in chunk_stasis_scores 
            if item['stasis_score'] >= self.stasis_threshold
        ]
        
        # Sort by stasis score (descending) and take top_k
        filtered.sort(key=lambda x: x['stasis_score'], reverse=True)
        top_results = filtered[:self.top_k]
        
        # Return chunks with metadata
        return [
            {
                **item['chunk'],
                '_stasis_score': item['stasis_score'],
                '_primary_prob': item['primary_prob'],
                '_prob_variance': item['prob_variance']
            }
            for item in top_results
        ]


# Example Usage
if __name__ == "__main__":
    # Simulated RAG retrieval results
    retrieved_chunks = [
        {'text': 'Function calculates total sum', 'source': 'utils.py'},
        {'text': 'Database connection setup', 'source': 'db.py'},
        {'text': 'Sum calculation implementation', 'source': 'math_utils.py'},
        {'text': 'Error handling for connections', 'source': 'errors.py'},
        {'text': 'Total sum aggregation logic', 'source': 'aggregator.py'},
    ]
    
    # Simulated similarity scores (e.g., from embedding cosine similarity)
    similarity_scores = [0.92, 0.45, 0.89, 0.38, 0.91]
    
    # Optional: Cross-check with other scoring methods
    keyword_match_scores = [0.88, 0.42, 0.90, 0.35, 0.87]
    recency_scores = [0.85, 0.50, 0.88, 0.40, 0.90]
    
    # Initialize filter
    filter_system = ProbabilityStasisFilter(top_k=3, stasis_threshold=0.4)
    
    # Apply probability-stasis filtering
    filtered_results = filter_system.filter_chunks(
        chunks=retrieved_chunks,
        similarity_scores=similarity_scores,
        cross_check_scores=[keyword_match_scores, recency_scores]
    )
    
    # Display results
    print("=== PROBABILITY-STASIS FILTERED RESULTS ===\n")
    for i, result in enumerate(filtered_results, 1):
        print(f"Rank {i}:")
        print(f"  Text: {result['text']}")
        print(f"  Source: {result['source']}")
        print(f"  StasisScore: {result['_stasis_score']:.4f}")
        print(f"  Primary Probability: {result['_primary_prob']:.4f}")
        print(f"  Variance: {result['_prob_variance']:.4f}")
        print()
