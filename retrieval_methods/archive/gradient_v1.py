"""
Gradient Proximity Search - Chaining Algorithm with Rarity-Based Boosting
"""
from typing import List, Dict, Any, Tuple
from collections import Counter
import math
import re

class GradientProximitySearch:
    def __init__(self, documents: List[str], initial_window: int = 20, 
                 base_strength_boost: float = 0.7, min_weight_threshold: float = 0.05, 
                 common_word_threshold: float = 0.8):
        self.documents = documents
        self.initial_window = initial_window
        self.base_strength_boost = base_strength_boost
        self.min_weight_threshold = min_weight_threshold
        self.common_word_threshold = common_word_threshold
        
        # Enhanced stopwords (expanded list)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
            'who', 'when', 'where', 'why', 'how',
            'kind', 'type', 'sort', 'stuff', 'thing', 'things', 'way', 'ways',
            'lot', 'lots', 'much', 'many', 'some', 'any', 'all', 'more', 'most',
            'other', 'another', 'such', 'just', 'very', 'too', 'also', 'even',
            'really', 'quite', 'pretty', 'actually', 'basically', 'literally'
        }

        self.word_doc_count = self._build_word_doc_frequency()

    def _build_word_doc_frequency(self) -> Dict[str, int]:
        word_count = Counter()
        for doc in self.documents:
            words = set(self._tokenize(doc.lower()))
            for word in words:
                if word not in self.stop_words and len(word) > 2:
                    word_count[word] += 1
        return dict(word_count)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _get_important_words(self, query: str) -> Tuple[List[str], Dict[str, float]]:
        words = self._tokenize(query.lower())
        meaningful_words = [w for w in words if w not in self.stop_words and len(w) > 2]

        total_docs = len(self.documents)
        word_scores = []

        for word in meaningful_words:
            doc_count = self.word_doc_count.get(word, 0)
            if doc_count > 0:
                rarity_score = math.log(total_docs / doc_count)
                word_scores.append((word, rarity_score))

        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        words_list = [word for word, score in word_scores]
        rarity_dict = {word: score for word, score in word_scores}
        
        return words_list, rarity_dict

    def _gradient_weight(self, distance: int, decay_rate: float) -> float:
        return math.exp(-distance / decay_rate)

    def _radiate_search(self, doc_words: List[str], anchor_pos: int, 
                       query_words: set, word_rarity_scores: Dict[str, float]) -> Tuple[float, dict]:
        decay_rate = self.initial_window / 3
        matched_words = {}
        visited_positions = {anchor_pos}

        search_frontier = [(anchor_pos, 1.0, 0)]
        
        max_rarity = max(word_rarity_scores.values()) if word_rarity_scores else 1.0
        total_docs = len(self.documents)
        commonality_threshold = total_docs * self.common_word_threshold

        while search_frontier:
            current_pos, current_strength, anchor_distance = search_frontier.pop(0)

            if current_strength < self.min_weight_threshold:
                continue

            for offset in range(1, self.initial_window + 1):
                weight = self._gradient_weight(offset, decay_rate) * current_strength

                if weight < self.min_weight_threshold:
                    break

                # Check right
                right_pos = current_pos + offset
                if right_pos < len(doc_words) and right_pos not in visited_positions:
                    word = doc_words[right_pos]
                    if word in query_words:
                        visited_positions.add(right_pos)
                        matched_words[word] = (right_pos, anchor_distance + offset, weight)
                        
                        # Dynamic boost based on word rarity
                        word_rarity = word_rarity_scores.get(word, 0.5)
                        normalized_rarity = word_rarity / max_rarity
                        dynamic_boost = self.base_strength_boost * normalized_rarity
                        
                        new_strength = min(1.0, current_strength + dynamic_boost)
                        
                        # Only continue chaining if word is rare enough
                        word_doc_count = self.word_doc_count.get(word, 0)
                        if word_doc_count <= commonality_threshold:
                            search_frontier.append((right_pos, new_strength, anchor_distance + offset))

                # Check left
                left_pos = current_pos - offset
                if left_pos >= 0 and left_pos not in visited_positions:
                    word = doc_words[left_pos]
                    if word in query_words:
                        visited_positions.add(left_pos)
                        matched_words[word] = (left_pos, anchor_distance + offset, weight)
                        
                        word_rarity = word_rarity_scores.get(word, 0.5)
                        normalized_rarity = word_rarity / max_rarity
                        dynamic_boost = self.base_strength_boost * normalized_rarity
                        
                        new_strength = min(1.0, current_strength + dynamic_boost)
                        
                        word_doc_count = self.word_doc_count.get(word, 0)
                        if word_doc_count <= commonality_threshold:
                            search_frontier.append((left_pos, new_strength, anchor_distance + offset))

        total_score = sum(weight for _, _, weight in matched_words.values())
        return total_score, matched_words

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        important_words, word_rarity_scores = self._get_important_words(query)

        if not important_words:
            return []

        results = []
        anchor_word = important_words[0]
        query_words = set(important_words[1:])

        for doc_idx, document in enumerate(self.documents):
            doc_words = self._tokenize(document.lower())

            for pos, word in enumerate(doc_words):
                if word == anchor_word:
                    score, matched_words = self._radiate_search(
                        doc_words, pos, query_words, word_rarity_scores
                    )

                    all_matched = {anchor_word}
                    all_matched.update(matched_words.keys())

                    results.append({
                        'score': score,
                        'document': document,
                        'doc_idx': doc_idx,
                        'anchor_word': anchor_word,
                        'anchor_position': pos,
                        'matched_words': list(all_matched),
                        'match_count': len(all_matched)
                    })

        # Deduplicate: Keep best match per document
        doc_best_matches = {}
        for result in results:
            doc_idx = result['doc_idx']
            if doc_idx not in doc_best_matches or result['score'] > doc_best_matches[doc_idx]['score']:
                doc_best_matches[doc_idx] = result

        deduplicated_results = list(doc_best_matches.values())
        deduplicated_results.sort(key=lambda x: x['score'], reverse=True)
        
        return deduplicated_results[:top_k]
