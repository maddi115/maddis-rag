#!/usr/bin/env python3
"""
Gradient Proximity Search with Rarity-Based Chaining & Semantic Filtering
Beautiful structured output with narrative summary and tiered results
"""
import sys
import chromadb
from collections import Counter
import math
import re

class GradientProximitySearch:
    def __init__(self, documents, initial_window=20, base_strength_boost=0.7, min_weight_threshold=0.05, common_word_threshold=0.8):
        self.documents = documents
        self.initial_window = initial_window
        self.base_strength_boost = base_strength_boost
        self.min_weight_threshold = min_weight_threshold
        self.common_word_threshold = common_word_threshold
        
        # Message boundary markers
        self.boundary_markers = {'agentmaddi', 'madddyyyyi'}

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

    def _build_word_doc_frequency(self):
        word_count = Counter()
        for doc in self.documents:
            words = set(self._tokenize(doc.lower()))
            for word in words:
                if word not in self.stop_words and len(word) > 2:
                    word_count[word] += 1
        return word_count

    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def _get_important_words(self, query):
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

    def _gradient_weight(self, distance, decay_rate):
        return math.exp(-distance / decay_rate)

    def _radiate_search(self, doc_words, anchor_pos, query_words, word_rarity_scores):
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

                right_pos = current_pos + offset
                if right_pos < len(doc_words) and right_pos not in visited_positions:
                    word = doc_words[right_pos]
                    if word in query_words:
                        visited_positions.add(right_pos)
                        matched_words[word] = (right_pos, anchor_distance + offset, weight)
                        
                        word_rarity = word_rarity_scores.get(word, 0.5)
                        normalized_rarity = word_rarity / max_rarity
                        dynamic_boost = self.base_strength_boost * normalized_rarity
                        
                        new_strength = min(1.0, current_strength + dynamic_boost)
                        
                        word_doc_count = self.word_doc_count.get(word, 0)
                        if word_doc_count <= commonality_threshold:
                            search_frontier.append((right_pos, new_strength, anchor_distance + offset))

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

    def _generate_summary(self, results, query, anchor_word):
        """Generate narrative summary from results"""
        top_results = results[:10]
        
        summary_lines = []
        summary_lines.append(f"💬 WHAT THE DATA REVEALS: {query}")
        summary_lines.append("=" * 78)
        summary_lines.append(f"Based on {len(results)} conversations analyzed:\n")
        
        # Extract top themes
        anchor_contexts = []
        for r in top_results:
            if r['score'] > 0.3:
                ctx = r['context'].replace('>>>', '').replace('<<<', '')
                anchor_contexts.append(ctx[:100])
        
        if anchor_contexts:
            summary_lines.append(f"🎯 TOP THEMES AROUND '{anchor_word.upper()}':")
            summary_lines.append(f"   • {anchor_contexts[0][:70]}...")
            if len(anchor_contexts) > 1:
                summary_lines.append(f"   • {anchor_contexts[1][:70]}...")
            if len(anchor_contexts) > 2:
                summary_lines.append(f"   • {anchor_contexts[2][:70]}...")
            summary_lines.append("")
        
        # Score distribution
        high = len([r for r in results if r['score'] > 0.5])
        medium = len([r for r in results if 0.1 <= r['score'] <= 0.5])
        low = len([r for r in results if r['score'] < 0.1])
        
        summary_lines.append(f"📊 CONFIDENCE DISTRIBUTION:")
        summary_lines.append(f"   • High confidence (>0.5): {high} matches")
        summary_lines.append(f"   • Medium confidence (0.1-0.5): {medium} matches")
        summary_lines.append(f"   • Low confidence (<0.1): {low} matches")
        
        return "\n".join(summary_lines)

    def _extract_summary(self, context):
        """Extract and format context with threading using →→"""
        # Split by both boundary markers using regex
        import re
        messages = re.split(r'\b(agentmaddi|madddyyyyi)\b', context)
        
        # Reconstruct messages (rejoin marker with following text)
        reconstructed = []
        i = 0
        while i < len(messages):
            if messages[i].strip():
                if messages[i] in ['agentmaddi', 'madddyyyyi']:
                    # This is a marker, skip it (we use it as boundary only)
                    i += 1
                    continue
                else:
                    reconstructed.append(messages[i].strip())
            i += 1
        
        messages = [msg for msg in reconstructed if msg]
        
        if not messages:
            return context
        
        # Find which message contains the anchor
        anchor_message_idx = None
        for idx, msg in enumerate(messages):
            if '>>>' in msg and '<<<' in msg:
                anchor_message_idx = idx
                break
        
        # Format the output
        if len(messages) == 1:
            # Single message, just return it
            return messages[0]
        else:
            # Multiple messages - show as thread with →→
            formatted_lines = []
            for idx, msg in enumerate(messages):
                if idx == 0:
                    formatted_lines.append(msg)
                else:
                    formatted_lines.append(f"   →→ {msg}")
            
            return '\n'.join(formatted_lines)

    def search(self, query):
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

                    # Capture thread context - allow up to 3 message boundaries
                    max_boundaries = 3
                    boundary_count = 0
                    
                    # Scan left to find boundary (but allow multiple messages)
                    context_start = pos
                    for i in range(pos - 1, max(0, pos - 90), -1):
                        if doc_words[i] in self.boundary_markers:
                            boundary_count += 1
                            if boundary_count >= max_boundaries:
                                context_start = i + 1
                                break
                    else:
                        context_start = max(0, pos - 90)
                    
                    # Reset boundary count for right scan
                    boundary_count = 0
                    
                    # Scan right to find boundary (but allow multiple messages)
                    context_end = pos
                    for i in range(pos + 1, min(len(doc_words), pos + 90)):
                        if doc_words[i] in self.boundary_markers:
                            boundary_count += 1
                            if boundary_count >= max_boundaries:
                                context_end = i
                                break
                    else:
                        context_end = min(len(doc_words), pos + 90)
                    
                    context_words = doc_words[context_start:context_end + 1]
                    
                    anchor_pos_in_context = pos - context_start
                    
                    if 0 <= anchor_pos_in_context < len(context_words):
                        context_words[anchor_pos_in_context] = f">>>{context_words[anchor_pos_in_context]}<<<"
                    
                    context = ' '.join(context_words)

                    results.append({
                        'document_idx': doc_idx,
                        'document': document,
                        'score': score,
                        'anchor_position': pos,
                        'anchor_word': anchor_word,
                        'matched_words': list(all_matched),
                        'match_count': len(all_matched),
                        'context': context
                    })

        doc_best_matches = {}
        for result in results:
            doc_idx = result['document_idx']
            if doc_idx not in doc_best_matches or result['score'] > doc_best_matches[doc_idx]['score']:
                doc_best_matches[doc_idx] = result

        deduplicated_results = list(doc_best_matches.values())
        deduplicated_results.sort(key=lambda x: x['score'], reverse=True)

        return deduplicated_results, anchor_word

    def format_results(self, results, query, anchor_word):
        """Format results in clean, dyslexia-friendly layout"""
        output = []
        
        # Generate summary
        summary = self._generate_summary(results, query, anchor_word)
        output.append(summary)
        output.append("")
        
        # Header
        output.append("━" * 78)
        output.append(f"📋 ALL {len(results)} RESULTS · Sorted by Relevance")
        output.append("━" * 78)
        output.append("")
        
        # Categorize results
        high = [r for r in results if r['score'] > 0.5]
        medium = [r for r in results if 0.1 <= r['score'] <= 0.5]
        low = [r for r in results if r['score'] < 0.1]
        
        # High confidence
        if high:
            output.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━🔥 HIGH CONFIDENCE MATCHES (Score > 0.5)")
            output.append("")
            for i, result in enumerate(high, 1):
                summary = self._extract_summary(result['context'])
                output.append(f"[{i}] Score: {result['score']:.3f}")
                output.append(f"{summary}")
                output.append("")
        
        # Medium confidence
        if medium:
            output.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━⭐ MEDIUM CONFIDENCE MATCHES (Score 0.1 - 0.5)")
            output.append("")
            start_idx = len(high) + 1
            for i, result in enumerate(medium, start_idx):
                summary = self._extract_summary(result['context'])
                output.append(f"[{i}] Score: {result['score']:.3f}")
                output.append(f"{summary}")
                output.append("")
        
        # Low confidence
        if low:
            output.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━📌 LOW CONFIDENCE MATCHES (Score 0.0 - 0.1)")
            output.append("")
            start_idx = len(high) + len(medium) + 1
            for i, result in enumerate(low, start_idx):
                summary = self._extract_summary(result['context'])
                output.append(f"[{i}] Score: {result['score']:.3f}")
                output.append(f"{summary}")
                output.append("")
        
        # Footer
        output.append("━" * 78)
        output.append(f"📊 SUMMARY: {len(high)} high confidence · {len(medium)} medium confidence · {len(low)} low confidence")
        
        return "\n".join(output)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 newv2Gradient_Proximity_Search_Rarity_Based_Chaining.py \"your question\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print("\n" + "="*78)
    print("LOADING DATA FROM CHROMADB")
    print("="*78)

    client = chromadb.PersistentClient(path="./agentmaddi_chroma_db")
    collection = client.get_collection("agentmaddi_history")
    all_data = collection.get()
    documents = all_data['documents']

    print(f"✓ Loaded {len(documents)} documents\n")

    search = GradientProximitySearch(
        documents,
        initial_window=20,
        base_strength_boost=0.7,
        min_weight_threshold=0.05,
        common_word_threshold=0.8
    )

    results, anchor_word = search.search(query)
    formatted_output = search.format_results(results, query, anchor_word)
    
    print(formatted_output)
    print()