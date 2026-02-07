#!/usr/bin/env python3
"""
Transportation Model - Algorithmic Vessel Assembly
Converts Trinity retrieval results into grounded, structured answers
"""
import re
from typing import List, Dict, Any, Optional


class TransportationModel:
    """
    Takes raw retrieval results from Trinity (Gradient, Vector, Stasis)
    and assembles them into vessel-based answers WITHOUT using an LLM.
    """
    
    def __init__(
        self,
        high_confidence_threshold: float = 0.5,
        medium_confidence_threshold: float = 0.2,
        low_confidence_threshold: float = 0.05
    ):
        self.high_threshold = high_confidence_threshold
        self.medium_threshold = medium_confidence_threshold
        self.low_threshold = low_confidence_threshold
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """Determine query type algorithmically"""
        query_lower = query.lower()
        
        if any(query_lower.startswith(word) for word in ['does', 'is', 'do', 'can', 'should', 'would']):
            return {'type': 'yes_no', 'subject': self._extract_subject(query)}
        
        if any(word in query_lower for word in ['what are', 'list', 'which', 'all']):
            return {'type': 'list', 'subject': self._extract_subject(query)}
        
        if any(word in query_lower for word in [' or ', 'versus', 'vs', 'prefer', 'better']):
            return {'type': 'comparison', 'subject': self._extract_subject(query)}
        
        if any(query_lower.startswith(word) for word in ['when', 'what time']):
            return {'type': 'temporal', 'subject': self._extract_subject(query)}
        
        if query_lower.startswith('where'):
            return {'type': 'location', 'subject': self._extract_subject(query)}
        
        return {'type': 'factual', 'subject': self._extract_subject(query)}
    
    def _extract_subject(self, query: str) -> str:
        """Extract the main subject from query"""
        if 'agentmaddi' in query.lower():
            return 'agentmaddi'
        return 'unknown'
    
    def _clean_context(self, context: str) -> str:
        """Remove highlight markers and clean up text"""
        cleaned = context.replace('>>>', '').replace('<<<', '')
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
    def extract_vessels(
        self,
        gradient_results: List[Dict],
        vector_results: List[Dict],
        stasis_results: List[Dict]
    ) -> Dict[str, Any]:
        """Extract BEST vessel from EACH method - one from Gradient, one from Vector, one from Stasis"""
        vessels = {
            'primary': None,
            'supporting': [],
            'excluded': [],
            'metadata': {
                'total_sources': 0,
                'high_confidence_count': 0,
                'medium_confidence_count': 0,
                'low_confidence_count': 0
            }
        }
        
        # Get BEST from each method
        best_gradient = None
        best_vector = None
        best_stasis = None
        
        # GRADIENT - get highest score
        if gradient_results:
            for result in gradient_results:
                score = result.get('score', 0.0)
                if score >= self.low_threshold:
                    if not best_gradient or score > best_gradient['confidence']:
                        best_gradient = {
                            'content': self._clean_context(result.get('context', '')),
                            'confidence': score,
                            'source_method': 'Gradient Proximity',
                            'match_type': 'gradient_chain'
                        }
        
        # VECTOR - get highest score
        if vector_results:
            for result in vector_results:
                score = result.get('score', 0.0)
                if score >= self.low_threshold:
                    if not best_vector or score > best_vector['confidence']:
                        best_vector = {
                            'content': result.get('document', ''),
                            'confidence': score,
                            'source_method': 'Vector Search',
                            'match_type': 'semantic_similarity'
                        }
        
        # STASIS - get highest score
        if stasis_results:
            for result in stasis_results:
                score = result.get('stasis_score', 0.0)
                if score >= self.low_threshold:
                    if not best_stasis or score > best_stasis['confidence']:
                        best_stasis = {
                            'content': result.get('document', ''),
                            'confidence': score,
                            'source_method': 'Stasis',
                            'match_type': 'probability_stable'
                        }
        
        # Collect all three best results
        candidates = []
        if best_gradient:
            candidates.append(best_gradient)
        if best_vector:
            candidates.append(best_vector)
        if best_stasis:
            candidates.append(best_stasis)
        
        # Sort by confidence to get primary
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Assign primary (highest confidence across all methods)
        if candidates:
            vessels['primary'] = {
                'content': candidates[0]['content'],
                'confidence': candidates[0]['confidence'],
                'source_method': candidates[0]['source_method'],
                'match_type': candidates[0]['match_type'],
                'locked': True,
                'role': 'direct_answer'
            }
            
            if candidates[0]['confidence'] >= self.high_threshold:
                vessels['metadata']['high_confidence_count'] += 1
            elif candidates[0]['confidence'] >= self.medium_threshold:
                vessels['metadata']['medium_confidence_count'] += 1
            else:
                vessels['metadata']['low_confidence_count'] += 1
            
            # Assign supporting (the other two methods)
            for candidate in candidates[1:]:
                vessels['supporting'].append({
                    'content': candidate['content'],
                    'confidence': candidate['confidence'],
                    'source_method': candidate['source_method'],
                    'match_type': candidate['match_type'],
                    'locked': True,
                    'role': 'supporting_evidence'
                })
                
                if candidate['confidence'] >= self.high_threshold:
                    vessels['metadata']['high_confidence_count'] += 1
                elif candidate['confidence'] >= self.medium_threshold:
                    vessels['metadata']['medium_confidence_count'] += 1
                else:
                    vessels['metadata']['low_confidence_count'] += 1
        
        vessels['metadata']['total_sources'] = len(vessels['supporting']) + (1 if vessels['primary'] else 0)
        
        return vessels
    
    def assemble_answer(
        self,
        query: str,
        vessels: Dict[str, Any],
        query_classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assemble the final answer using clean format with vessels from all 3 methods"""
        
        if not vessels['primary']:
            verdict = 'INSUFFICIENT_EVIDENCE'
            confidence = 0.0
        else:
            verdict = self._determine_verdict(query_classification, vessels)
            confidence = vessels['primary']['confidence']
        
        # Assemble based on query type
        if query_classification['type'] == 'yes_no':
            assembled = self._assemble_yes_no(query, vessels, verdict, confidence)
        elif query_classification['type'] == 'list':
            assembled = self._assemble_list(query, vessels, verdict, confidence)
        elif query_classification['type'] == 'temporal':
            assembled = self._assemble_temporal(query, vessels, verdict, confidence)
        else:
            assembled = self._assemble_factual(query, vessels, verdict, confidence)
        
        return assembled
    
    def _determine_verdict(self, classification: Dict, vessels: Dict) -> str:
        """Determine YES/NO verdict algorithmically"""
        if classification['type'] == 'yes_no':
            primary_content = vessels['primary']['content'].lower()
            if any(word in primary_content for word in ['i like', 'i love', 'i enjoy', 'awesome', 'great', 'yes']):
                return 'YES'
            elif any(word in primary_content for word in ['i hate', 'i dont', "don't", 'dislike', 'no']):
                return 'NO'
            else:
                return 'YES'
        return 'FOUND'
    
    def _assemble_yes_no(self, query: str, vessels: Dict, verdict: str, confidence: float) -> Dict[str, Any]:
        """Assemble with vessels from all 3 methods, clearly showing sources"""
        if verdict == 'INSUFFICIENT_EVIDENCE':
            text = f"Insufficient evidence to answer: {query}"
        else:
            primary = vessels['primary']
            
            # Start with primary vessel WITH SOURCE
            text = f'According to the retrieved logs: "{primary["content"]}" — (Source: {primary["source_method"]}, Confidence: {primary["confidence"]:.3f})'
            
            # Add supporting vessels WITH SOURCES
            if vessels['supporting']:
                for supp in vessels['supporting']:
                    if len(supp['content']) < 250:
                        text += f'. Additionally: "{supp["content"]}" — (Source: {supp["source_method"]}, Confidence: {supp["confidence"]:.3f})'
            
            text += '.'
        
        return {
            'query': query,
            'verdict': verdict,
            'confidence': confidence,
            'assembled_text': text,
            'vessels_used': {
                'primary': vessels['primary'],
                'supporting': vessels['supporting']
            },
            'needs_llm_polish': False
        }
    
    def _assemble_list(self, query: str, vessels: Dict, verdict: str, confidence: float) -> Dict[str, Any]:
        """Assemble list with sources"""
        if verdict == 'INSUFFICIENT_EVIDENCE':
            text = f"No clear evidence found for: {query}"
        else:
            primary = vessels['primary']
            text = f'According to the logs: "{primary["content"]}" — ({primary["source_method"]})'
            
            if vessels['supporting']:
                for supp in vessels['supporting']:
                    if len(supp['content']) < 200:
                        text += f', "{supp["content"]}" — ({supp["source_method"]})'
            
            text += '.'
        
        return {
            'query': query,
            'verdict': verdict,
            'confidence': confidence,
            'assembled_text': text,
            'vessels_used': vessels,
            'needs_llm_polish': False
        }
    
    def _assemble_temporal(self, query: str, vessels: Dict, verdict: str, confidence: float) -> Dict[str, Any]:
        """Assemble temporal answer with sources"""
        if verdict == 'INSUFFICIENT_EVIDENCE':
            text = f"No temporal information found for: {query}"
        else:
            primary = vessels['primary']
            text = f'According to the retrieved logs: "{primary["content"]}" — (Source: {primary["source_method"]}, Confidence: {primary["confidence"]:.3f})'
            
            if vessels['supporting']:
                for supp in vessels['supporting']:
                    if len(supp['content']) < 200:
                        text += f'. Additionally: "{supp["content"]}" — (Source: {supp["source_method"]}, Confidence: {supp["confidence"]:.3f})'
            
            text += '.'
        
        return {
            'query': query,
            'verdict': verdict,
            'confidence': confidence,
            'assembled_text': text,
            'vessels_used': vessels,
            'needs_llm_polish': False
        }
    
    def _assemble_factual(self, query: str, vessels: Dict, verdict: str, confidence: float) -> Dict[str, Any]:
        """Assemble general factual answer with sources"""
        if verdict == 'INSUFFICIENT_EVIDENCE':
            text = f"No evidence found for: {query}"
        else:
            primary = vessels['primary']
            text = f'According to the logs: "{primary["content"]}" — (Source: {primary["source_method"]}, Confidence: {primary["confidence"]:.3f})'
            
            if vessels['supporting']:
                for supp in vessels['supporting']:
                    if len(supp['content']) < 200:
                        text += f'. Additionally: "{supp["content"]}" — (Source: {supp["source_method"]}, Confidence: {supp["confidence"]:.3f})'
            
            text += '.'
        
        return {
            'query': query,
            'verdict': verdict,
            'confidence': confidence,
            'assembled_text': text,
            'vessels_used': vessels,
            'needs_llm_polish': False
        }
    
    def transport(
        self,
        query: str,
        gradient_results: List[Dict],
        vector_results: List[Dict],
        stasis_results: List[Dict]
    ) -> Dict[str, Any]:
        """Main transport function"""
        query_classification = self.classify_query(query)
        vessels = self.extract_vessels(gradient_results, vector_results, stasis_results)
        assembled = self.assemble_answer(query, vessels, query_classification)
        
        assembled['query_classification'] = query_classification
        assembled['vessels'] = vessels
        assembled['transport_metadata'] = {
            'high_threshold': self.high_threshold,
            'medium_threshold': self.medium_threshold,
            'low_threshold': self.low_threshold
        }
        
        return assembled


def format_output(result: Dict[str, Any]) -> str:
    """Format transportation result for display"""
    lines = []
    lines.append("\n" + "="*78)
    lines.append("🚀 TRANSPORTATION MODEL OUTPUT")
    lines.append("="*78)
    lines.append(f"\n{result['assembled_text']}\n")
    lines.append("="*78)
    
    return "\n".join(lines)
