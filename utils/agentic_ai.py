from typing import List, Dict, Any, Optional
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class AgentTools:
    """Tools for the agentic AI system"""
    
    @staticmethod
    def format_response(content: str) -> str:
        """Format response for better readability"""
        # Clean extra whitespace and split into lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        formatted_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Handle bullet points
            if line.startswith('•'):
                formatted_content.append(line)
            # Handle numbered lists
            elif any(line.startswith(f"{num}.") for num in range(1, 10)):
                formatted_content.append('\n' + line)
            # Handle regular paragraphs
            else:
                if i > 0 and not any(lines[i-1].startswith(marker) for marker in ['•', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']):
                    formatted_content.append(' ' + line)
                else:
                    formatted_content.append(line)
            i += 1
        
        # Join content and format paragraphs
        content = ' '.join(formatted_content)
        content = content.replace('. ', '.\n\n')
        
        return content.strip()

    @staticmethod
    def evaluate_response_quality(response: str) -> float:
        """Enhanced quality evaluation"""
        metrics = {
            'length': len(response) > 200,  # Increased minimum length
            'structure': all(marker in response for marker in ['DEFINISI', 'PEMBAHASAN', 'STANDAR']),
            'formatting': '•' in response or any(str(i)+'.' in response for i in range(1,10)),
            'coherence': not any(marker in response for marker in ['I apologize', 'I\'m sorry', 'tidak dapat']),
            'sources': 'REFERENSI' in response
        }
        weights = {'length': 0.15, 'structure': 0.25, 'formatting': 0.2, 'coherence': 0.2, 'sources': 0.2}
        return sum(metrics[k] * weights[k] for k in metrics)

    @staticmethod
    def extract_key_concepts(text: str) -> List[str]:
        """Enhanced concept extraction"""
        key_markers = [
            'adalah', 'merupakan', 'didefinisikan sebagai', 'diartikan sebagai',
            'mencakup', 'terdiri dari', 'meliputi', 'mengacu pada'
        ]
        concepts = []
        for marker in key_markers:
            if marker in text.lower():
                # Extract the sentence containing the marker
                idx = text.lower().find(marker)
                start = text.rfind('.', 0, idx) + 1
                end = text.find('.', idx)
                if start < end:
                    concepts.append(text[start:end].strip())
        return concepts

class SearchAgent:
    """Enhanced search agent"""
    
    def __init__(self):
        self.previous_queries = []
        self.successful_patterns = {}
    
    def optimize_query(self, query: str) -> str:
        """Improved query optimization"""
        # Add domain-specific context
        domain_terms = {
            'arsip': 'manajemen arsip dan dokumentasi',
            'dokumen': 'pengelolaan dokumen dan rekod',
            'preservasi': 'preservasi dan konservasi arsip',
            'akses': 'akses dan keamanan arsip',
            'retensi': 'jadwal retensi arsip',
            'klasifikasi': 'klasifikasi dan pengorganisasian arsip'
        }
        
        optimized_query = query
        for term, context in domain_terms.items():
            if term in query.lower():
                optimized_query = f"{context}: {query}"
                break
        
        self.previous_queries.append(optimized_query)
        return optimized_query
    
    def suggest_related_queries(self, query: str) -> List[str]:
        """Suggest related queries based on current query"""
        related = []
        # Map common relationships
        relationships = {
            'definisi': ['prosedur', 'standar', 'contoh'],
            'prosedur': ['definisi', 'standar', 'praktik terbaik'],
            'standar': ['regulasi', 'praktik terbaik', 'implementasi']
        }
        
        for key, values in relationships.items():
            if key in query.lower():
                for value in values:
                    related.append(query.replace(key, value))
        
        return related[:3]  # Return top 3 suggestions

class ResponseAgent:
    """Enhanced response agent"""
    
    def __init__(self):
        self.tools = AgentTools()
        self.quality_threshold = 0.7
    
    def enhance_response(self, response: str, context: Optional[List[Document]] = None) -> str:
        """Improved response enhancement"""
        # Format the response first
        response = AgentTools.format_response(response)
        
        # Evaluate quality
        quality = self.tools.evaluate_response_quality(response)
        
        if quality < self.quality_threshold:
            # Add section headers if missing
            if 'DEFINISI' not in response:
                response = "DEFINISI\n" + response
            
            # Add source references if available
            if context:
                response += "\n\nSumber Referensi:\n"
                sources = set()
                for doc in context:
                    if 'source' in doc.metadata:
                        sources.add(doc.metadata['source'])
                for source in sources:
                    response += f"• {source}\n"
        
        return response
    
    def should_ask_clarification(self, query: str) -> bool:
        """Determine if clarification is needed"""
        ambiguous_markers = ['bagaimana', 'apa saja', 'seperti apa']
        return any(marker in query.lower() for marker in ambiguous_markers)
    
    def generate_clarifying_questions(self, query: str) -> List[str]:
        """Generate clarifying questions if needed"""
        questions = []
        if 'prosedur' in query.lower():
            questions.append("Apakah Anda mencari prosedur untuk konteks spesifik?")
        if 'standar' in query.lower():
            questions.append("Apakah Anda membutuhkan standar nasional atau internasional?")
        return questions

class KnowledgeAgent:
    """Enhanced knowledge agent"""
    
    def __init__(self):
        self.knowledge_gaps = set()
        self.source_reliability = {}
    
    def identify_knowledge_gaps(self, query: str, response: str) -> List[str]:
        """Identify missing knowledge areas"""
        gaps = []
        if 'tidak ditemukan' in response.lower() or 'tidak tersedia' in response.lower():
            self.knowledge_gaps.add(query)
            gaps.append(f"Perlu menambahkan informasi tentang: {query}")
        return gaps
    
    def suggest_sources(self, topic: str) -> List[Dict[str, str]]:
        """Improved source suggestions"""
        source_mapping = {
            'standar': [
                {'name': 'ISO 15489', 'type': 'standard', 'reliability': 0.95},
                {'name': 'ICA Guidelines', 'type': 'guideline', 'reliability': 0.9},
                {'name': 'ANRI Standards', 'type': 'regulation', 'reliability': 0.9}
            ],
            'prosedur': [
                {'name': 'ANRI Best Practices', 'type': 'guide', 'reliability': 0.9},
                {'name': 'SAA Procedures', 'type': 'guide', 'reliability': 0.85}
            ],
            # ...rest of mapping...
        }
        
        suggestions = []
        for key, sources in source_mapping.items():
            if key in topic.lower():
                for source in sources:
                    suggestions.append(source)
        
        return suggestions

    def evaluate_source_reliability(self, source: str) -> float:
        """Evaluate source reliability"""
        reliability_scores = {
            'ICA': 0.95,
            'ISO': 0.95,
            'ANRI': 0.90,
            'UU': 0.90,
            'journal': 0.85
        }
        
        for key, score in reliability_scores.items():
            if key.lower() in source.lower():
                return score
        return 0.7  # Default reliability score
