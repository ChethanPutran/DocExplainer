import re
from typing import List, Optional, Tuple


class TextUtils:
    """Utility functions for text processing"""
    
    @staticmethod
    def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text to max length"""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction (can be improved with NLP)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Count frequencies
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        
        # Sort by frequency
        keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return [k for k, _ in keywords[:max_keywords]]
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    @staticmethod
    def count_sentences(text: str) -> int:
        """Count sentences in text"""
        return len(TextUtils.split_sentences(text))
    
    @staticmethod
    def get_reading_time(text: str, words_per_minute: int = 200) -> float:
        """Estimate reading time in minutes"""
        word_count = TextUtils.count_words(text)
        return word_count / words_per_minute
    
    @staticmethod
    def extract_phrases(text: str, min_words: int = 2, max_words: int = 5) -> List[str]:
        """Extract phrases of specific length"""
        words = text.split()
        phrases = []
        
        for length in range(min_words, max_words + 1):
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i + length])
                if len(phrase) > 3:  # Ignore very short phrases
                    phrases.append(phrase)
        
        return phrases
    
    @staticmethod
    def highlight_terms(text: str, terms: List[str], 
                        highlight_format: str = "<b>{}</b>") -> str:
        """Highlight terms in text"""
        result = text
        for term in terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            result = pattern.sub(lambda m: highlight_format.format(m.group()), result)
        return result
    
    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """Extract citations from text (e.g., [1], (Smith et al., 2020))"""
        citations = []
        
        # Find numeric citations [1], [2,3], etc.
        numeric = re.findall(r'\[[\d,\s]+\]', text)
        citations.extend(numeric)
        
        # Find author-year citations (Smith et al., 2020)
        author_year = re.findall(r'\([A-Za-z\s]+(?:et al\.)?,\s*\d{4}[a-z]?\)', text)
        citations.extend(author_year)
        
        return citations
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
    
    @staticmethod
    def extract_abbreviations(text: str) -> List[Tuple[str, str]]:
        """Extract abbreviations and their definitions"""
        # Pattern: Definition (ABBR)
        pattern = r'([A-Za-z\s]+?)\s*\(([A-Z]{2,})\)'
        matches = re.findall(pattern, text)
        
        # Pattern: ABBR (Definition)
        pattern2 = r'([A-Z]{2,})\s*\(([A-Za-z\s]+?)\)'
        matches2 = re.findall(pattern2, text)
        
        return [(definition.strip(), abbr) for definition, abbr in matches] + \
               [(definition.strip(), abbr) for abbr, definition in matches2]
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract email addresses"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs"""
        pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
        return re.findall(pattern, text)
    
    @staticmethod
    def remove_markdown(text: str) -> str:
        """Remove markdown formatting"""
        # Remove headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # Remove bold/italic
        text = re.sub(r'[*_]{1,3}(.*?)[*_]{1,3}', r'\1', text)
        # Remove links
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # Remove code blocks
        text = re.sub(r'`{3}.*?`{3}', '', text, flags=re.DOTALL)
        # Remove inline code
        text = re.sub(r'`(.*?)`', r'\1', text)
        return text