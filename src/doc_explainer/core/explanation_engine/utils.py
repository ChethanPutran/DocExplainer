import re
from typing import List, Optional


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """Extract keywords from text"""
    # Simple keyword extraction - in production, use NLP
    words = text.lower().split()
    # Remove common words and short words
    keywords = [w for w in words if len(w) > 3 and w not in COMMON_WORDS]
    # Remove duplicates and limit
    seen = set()
    unique_keywords = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique_keywords.append(w)
    
    return unique_keywords[:max_keywords]


def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Estimate reading time in minutes"""
    word_count = len(text.split())
    return max(1, round(word_count / words_per_minute))


def truncate_text(text: str, max_length: int = 300) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def clean_search_query(query: str) -> str:
    """Clean search query for URL encoding"""
    # Remove special characters
    query = re.sub(r'[^\w\s]', ' ', query)
    # Remove extra spaces
    query = ' '.join(query.split())
    return query


# Common words to ignore in keyword extraction
COMMON_WORDS = {
    'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have',
    'are', 'was', 'were', 'has', 'had', 'but', 'not', 'what',
    'all', 'when', 'where', 'why', 'how', 'can', 'will', 'just',
    'more', 'some', 'such', 'than', 'then', 'them', 'they',
    'their', 'there', 'these', 'those', 'about', 'would',
    'could', 'should', 'into', 'like', 'than', 'then'
}


def format_resource_title(concept: str, resource_type: str, level: str) -> str:
    """Format a resource title"""
    type_names = {
        'video': 'Video Tutorial',
        'article': 'Reading',
        'exercise': 'Practice Exercises',
        'course': 'Course'
    }
    
    type_name = type_names.get(resource_type, resource_type.capitalize())
    return f"{type_name}: {concept.title()} ({level})"


def get_difficulty_emoji(level: str) -> str:
    """Get emoji for difficulty level"""
    level_emoji = {
        'beginner': '🌱',
        'intermediate': '📚',
        'advanced': '🎓',
        'expert': '🏆'
    }
    return level_emoji.get(level.lower(), '📖')