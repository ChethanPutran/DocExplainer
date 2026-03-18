import random
import string
from typing import List, Any, Dict


def generate_id(prefix: str = "q") -> str:
    """Generate a random ID"""
    import uuid
    return f"{prefix}_{str(uuid.uuid4())[:8]}"


def shuffle_alternatives(options: List[Any]) -> List[Any]:
    """Shuffle options while preserving correct answer"""
    shuffled = options.copy()
    random.shuffle(shuffled)
    return shuffled


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    return answer.strip().lower()


def extract_keywords(text: str, max_keywords: int = 3) -> List[str]:
    """Extract keywords from text"""
    words = text.lower().split()
    # Remove common words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'were', 'be', 'been',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                 'could', 'may', 'might', 'must', 'can'}
    
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique_keywords.append(w)
    
    return unique_keywords[:max_keywords]


def format_question_text(text: str, max_length: int = 200) -> str:
    """Format question text"""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> tuple:
    """Calculate confidence interval for scores"""
    import numpy as np
    from scipy import stats
    
    if len(scores) < 2:
        return (0.0, 0.0)
    
    mean = np.mean(scores)
    sem = stats.sem(scores)
    interval = sem * stats.t.ppf((1 + confidence) / 2., len(scores) - 1)
    
    return (mean - interval, mean + interval)


def create_feedback_message(result: Dict) -> str:
    """Create a user-friendly feedback message"""
    score = result.get('percentage', 0)
    correct = result.get('correct_count', 0)
    total = result.get('total_questions', 0)
    
    if score >= 90:
        return f"Excellent! You got {correct}/{total} correct. You've mastered this material!"
    elif score >= 70:
        return f"Good job! You got {correct}/{total} correct. A little more practice and you'll have it."
    elif score >= 50:
        return f"You got {correct}/{total} correct. Keep practicing - you're making progress!"
    else:
        weak_concepts = result.get('weakest_concepts', [])
        if weak_concepts:
            concepts_str = ', '.join(weak_concepts)
            return f"You might need more practice with: {concepts_str}. Let's focus on these areas."
        else:
            return f"You got {correct}/{total} correct. Keep studying and try again!"