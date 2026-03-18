import random
import re
from .base import BaseQuestionStrategy
from ...models.schemas import Question
from ...models.enums import QuestionType, DifficultyLevel


class FillBlankStrategy(BaseQuestionStrategy):
    """Strategy for generating fill-in-the-blank questions"""
    
    def get_question_type(self) -> QuestionType:
        return QuestionType.FILL_BLANK
    
    def generate(self, concept: str, difficulty: DifficultyLevel) -> Question:
        """Generate a fill-in-the-blank question"""
        adjustments = self._get_difficulty_adjustments(difficulty)
        
        # Templates with blanks (marked by ___)
        templates = [
            {
                "text": f"{concept} is a fundamental concept in ________.",
                "answer": "its field of study",
                "explanation": f"is fundamental to its domain"
            },
            {
                "text": f"The main purpose of {concept} is to ________.",
                "answer": "solve specific problems",
                "explanation": f"is designed to address particular challenges"
            },
            {
                "text": f"{concept} works by ________.",
                "answer": "processing information systematically",
                "explanation": f"operates through systematic information processing"
            },
            {
                "text": f"To understand {concept}, you need to know about ________.",
                "answer": "prerequisite concepts",
                "explanation": f"builds upon foundational knowledge"
            },
            {
                "text": f"{concept} can be applied to ________.",
                "answer": "various real-world scenarios",
                "explanation": f"has practical applications across different domains"
            },
            {
                "text": f"The key characteristic of {concept} is ________.",
                "answer": "its unique approach",
                "explanation": f"is distinguished by its particular methodology"
            },
            {
                "text": f"{concept} differs from other concepts in that it ________.",
                "answer": "has specific properties",
                "explanation": f"possesses unique attributes that set it apart"
            }
        ]
        
        # Select template based on difficulty
        if difficulty == DifficultyLevel.BEGINNER:
            # Simpler templates for beginners
            template = random.choice(templates[:3])
        elif difficulty == DifficultyLevel.INTERMEDIATE:
            template = random.choice(templates[3:5])
        else:
            # More complex templates for advanced learners
            template = random.choice(templates[5:])
        
        question_text = template["text"]
        correct_answer = template["answer"]
        explanation_suffix = template["explanation"]
        
        # For advanced difficulty, make the blank more challenging
        if difficulty == DifficultyLevel.ADVANCED:
            # Replace specific terms with blanks
            words = question_text.split()
            if len(words) > 6:
                # Replace a key term with blank
                key_term_index = random.randint(3, min(6, len(words) - 2))
                key_term = words[key_term_index]
                words[key_term_index] = "________"
                question_text = " ".join(words)
                correct_answer = key_term.rstrip('.,!?;:')
        
        # Generate hints based on difficulty
        hints = []
        if random.random() < adjustments['hint_probability']:
            if difficulty == DifficultyLevel.BEGINNER:
                hints = [
                    f"Think about the definition of {concept}.",
                    f"What is the main role of {concept}?",
                    f"Consider where {concept} is typically used."
                ]
            elif difficulty == DifficultyLevel.INTERMEDIATE:
                hints = [
                    f"How does {concept} function in practice?",
                    f"What makes {concept} unique?",
                    f"Consider the applications of {concept}."
                ]
            else:
                hints = [
                    f"Analyze the relationship between {concept} and related concepts.",
                    f"What are the underlying principles of {concept}?",
                    f"Consider edge cases or advanced applications."
                ]
        
        return Question(
            id=self._generate_id(),
            text=question_text,
            type=QuestionType.FILL_BLANK,
            difficulty=difficulty,
            concept=concept,
            options=[],  # No options for fill-in-blank
            correct_answer=correct_answer,
            explanation=f"The blank should be filled with '{correct_answer}' because {concept} {explanation_suffix}.",
            hints=hints[:2]  # Limit to 2 hints
        )