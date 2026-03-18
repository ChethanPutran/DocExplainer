import random
from .base import BaseQuestionStrategy
from ...models.schemas import Question, QuestionOption
from ...models.enums import QuestionType, DifficultyLevel


class MultipleChoiceStrategy(BaseQuestionStrategy):
    """Strategy for generating multiple choice questions"""
    
    def get_question_type(self) -> QuestionType:
        return QuestionType.MULTIPLE_CHOICE
    
    def generate(self, concept: str, difficulty: DifficultyLevel) -> Question:
        """Generate a multiple choice question"""
        adjustments = self._get_difficulty_adjustments(difficulty)
        
        # Question templates
        templates = [
            f"What is the correct definition of {concept}?",
            f"Which of the following best describes {concept}?",
            f"What is the primary characteristic of {concept}?",
            f"How would you explain {concept} to a beginner?",
            f"What is the main purpose of {concept}?"
        ]
        
        question_text = random.choice(templates)
        
        # Generate correct answer
        correct_answer = f"{concept} is a fundamental concept in its domain."
        
        # Generate distractors
        distractors = self._generate_plausible_distractors(
            concept, 
            correct_answer,
            adjustments['distractor_count']
        )
        
        # Create options
        options = []
        
        # Add correct answer
        options.append(QuestionOption(
            text=correct_answer,
            is_correct=True,
            explanation=f"This correctly defines {concept}."
        ))
        
        # Add distractors
        for distractor in distractors:
            options.append(QuestionOption(
                text=distractor,
                is_correct=False,
                explanation=f"This is not correct because {distractor} doesn't accurately describe {concept}."
            ))
        
        # Shuffle options
        random.shuffle(options)
        
        # Generate hints
        hints = [
            f"Think about what makes {concept} unique.",
            f"Consider the key characteristics of {concept}.",
            f"Remember the definition we discussed earlier."
        ]
        
        return Question(
            id=self._generate_id(),
            text=question_text,
            type=QuestionType.MULTIPLE_CHOICE,
            difficulty=difficulty,
            concept=concept,
            options=options,
            correct_answer=correct_answer,
            explanation=f"{concept} is best defined as {correct_answer}",
            hints=hints[:2] if random.random() < adjustments['hint_probability'] else []
        )