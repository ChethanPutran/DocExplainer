import random
from .base import BaseQuestionStrategy
from ...models.schemas import Question, QuestionOption
from ...models.enums import QuestionType, DifficultyLevel


class TrueFalseStrategy(BaseQuestionStrategy):
    """Strategy for generating true/false questions"""
    
    def get_question_type(self) -> QuestionType:
        return QuestionType.TRUE_FALSE
    
    def generate(self, concept: str, difficulty: DifficultyLevel) -> Question:
        """Generate a true/false question"""
        adjustments = self._get_difficulty_adjustments(difficulty)
        
        # Determine if statement should be true or false
        is_true = random.choice([True, False])
        
        if is_true:
            # True statement templates
            templates = [
                f"{concept} is an important concept in its field.",
                f"{concept} has practical applications.",
                f"Understanding {concept} requires foundational knowledge.",
                f"{concept} can be applied to solve problems."
            ]
            correct_answer = "true"
            explanation = f"This statement is true. {concept} is indeed an important concept."
        else:
            # False statement templates
            templates = [
                f"{concept} is unrelated to any other concepts.",
                f"{concept} has no practical applications.",
                f"{concept} can be mastered without practice.",
                f"{concept} is only used in theoretical contexts."
            ]
            correct_answer = "false"
            explanation = f"This statement is false. {concept} does have connections and applications."
        
        question_text = random.choice(templates)
        
        # Create options
        options = [
            QuestionOption(text="True", is_correct=(correct_answer == "true"),
                         explanation="Select True if the statement is correct."),
            QuestionOption(text="False", is_correct=(correct_answer == "false"),
                         explanation="Select False if the statement is incorrect.")
        ]
        
        # Generate hints
        hints = [
            f"Think carefully about {concept}.",
            f"Consider what you know about {concept}'s properties."
        ]
        
        return Question(
            id=self._generate_id(),
            text=question_text,
            type=QuestionType.TRUE_FALSE,
            difficulty=difficulty,
            concept=concept,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation,
            hints=hints[:1] if random.random() < adjustments['hint_probability'] else []
        )