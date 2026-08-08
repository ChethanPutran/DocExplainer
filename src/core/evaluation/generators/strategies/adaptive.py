import json
import random
from typing import Optional
from .base import BaseQuestionStrategy
from ...models.schemas import Question
from ...models.enums import QuestionType, DifficultyLevel
from src.core.agent.base import LLMInterface
from src.core.agent.prompts.templates import question_generation_template


class AdaptiveStrategy(BaseQuestionStrategy):
    """Strategy for generating adaptive questions using LLM"""
    
    def __init__(self, generator, llm_wrapper: Optional[LLMInterface] = None):
        super().__init__(generator)
        self.llm_wrapper = llm_wrapper
        self.question_generation_template = question_generation_template
    
    def get_question_type(self) -> QuestionType:
        return QuestionType.SHORT_ANSWER  # Default, but can generate any type
    
    def generate(self, concept: str, difficulty: DifficultyLevel) -> Question:
        """Generate an adaptive question using LLM"""
        if not self.llm_wrapper:
            # Fallback to multiple choice if no LLM
            return self.generator.get_strategy(QuestionType.MULTIPLE_CHOICE).generate(concept, difficulty)
        
        try: 
            # Set up LLM for JSON output
            self.llm_wrapper.set_prompt_template(self.question_generation_template, json_output=True)
            
            # Generate question
            response = self.llm_wrapper.generate({
                "concept": concept,
                "difficulty": difficulty.value
            })
            
            # Parse response
            if isinstance(response, str):
                question_data = json.loads(response)
            else:
                question_data = response
            
            return self._create_question_from_data(question_data, concept, difficulty)
            
        except Exception as e:
            print(f"LLM question generation failed: {e}")
            # Fallback to multiple choice
            return self.generator.get_strategy(QuestionType.MULTIPLE_CHOICE).generate(concept, difficulty)
    
    
    def _create_question_from_data(self, data: dict, concept: str, 
                                   difficulty: DifficultyLevel) -> Question:
        """Create Question object from LLM response data"""
        from ...models.schemas import QuestionOption
        
        question_type = QuestionType(data.get("question_type", "multiple_choice"))
        options = []
        
        if "options" in data:
            for opt_data in data["options"]:
                options.append(QuestionOption(
                    text=opt_data["text"],
                    is_correct=opt_data.get("is_correct", False),
                    explanation=opt_data.get("explanation", "")
                ))
        
        return Question(
            id=self._generate_id(),
            text=data["question_text"],
            type=question_type,
            difficulty=difficulty,
            concept=concept,
            options=options,
            correct_answer=data["correct_answer"],
            explanation=data.get("explanation", ""),
            hints=data.get("hints", [])
        )