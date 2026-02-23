from src.core.memory.session import Context
from src.core.agent.output_parsers import explanation_output_parser
from src.core.agent.prompts import ask_prompt,summarize_prompt,explain_prompt
from src.core.agent.llm_wrapper import LLMWrapper
from src.core.agent.models import ExplanationStyleEnum, Explanation,ExplanationMetadata
from src.core.explanation_engine.resource_recommender import ResourceRecommender
import logging
from langchain_classic.output_parsers import RetryWithErrorOutputParser

# Set up logging for parsing errors
logger = logging.getLogger(__name__)



class AdaptiveExplainer:
    def __init__(
        self,
        embedding_model=None,
        explanation_style: ExplanationStyleEnum = ExplanationStyleEnum.BEGINNER,
    ):
        self.llm = LLMWrapper()
        self.embedding_model = embedding_model
        self.explanation_style = explanation_style
        self.recommender = ResourceRecommender()
        # Initialize the Retry Parser
        self.retry_parser = RetryWithErrorOutputParser.from_llm(
            parser=explanation_output_parser, 
            llm=self.llm.get_model()
        )

    # --- Core Pipeline Logic ---

    def summarize(self, text: str, context: Context) -> Explanation:
        inputs = self._get_common_inputs(context)
        inputs["selected_text"] = text
        inputs["format_instructions"] = explanation_output_parser.get_format_instructions()
        
        # Structure logic: Add specific length constraints for summary
        inputs["structure"] = "Bullet points"
        inputs["length"] = "Concise summary"
        
        formatted_prompt = summarize_prompt.format(**inputs)
        return self._generate_and_parse(formatted_prompt, source_text=text, context=context)

    def explain(self, text: str, context: Context) -> Explanation:
        inputs = self._get_common_inputs(context)
        inputs["selected_text"] = text
        inputs["format_instructions"] = explanation_output_parser.get_format_instructions()
        
        formatted_prompt = explain_prompt.format(**inputs)
        return self._generate_and_parse(formatted_prompt, source_text=text, context=context)

    def ask(self, question: str, context: Context) -> Explanation:
        inputs = self._get_common_inputs(context)
        inputs["question"] = question
        # For Q&A, selected_text might be the current section/paragraph text
        inputs["selected_text"] = context.document_context.get("text", "") if context.document_context else ""
        inputs["format_instructions"] = explanation_output_parser.get_format_instructions()
        
        formatted_prompt = ask_prompt.format(**inputs)
        return self._generate_and_parse(formatted_prompt, source_text=question, context=context)

    def _generate_and_parse(self, formatted_prompt: str, source_text: str, context: Context) -> Explanation:
        # 1. LLM provides the Pydantic model
        self.llm.set_prompt(formatted_prompt)
        self.llm.generate()
        parsed_llm = explanation_output_parser.parse(raw_output)
        
        # 2. Recommender converts 'suggestions' into 'actual resources'
        enriched_resources = []
        for sug in parsed_llm.suggested_resources:
            if sug.resource_type == "video":
                res = self.recommender.recommend_videos(sug.concept, sug.difficulty)
            elif sug.resource_type == "article":
                res = self.recommender.recommend_articles(sug.concept, sug.difficulty)
            else:
                res = self.recommender.recommend_exercises(sug.concept, sug.difficulty)
            enriched_resources.append(res)
            
        # 3. Combine everything into the final Explanation object
        return Explanation(
            explanation=parsed_llm.explanation,
            style=parsed_llm.style,
            context_used=parsed_llm.context_used,
            known_concepts_used=parsed_llm.known_concepts_used,
            unknown_concepts_explained=parsed_llm.unknown_concepts_explained,
            follow_up_questions=parsed_llm.follow_up_questions,
            metadata=parsed_llm.metadata,
            resources=enriched_resources  # <--- The key addition
        )
   

    # --- Helpers ---

    def _get_common_inputs(self, context: Context):
        """Helper to extract common variables from the BKT User Model."""
        known = []
        unknown = []
        
        # Access the UserKnowledgeState from your Context object
        if context and context.user_knowledge:
            states = context.user_knowledge.knowledge_states # Dict[Concept, KnowledgeState]
            
            for concept, state in states.items():
                # BKT Logic: High knowledge + high confidence = Known
                if state.p_knowledge > 0.7 and state.confidence > 0.6:
                    known.append(concept.name)
                # Low knowledge + significant attempts = Knowledge Gap
                elif state.p_knowledge < 0.4 and state.n_attempts > 0:
                    unknown.append(concept.name)

        return {
            "context_summary": self._context_snippet(context),
            # Pass clean comma-separated strings to the LLM
            "known_concepts": ", ".join(known) if known else "Fundamental basics",
            "unknown_concepts": ", ".join(unknown) if unknown else "None identified yet",
            "tone": "encouraging and academic",
            "complexity": self.explanation_style.value,
            "math_level": "descriptive",
            "structure": "hierarchical",
            "length": "concise"
        }

    
    # def summarize(self, text: str, context: Context) -> Explanation:
    #     summary = self._safe_summary(text,context)
    #     return self._build_response(
    #         text=summary,
    #         source=text,
    #         context=context,
    #         follow_ups=["Do you want a shorter 1-line summary?", "Should I expand any part in detail?"],
    #     )
    
    # def explain(self, text: str, context: Context) -> Explanation:
    #     context_snippet = self._context_snippet(context)
    #     explanation = (
    #         f"This passage discusses: {text.strip()}\n\n"
    #         f"In simple terms, it means the section is focusing on the key idea above and its role in the document."
    #     )
    #     if context_snippet:
    #         explanation += f"\n\nEarlier context: {context_snippet}"

    #     return self._build_response(
    #         text=explanation,
    #         source=text,
    #         context=context,
    #         follow_ups=[
    #             "Do you want this explained with an example?",
    #             "Should I explain prerequisite concepts first?",
    #         ],
    #     )

    # def ask(self, text: str, context: Context) -> Explanation:
    #     answer = (
    #         "I received your question and mapped it to the current document context. "
    #         f"Question: {text.strip()}"
    #     )
    #     return self._build_response(
    #         text=answer,
    #         source=text,
    #         context=context,
    #         follow_ups=["Do you want a concise answer or detailed answer?"],
    #     )
  

    def _safe_summary(self, text: str,context, max_words: int = 40) -> str:
        # words = text.split()
        # if len(words) <= max_words:
        #     return " ".join(words)
        # return " ".join(words[:max_words]) + " ..."

        return ""

    def _context_snippet(self, context: Context) -> str:
        if not context or not context.document_context:
            return "No specific document context available."
        
        # document_context is likely a dict or object with a 'text' field
        text = ""
        if isinstance(context.document_context, dict):
            text = context.document_context.get("text", "")
        else:
            text = getattr(context.document_context, "text", "")
            
        return text[:300].strip() + "..." if text else "Context empty."

    def _estimate_complexity(self, text: str) -> float:
        words = len(text.split())
        if words < 20: return 0.2
        if words < 80: return 0.5
        return 0.8

    def _build_fallback_response(self, message: str, source: str, context: Context) -> Explanation:
        """Returns a valid Explanation object when the LLM fails."""
        return Explanation(
            explanation=message,
            style={"mode": self.explanation_style.value, "depth": "fixed"},
            context_used={"error_state": True},
            known_concepts_used=[],
            unknown_concepts_explained=[],
            follow_up_questions=["Could you try rephrasing the question?", "Explain it in simpler terms?"],
            metadata=ExplanationMetadata(
                estimated_complexity=self._estimate_complexity(source),
                user_knowledge_matched=False,
                gap_bridging=False,
            ),
        )
    