from typing import Dict, List, Tuple
from core.memory.session import Context
from src.core.knowlege_modelling.knowledge_tracing import ConceptGraph
from src.core.knowlege_modelling.user_model import KnowledgeState, UserKnowledgeState
# import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
from enum import Enum

class ExplanationDepth(Enum):
    ADAPTIVE = 'adaptive'
    FIXED = 'fixed'

class ExplanationStructure(Enum):
    ANALOGY_FIRST = 'analogy_first'
    FORMAL_FIRST = 'formal_first'

class ExplanationLength(Enum):
    CONCISE = 'concise'
    DETAILED = 'detailed'
    COMPREHENSIVE = 'comprehensive'

class ExplanationTone(Enum):
    FRIENDLY = 'friendly'
    FORMAL = 'formal'
    BALANCED = 'balanced'

class ExplanationComplexity(Enum):
    SIMPLE = 'simple'
    TECHNICAL = 'technical'
    FORMAL = 'formal'

class ExplanationRelevance(Enum):
    CONTEXTUAL = 'contextual'
    GENERAL = 'general'
    SPECIFIC = 'specific'

class ExplanationMathLevel(Enum):
    NONE = 'none'
    KEY_EQUATIONS = 'key_equations'
    FULL_FORMULATIONS = 'full_formulations'

class ExplanationStyleEnum(Enum):
    BEGINNER = 'beginner'
    INTERMEDIATE = 'intermediate'
    ADVANCED = 'advanced'

class ExplanationStyle:
    def __init__(self,
                explanation_style: Enum = ExplanationStyleEnum.BEGINNER,
                explanation_depth: Enum = ExplanationDepth.ADAPTIVE,
                explanation_structure: Enum = ExplanationStructure.ANALOGY_FIRST,
                explanation_length: Enum = ExplanationLength.CONCISE,
                explanation_tone: Enum = ExplanationTone.FRIENDLY,
                explanation_complexity: Enum = ExplanationComplexity.SIMPLE,
                explanation_relevance: Enum = ExplanationRelevance.CONTEXTUAL,
                explanation_math_level: Enum = ExplanationMathLevel.NONE) -> None:
        self.explanation_style = explanation_style
        self.explanation_depth = explanation_depth
        self.explanation_structure = explanation_structure
        self.explanation_length = explanation_length
        self.explanation_tone = explanation_tone
        self.explanation_complexity = explanation_complexity
        self.explanation_relevance = explanation_relevance
        self.explanation_math_level = explanation_math_level
    
class ExplanationStyleBuilder:
    @staticmethod
    def build_style(explanation_style: ExplanationStyleEnum) -> ExplanationStyle:
        """
        Build ExplanationStyle based on predefined templates:
        'beginner': {
            'tone': 'friendly and encouraging',
            'complexity': 'simple analogies and everyday examples',
            'math_level': 'avoid equations, use intuitive descriptions',
            'structure': 'start with "Imagine that..." or "Think of it like..."',
                                    'length': 'concise (3-5 sentences)'
        'intermediate': {
            'tone': 'balanced and informative',
            'complexity': 'clear definitions with examples',
            'math_level': 'include key equations with explanations',
            'structure': 'definition → example → application',
            'length': 'detailed (5-8 sentences)'
        'advanced': {
            'tone': 'formal and precise',
            'complexity': 'technical definitions and formal reasoning',
            'math_level': 'full mathematical formulations',
            'structure': 'formal definition → theorem/proof → implications',
            'length': 'comprehensive (8-12 sentences)'
        }
        """
        if explanation_style == ExplanationStyleEnum.BEGINNER:
            return ExplanationStyle(
                explanation_style=ExplanationStyleEnum.BEGINNER,
                explanation_tone=ExplanationTone.FRIENDLY,
                explanation_complexity=ExplanationComplexity.SIMPLE,
                explanation_math_level=ExplanationMathLevel.NONE,
                explanation_structure=ExplanationStructure.ANALOGY_FIRST,
                explanation_length=ExplanationLength.CONCISE    
            )
        elif explanation_style == ExplanationStyleEnum.ADVANCED:
            return ExplanationStyle(
                explanation_style=ExplanationStyleEnum.ADVANCED,
                explanation_tone=ExplanationTone.FORMAL,
                explanation_complexity=ExplanationComplexity.FORMAL,
                explanation_math_level=ExplanationMathLevel.FULL_FORMULATIONS,
                explanation_structure=ExplanationStructure.FORMAL_FIRST,
                explanation_length=ExplanationLength.COMPREHENSIVE    
            )
        else:
            return ExplanationStyle(
                explanation_style=ExplanationStyleEnum.INTERMEDIATE,
                explanation_tone=ExplanationTone.BALANCED,
                explanation_complexity=ExplanationComplexity.TECHNICAL,
                explanation_math_level=ExplanationMathLevel.KEY_EQUATIONS,
                explanation_structure=ExplanationStructure.FORMAL_FIRST,
                explanation_length=ExplanationLength.DETAILED    
            )
        

class ExplanationMetadata:
    def __init__(self,
                 estimated_complexity: float,
                 user_knowledge_matched: bool,
                 gap_bridging: bool) -> None:
        self.estimated_complexity = estimated_complexity
        self.user_knowledge_matched = user_knowledge_matched
        self.gap_bridging = gap_bridging
        
class Explanation:
    def __init__(self, text: str, style: Dict, context: Dict, known_concepts: List[str],
                 unknown_concepts: List[str], follow_up_questions: List[str],
                 metadata: ExplanationMetadata):
        self.explanation = text
        self.style = style
        self.known_concepts_used = known_concepts
        self.unknown_concepts_explained = unknown_concepts
        self.follow_up_questions = follow_up_questions
        self.metadata = metadata
        self.context_used = None


class AdaptiveExplainer:
    """
    Generates context-aware explanations based on:
    1. Selected text
    2. Document context
    3. User knowledge level
    4. Explanation depth preference
    """
    
    def __init__(self, llm_client, embedding_model=None, explanation_style: ExplanationStyleEnum = ExplanationStyleEnum.BEGINNER):
        self.llm = llm_client
        self.embedding_model = embedding_model
        self.explanation_style = explanation_style

    def explain(self, text: str, context: Dict, concept_graph: ConceptGraph, user_state: UserKnowledgeState) -> Explanation:
        # Generate explanation using the adaptive method
        explanation_obj = self.generate_explanation(
            selected_text=text,
            context=context,
            concept_graph=concept_graph,
            user_knowledge=user_state,
            depth_preference='adaptive'
        )
        return explanation_obj
    
    def summarize(self, text: str, context: Context) -> Explanation:
        # # Generate summary explanation
        # explanation_obj = self.generate_explanation(
        #     selected_text=text,
        #     context=context,
        #     concept_graph=None,
        #     user_knowledge={},
        #     depth_preference='concise'
        # )
        # return explanation_obj
        return "Summary explanation placeholder"
    
    def explain(self, text: str, context: Context) -> Explanation:
        # # Generate summary explanation
        # explanation_obj = self.generate_explanation(
        #     selected_text=text,
        #     context=context,
        #     concept_graph=None,
        #     user_knowledge={},
        #     depth_preference='concise'
        # )
        # return explanation_obj
        return "Summary explanation placeholder"
    
    
    def ask(self, text: str, context: Context) -> Explanation:
        # # Generate summary explanation
        # explanation_obj = self.generate_explanation(
        #     selected_text=text,
        #     context=context,
        #     concept_graph=None,
        #     user_knowledge={},
        #     depth_preference='concise'
        # )
        # return explanation_obj
        return "Summary explanation placeholder"
    
    def generate_explanation(self, 
                           selected_text: str,
                           context: Dict,
                           concept_graph: ConceptGraph,
                           user_knowledge: UserKnowledgeState,
                           depth_preference: str = 'adaptive') -> Explanation:
        """
        Generate adaptive explanation
        
        Args:
            selected_text: Text selected by user
            context: Dictionary with document context
            user_knowledge: User knowledge state
            depth_preference: 'beginner', 'intermediate', 'advanced', or 'adaptive'
        """
        print(f"Generating {depth_preference} explanation...")
        
        # Determine optimal explanation style
        if depth_preference == 'adaptive':
            explanation_style = self._determine_optimal_style(selected_text, user_knowledge)
        else:
            explanation_style = self.explanation_styles.get(depth_preference, 
                                                          self.explanation_styles['intermediate'])
        
        # Extract relevant context
        relevant_context = self._extract_relevant_context(selected_text, context)
        
        # Identify known/unknown concepts for user
        known_concepts, unknown_concepts = self._identify_concept_gaps(
            selected_text, user_knowledge, context.get('concepts', [])
        )
        
        # Build prompt
        prompt = self._build_explanation_prompt(
            selected_text=selected_text,
            context=relevant_context,
            known_concepts=known_concepts,
            unknown_concepts=unknown_concepts,
            explanation_style=explanation_style
        )
        
        # Generate explanation
        explanation = self._call_llm(prompt)
        
        # Post-process explanation
        processed_explanation = self._post_process_explanation(
            explanation, selected_text, context
        )
        
        # Generate follow-up questions (optional)
        follow_up_questions = self._generate_follow_up_questions(
            selected_text, processed_explanation, user_knowledge
        )
        
        explanation = Explanation(
            text=processed_explanation,
            style=explanation_style,
            context=relevant_context,
            known_concepts=known_concepts,
            unknown_concepts=unknown_concepts,
            follow_up_questions=follow_up_questions,
            metadata=ExplanationMetadata(
                estimated_complexity=self._estimate_complexity(selected_text),
                user_knowledge_matched=len(known_concepts) > 0,
                gap_bridging=len(unknown_concepts) > 0
            ))
        return explanation
        
    
    def _determine_optimal_style(self, text: str, user_knowledge: Dict) -> Dict:
        """Determine optimal explanation style based on text complexity and user knowledge"""
        # Estimate text complexity
        complexity = self._estimate_complexity(text)
        
        # Get user's average knowledge level
        if user_knowledge and 'metrics' in user_knowledge:
            avg_knowledge = user_knowledge['metrics'].get('average_knowledge', 0.5)
        else:
            avg_knowledge = 0.5
        
        # Determine style based on complexity and knowledge
        if complexity > 0.7 and avg_knowledge > 0.7:
            return self.explanation_styles['advanced']
        elif complexity > 0.5 and avg_knowledge > 0.5:
            return self.explanation_styles['intermediate']
        else:
            return self.explanation_styles['beginner']
    
    def _estimate_complexity(self, text: str) -> float:
        """Estimate text complexity (0-1)"""
        # Simple heuristics
        complexity = 0.0
        
        # Length
        words = len(text.split())
        if words > 100:
            complexity += 0.2
        elif words > 50:
            complexity += 0.1
        
        # Technical terms
        technical_terms = ['theorem', 'proof', 'equation', 'algorithm', 
                          'complexity', 'optimization', 'derivative']
        for term in technical_terms:
            if term in text.lower():
                complexity += 0.1
        
        # Math symbols
        math_pattern = r'[\+\-\*/=\^\[\]{}()]'
        if re.search(math_pattern, text):
            complexity += 0.2
        
        # Formal language markers
        formal_markers = ['therefore', 'hence', 'thus', 'moreover', 'furthermore']
        for marker in formal_markers:
            if marker in text.lower():
                complexity += 0.05
        
        return min(1.0, complexity)
    
    def _extract_relevant_context(self, selected_text: str, context: Dict) -> Dict:
        """Extract relevant context from document hierarchy"""
        relevant_chunks = []
        
        if 'hierarchy' in context:
            # Search in sentences and paragraphs
            for chunk_type in ['sentences', 'paragraphs']:
                if chunk_type in context['hierarchy']:
                    for chunk in context['hierarchy'][chunk_type]:
                        # Simple relevance: check for overlapping words
                        selected_words = set(selected_text.lower().split())
                        chunk_words = set(chunk.text.lower().split())
                        overlap = len(selected_words & chunk_words)
                        
                        if overlap > 2:  # At least 3 overlapping words
                            relevant_chunks.append({
                                'text': chunk.text,
                                'type': chunk.chunk_type,
                                'relevance': overlap / len(selected_words)
                            })
        
        # Get section containing the text
        section_context = ""
        if 'hierarchy' in context and 'sections' in context['hierarchy']:
            for section in context['hierarchy']['sections']:
                if selected_text in section.text:
                    section_context = section.text[:500] + "..."
                    break
        
        return {
            'relevant_chunks': relevant_chunks[:3],  # Top 3 most relevant
            'section_context': section_context,
            'summary': self._summarize_context(relevant_chunks)
        }
    
    def _summarize_context(self, relevant_chunks: List[Dict]) -> str:
        """Create summary of relevant context"""
        if not relevant_chunks:
            return "No specific context found."
        
        texts = [chunk['text'][:200] for chunk in relevant_chunks[:2]]
        return "Context includes: " + "... ".join(texts)
    
    def _identify_concept_gaps(self, text: str, user_knowledge: Dict, 
                              all_concepts: List[str]) -> Tuple[List[str], List[str]]:
        """Identify known and unknown concepts in the text"""
        known_concepts = []
        unknown_concepts = []
        
        if not user_knowledge or 'known_concepts' not in user_knowledge:
            return known_concepts, unknown_concepts
        
        # Get user's known concepts
        user_known = [c['concept'] for c in user_knowledge.get('known_concepts', [])]
        
        # Check each concept in text
        text_lower = text.lower()
        for concept in all_concepts:
            concept_lower = concept.lower()
            if concept_lower in text_lower:
                if concept in user_known:
                    known_concepts.append(concept)
                else:
                    unknown_concepts.append(concept)
        
        return known_concepts[:10], unknown_concepts[:10]  # Limit to 10 each
    
    def _build_explanation_prompt(self, selected_text: str, context: Dict,
                                 known_concepts: List[str], unknown_concepts: List[str],
                                 explanation_style: Dict) -> str:
        """Build detailed prompt for explanation generation"""
        
        prompt_template = """
        You are an expert tutor explaining complex concepts to a student.
        
        TASK: Explain the following selected text in context.
        
        SELECTED TEXT:
        {selected_text}
        
        DOCUMENT CONTEXT:
        {context_summary}
        
        STUDENT'S BACKGROUND:
        - Already understands: {known_concepts}
        - Needs explanation for: {unknown_concepts}
        
        EXPLANATION STYLE:
        - Tone: {tone}
        - Complexity: {complexity}
        - Math level: {math_level}
        - Structure: {structure}
        - Length: {length}
        
        INSTRUCTIONS:
        1. Connect to concepts the student already knows
        2. Clearly explain any unknown concepts
        3. Use the specified explanation style
        4. Reference the document context when relevant
        5. Provide an intuitive understanding first, then add details
        6. Use analogies or examples if helpful
        7. If there are mathematical elements, explain their meaning
        
        Generate a clear, helpful explanation:
        """
        
        return prompt_template.format(
            selected_text=selected_text,
            context_summary=context.get('summary', 'No additional context'),
            known_concepts=", ".join(known_concepts) if known_concepts else "No specific prior knowledge",
            unknown_concepts=", ".join(unknown_concepts) if unknown_concepts else "All concepts are familiar",
            tone=explanation_style['tone'],
            complexity=explanation_style['complexity'],
            math_level=explanation_style['math_level'],
            structure=explanation_style['structure'],
            length=explanation_style['length']
        )
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with the prompt"""
        try:
            # Using OpenAI API as example
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful tutor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except:
            # Fallback for demo
            return f"Explanation of the selected text would connect it to the document context. " \
                   f"The key concepts would be explained based on the user's knowledge level."
    
    def _post_process_explanation(self, explanation: str, selected_text: str, 
                                 context: Dict) -> str:
        """Post-process explanation to add references and improve clarity"""
        
        # Add cross-references if available
        if 'hierarchy' in context and 'sections' in context['hierarchy']:
            sections = context['hierarchy']['sections']
            if sections:
                # Reference the containing section
                for section in sections:
                    if selected_text in section.text:
                        explanation += f"\n\n(This concept relates to '{section.metadata.get('title', 'this section')}' in the document.)"
                        break
        
        # Ensure explanation ends with a helpful note
        if not explanation.strip().endswith(('.', '!', '?')):
            explanation += "."
        
        return explanation
    
    def _generate_follow_up_questions(self, text: str, explanation: str, 
                                     user_knowledge: Dict) -> List[str]:
        """Generate follow-up questions to test understanding"""
        
        prompt = f"""
        Based on this text and explanation, generate 2-3 follow-up questions 
        to test understanding at different levels.
        
        Text: {text[:200]}
        Explanation: {explanation[:300]}
        
        Generate questions that:
        1. Test basic comprehension
        2. Test application of the concept
        3. Test deeper understanding (optional)
        
        Format as a numbered list.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are creating assessment questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )
            
            questions_text = response.choices[0].message.content
            # Parse numbered list
            questions = []
            for line in questions_text.split('\n'):
                if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 10)):
                    question = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                    if question:
                        questions.append(question)
            
            return questions[:3]  # Return up to 3 questions
        
        except:
            return [
                "Can you summarize the main idea in your own words?",
                "How would you apply this concept to a different situation?",
                "What part of this explanation was most helpful?"
            ]
    
    def generate_prerequisite_material(self, target_concept: str, 
                                      knowledge_gaps: List[Dict]) -> Dict:
        """Generate prerequisite learning material for identified gaps"""
        
        if not knowledge_gaps:
            return {"message": "No significant knowledge gaps identified."}
        
        # Focus on the biggest gap
        main_gap = knowledge_gaps[0]
        
        prompt = f"""
        Create a micro-lesson to teach this concept as a prerequisite for learning {target_concept}.
        
        Concept to teach: {main_gap['concept']}
        Target concept: {target_concept}
        
        Create a brief lesson that includes:
        1. A simple, intuitive definition
        2. 1-2 clear examples
        3. How this concept relates to {target_concept}
        4. A practice question with answer
        
        Keep it concise but complete.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are creating prerequisite learning material."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=400
            )
            
            lesson = response.choices[0].message.content
            
            # Suggest resources
            resources = self._suggest_resources(main_gap['concept'])
            
            return {
                'gap_concept': main_gap['concept'],
                'knowledge_gap': main_gap['knowledge_gap'],
                'micro_lesson': lesson,
                'suggested_resources': resources,
                'estimated_study_time': '10-15 minutes'
            }
        
        except:
            return {
                'gap_concept': main_gap['concept'],
                'message': f"Study the concept '{main_gap['concept']}' before tackling '{target_concept}'.",
                'suggested_action': "Review basic materials on this topic."
            }
    
    def _suggest_resources(self, concept: str) -> List[Dict]:
        """Suggest learning resources for a concept"""
        # In practice, this would query a resource database
        # For demo, return generic suggestions
        
        return [
            {
                'type': 'video',
                'title': f'Introduction to {concept}',
                'source': 'Khan Academy or YouTube educational channels',
                'duration': '5-10 minutes'
            },
            {
                'type': 'article',
                'title': f'Understanding {concept}: A Beginner\'s Guide',
                'source': 'Wikipedia or educational blogs',
                'reading_time': '5 minutes'
            },
            {
                'type': 'interactive',
                'title': f'{concept} Practice Exercises',
                'source': 'Online learning platforms',
                'activities': 'Quizzes and interactive examples'
            }
        ]