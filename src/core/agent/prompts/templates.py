from langchain_core.prompts import PromptTemplate

# --- EXPLAIN PROMPT ---
explain_prompt = PromptTemplate(
    template="""
    You are an expert tutor explaining complex concepts.
    
    TASK: Explain the following text.
    
    TEXT: {selected_text}
    CONTEXT: {context_summary}
    STUDENT BACKGROUND: Knows {known_concepts}, Needs {unknown_concepts}
    STYLE: Tone={tone}, Complexity={complexity}, Math={math_level}
    
    INSTRUCTIONS:
    1. Bridge the gap between known and unknown concepts.
    2. Use the specified style and math level.
    3. Provide intuitive understanding first, then details.
    4. Use analogies or examples if helpful.
    
    {format_instructions}
    """,
    input_variables=["selected_text", "context_summary", "known_concepts", 
                     "unknown_concepts", "tone", "complexity", "math_level"],
)

# --- SUMMARIZE PROMPT ---
summarize_prompt = PromptTemplate(
    template="""
    You are an expert tutor providing a high-level overview.
    
    TASK: Summarize the following passage efficiently.
    
    TEXT: {selected_text}
    CONTEXT: {context_summary}
    STUDENT BACKGROUND: {known_concepts}
    STYLE: Complexity={complexity}, Structure={structure}, Length={length}
    
    INSTRUCTIONS:
    1. Focus on core thesis and key takeaways.
    2. Omit minor details unless crucial.
    3. Match the student's background level.
    4. Use bullet points if appropriate.
    
    {format_instructions}
    """,
    input_variables=["selected_text", "context_summary", "known_concepts", 
                     "complexity", "structure", "length"],
)

# --- ASK (Q&A) PROMPT ---
ask_prompt = PromptTemplate(
    template="""
    You are an expert tutor answering a specific question.
    
    QUESTION: {question}
    RELEVANT TEXT: {selected_text}
    CONTEXT: {context_summary}
    STUDENT BACKGROUND: {known_concepts}
    
    INSTRUCTIONS:
    1. Answer directly using relevant text and context.
    2. If answer isn't in text, use expert knowledge but state it's external.
    3. Frame answer based on what student already knows.
    4. Be concise but thorough.
    
    {format_instructions}
    """,
    input_variables=["question", "selected_text", "context_summary", "known_concepts"],
)

# --- CONCEPT RELATIONSHIP PROMPT ---
relation_extractor_prompt = PromptTemplate(
    template="""
    You are an Ontological Engineer building a knowledge graph.
    
    ### Goal
    Extract structured relationships between specific concepts.
    
    ### Provided Concepts
    {concept_names}
    
    ### Context
    {context}

    ### Source Text
    {text}

    ### Allowed Relations
    - "depends_on": Use for "uses", "relies_on", "based_on", "built_on"
    - "is_a": Use for inheritance, categorization, "type of"
    - "part_of": Use for composition (A is component of B)
    - "enables": Use when A makes B possible but isn't strict dependency
    - "similar_to": Use for analogous concepts

    ### Output JSON Format:
    [
      {{
        "source": "concept_name",
        "target": "concept_name",
        "relation": "depends_on",
        "strength": 0.85,
        "attributes": {{
          "rationale": "brief reason",
          "context_type": "theoretical/practical"
        }}
      }}
    ]
    """,
    input_variables=["text", "concept_names", "context"],
)

# --- CONCEPT CANONICALIZATION PROMPT ---
concept_canonicalization_template = PromptTemplate(
    template="""
    Group the following concepts into canonical concepts.
    Merge synonyms and abbreviations.

    Concepts:
    {concepts}

    Output JSON:
    {{
      "canonical_name": ["alias1", "alias2"]
    }}
    """,
    input_variables=["concepts"],
)

# --- CONCEPT EXTRACTION PROMPT ---
concept_extraction_template = PromptTemplate(
    template="""
    Extract the key technical concepts from the following text.
    Return only the concepts as a comma-separated list.

    TEXT:
    {text}
    
    CONTEXT:
    {context}

    CONCEPTS:
    """,
    input_variables=["text", "context"],
)

# --- CONCEPT REFINEMENT PROMPT ---
concept_refinement_template = PromptTemplate(
    template="""
    You are building a learning concept graph.

    Clean and refine the following candidate phrases.

    Rules:
    - Remove generic phrases (e.g., "high performance", "this model")
    - Split compound concepts
    - Keep only atomic, teachable concepts
    - Normalize names (lowercase, singular if possible)

    Also generate short definition (1 line) for each.

    Candidates:
    {candidates}

    Context:
    {context}

    Output JSON:
    [
      {{
        "name": "...",
        "definition": "..."
      }}
    ]
    """,
    input_variables=["candidates", "context"],
)

# --- TEXT SUMMARIZATION PROMPT ---
text_summarization_template = PromptTemplate(
    template="""
    Summarize the TEXT into one specific sentence using the CONTEXT for technical accuracy.
    Avoid preamble.

    CONTEXT:
    {recent_context}

    TEXT:
    {current_text}

    SUMMARY:
    """,
    input_variables=["recent_context", "current_text"],
)


# --- Create prompt for LLM question generation ---

question_generation_template = PromptTemplate(
    template="""
    Generate a quiz question about the concept: {concept}
    Difficulty level: {difficulty}

    The question should:
        - Be clear and unambiguous
        - Test understanding of the concept
        - Be appropriate for {difficulty} level learners
        - Include a correct answer
        - Include a brief explanation
        
        Return the response in this JSON format:
        {{
            "question_text": "The question text",
            "question_type": "multiple_choice" or "true_false" or "fill_blank" or "short_answer",
            "options": [
                {{"text": "Option A", "is_correct": false}},
                {{"text": "Option B", "is_correct": true}}
            ] (only for multiple choice),
            "correct_answer": "The correct answer",
            "explanation": "Explanation of the correct answer",
            "hints": ["Hint 1", "Hint 2"]
        }}
        """,
        input_variables=["concept", "difficulty"])