from langchain_core.prompts import PromptTemplate

# explain_prompt = PromptTemplate(
#     template=""" 
#         You are an expert tutor explaining complex concepts to a student.
        
#         TASK: Explain the following selected text in context.
        
#         SELECTED TEXT:
#         {selected_text}
        
#         DOCUMENT CONTEXT:
#         {context_summary}
        
#         STUDENT'S BACKGROUND:
#         - Already understands: {known_concepts}
#         - Needs explanation for: {unknown_concepts}
        
#         EXPLANATION STYLE:
#         - Tone: {tone}
#         - Complexity: {complexity}
#         - Math level: {math_level}
#         - Structure: {structure}
#         - Length: {length}
        
#         INSTRUCTIONS:
#         1. Connect to concepts the student already knows
#         2. Clearly explain any unknown concepts
#         3. Use the specified explanation style
#         4. Reference the document context when relevant
#         5. Provide an intuitive understanding first, then add details
#         6. Use analogies or examples if helpful
#         7. If there are mathematical elements, explain their meaning
        
#         Generate a clear, helpful explanation: in the following format:
#         {format_instructions}
#         """,
#     input_variables=[
#         "selected_text",
#         "context_summary",
#         "known_concepts",
#         "unknown_concepts",
#         "tone",
#         "complexity",
#         "math_level",
#         "structure",
#         "length",
#     ],
#     partial_variables={
#         "format_instructions": explanation_output_parser.get_format_instructions()
#     },
# )


# # --- EXPLAIN PROMPT ---
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
    
    {format_instructions}
    """,
    input_variables=["selected_text", "context_summary", "known_concepts", "unknown_concepts", "tone", "complexity", "math_level"],
)

# --- SUMMARIZE PROMPT ---
summarize_prompt = PromptTemplate(
    template="""
    You are an expert tutor providing a high-level overview.
    
    TASK: Summarize the following passage efficiently for a student.
    TEXT: {selected_text}
    CONTEXT: {context_summary}
    STUDENT BACKGROUND: {known_concepts}
    STYLE: Complexity={complexity}, Structure={structure}, Length={length}
    
    INSTRUCTIONS:
    1. Focus on the core thesis and key takeaways.
    2. Omit minor details unless they are crucial for the context summary.
    3. Match the student's background level.
    
    {format_instructions}
    """,
    input_variables=["selected_text", "context_summary", "known_concepts", "complexity", "structure", "length"],
)

# --- ASK (Q&A) PROMPT ---
ask_prompt = PromptTemplate(
    template="""
    You are an expert tutor answering a specific question about a document.
    
    QUESTION: {question}
    RELEVANT TEXT: {selected_text}
    CONTEXT: {context_summary}
    STUDENT BACKGROUND: {known_concepts}
    
    INSTRUCTIONS:
    1. Answer the question directly using the relevant text and context.
    2. If the answer isn't in the text, use your expert knowledge but state you are adding external info.
    3. Frame the answer based on what the student already knows.
    
    {format_instructions}
    """,
    input_variables=["question", "selected_text", "context_summary", "known_concepts"],
)

# --- CONCEPT RELATIONSHIP PROMPT ---
relation_extractor_prompt = PromptTemplate(
    template="""
        You are constructing a prerequisite graph for learning.

        Concepts:
        {concept_names}

        Previous section: {context}

        Text:
        {text}

        For each concept, answer:
        "To understand X, what other concepts from the list must be understood first?"

        Only use given concepts.
        Only create meaningful prerequisite relationships.

        Output JSON:
        [
        {{
            "source": "prerequisite_concept",
            "target": "dependent_concept",
            "relation": "prerequisite_of"
        }}
        ]
        
    """,
    input_variables=["text", "concept_names", "context"],
)
# # --- CONCEPT RELATIONSHIP PROMPT ---
# relation_extractor_prompt = PromptTemplate(
#     template="""
#     TASK: Identify semantic relationships between technical concepts based on the provided text.
    
#     TEXT: "{text}"
#     CONCEPTS TO EVALUATE: {concept_names}
#     CONTEXT: "{context}"
#     VALID RELATION TYPES: {valid_relations}

#     INSTRUCTIONS:
#     1. Only use concepts from the provided list.
#     2. Assign a relation type strictly from the VALID RELATION TYPES list.
#     3. Determine direction: Identify which concept is the 'source' (cause/parent) and which is the 'target' (effect/child).
#     4. Return JSON ONLY. Do not include conversational filler or markdown code blocks.

#     FORMAT:
#     [
#         {{
#             "src": "concept_a", 
#             "tgt": "concept_b", 
#             "type": "relation_from_list", 
#             "why": "brief technical explanation"
#         }}
#     ]
#     """,
#     input_variables=["text", "concept_names", "context", "valid_relations"],
# )

# --- CONCEPT Canonicalization PROMPT ---
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
        input_variables=["concepts"],)

# --- CONCEPT EXTRACTION PROMPT ---
concept_extraction_template = PromptTemplate(
    template="""
        Extract the key technical concepts from the following text. 
        Return only the concepts as a comma-separated list.

        TEXT:
        {text}
        
        CONTEXT: "{context}"

        CONCEPTS:
        """,
        input_variables=["text","context"],)


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
        input_variables=["candidates","context"],)