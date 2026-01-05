import streamlit as st
from typing import Optional
import json
from datetime import datetime

class DocExplainerApp:
    """
    Main Streamlit application for DocExplainer
    """
    
    def __init__(self):
        st.set_page_config(
            page_title="DocExplainer - Context-Aware Document Tutor",
            page_icon="📘",
            layout="wide"
        )
        
        # Initialize session state
        if 'document_loaded' not in st.session_state:
            st.session_state.document_loaded = False
        if 'user_model' not in st.session_state:
            st.session_state.user_model = None
        if 'current_document' not in st.session_state:
            st.session_state.current_document = None
        if 'concept_graph' not in st.session_state:
            st.session_state.concept_graph = None
        if 'voice_active' not in st.session_state:
            st.session_state.voice_active = False
        
        # Initialize components
        self.init_components()
    
    def init_components(self):
        """Initialize all ML components"""
        from core.document_processor import HierarchicalDocumentProcessor
        from core.knowledge_graph import ConceptGraphBuilder
        from core.user_model import BayesianKnowledgeTracer
        from core.adaptive_explainer import AdaptiveExplainer
        from ui.voice_interface import VoiceInterface
        
        # Initialize with placeholders
        self.doc_processor = HierarchicalDocumentProcessor()
        self.graph_builder = ConceptGraphBuilder()
        self.user_model = BayesianKnowledgeTracer()
        self.explainer = AdaptiveExplainer(llm_client=None)  # Would connect to actual LLM
        self.voice_interface = VoiceInterface(self.explainer, self.user_model)
    
    def run(self):
        """Main application loop"""
        st.title("📘 DocExplainer - Context-Aware Document Tutor")
        st.markdown("An AI-augmented document viewer with adaptive explanations")
        
        # Sidebar
        with st.sidebar:
            st.header("Document Management")
            uploaded_file = st.file_uploader("Upload Document", type=['txt', 'pdf', 'md'])
            
            if uploaded_file:
                self.load_document(uploaded_file)
            
            st.header("User Profile")
            if st.session_state.user_model:
                user_profile = st.session_state.user_model.get_user_profile()
                st.metric("Concepts Tracked", user_profile['metrics']['total_concepts_tracked'])
                st.metric("Average Knowledge", f"{user_profile['metrics']['average_knowledge']:.1%}")
                st.metric("Total Interactions", user_profile['metrics']['total_interactions'])
            
            st.header("Voice Interface")
            voice_toggle = st.toggle("Enable Voice", st.session_state.voice_active)
            if voice_toggle != st.session_state.voice_active:
                st.session_state.voice_active = voice_toggle
                if voice_toggle:
                    st.info("Voice interface active. Say 'Explain this' or ask a question.")
                else:
                    st.info("Voice interface disabled.")
        
        # Main content area
        if st.session_state.document_loaded:
            self.display_document_explorer()
        else:
            self.display_welcome_screen()
    
    def load_document(self, uploaded_file):
        """Process uploaded document"""
        import PyPDF2  # For PDF processing
        
        try:
            # Read file content
            if uploaded_file.type == 'application/pdf':
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            else:
                text = uploaded_file.read().decode('utf-8')
            
            # Process document
            with st.spinner("Processing document..."):
                # Create hierarchical structure
                doc_structure = self.doc_processor.process_document(text, uploaded_file.name)
                
                # Extract concepts and build knowledge graph
                concept_data = self.graph_builder.extract_concepts_from_document(doc_structure['chunks'])
                
                # Initialize user model with concepts
                concept_list = [c['concept'] for c in concept_data['concepts']]
                self.user_model.initialize_user(concept_list)
                
                # Store in session state
                st.session_state.current_document = {
                    'structure': doc_structure,
                    'concepts': concept_data,
                    'text': text,
                    'filename': uploaded_file.name,
                    'loaded_at': datetime.now()
                }
                st.session_state.concept_graph = self.graph_builder
                st.session_state.user_model = self.user_model
                st.session_state.document_loaded = True
            
            st.success(f"Document loaded: {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Error loading document: {e}")
    
    def display_welcome_screen(self):
        """Display welcome screen when no document loaded"""
        st.markdown("""
        ## Welcome to DocExplainer! 🎓
        
        **DocExplainer** is an AI-powered document tutor that:
        
        🔹 **Explains selected text** in context
        🔹 **Understands the full document** structure
        🔹 **Models your knowledge** and adapts explanations
        🔹 **Recommends prerequisites** when needed
        🔹 **Supports voice interaction**
        
        ### How to use:
        1. Upload a document (PDF, TXT, or Markdown)
        2. Select text you want explained
        3. Choose explanation depth
        4. Ask follow-up questions
        
        ### Features:
        - **Adaptive explanations**: Beginner to advanced levels
        - **Knowledge tracking**: Bayesian model of your understanding
        - **Concept graphs**: Visualize relationships between ideas
        - **Prerequisite detection**: Identify and fill knowledge gaps
        - **Voice interface**: Ask questions naturally
        
        Upload a document to get started!
        """)
        
        # Example documents
        st.subheader("Try with sample content:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Machine Learning Basics"):
                sample_text = """
                Machine learning is a subset of artificial intelligence that enables 
                computers to learn from data without explicit programming.
                
                Supervised learning involves training a model on labeled data, 
                where each example has an input and a desired output.
                
                Unsupervised learning finds patterns in unlabeled data, 
                such as clustering similar data points together.
                """
                st.session_state.sample_text = sample_text
        
        with col2:
            if st.button("Neural Networks"):
                sample_text = """
                Neural networks are computing systems inspired by biological neurons.
                
                A neural network consists of layers of interconnected nodes (neurons).
                Each connection has a weight that adjusts during training.
                
                Backpropagation is the algorithm used to train neural networks 
                by propagating errors backward through the network.
                """
                st.session_state.sample_text = sample_text
        
        with col3:
            if st.button("Statistics Fundamentals"):
                sample_text = """
                Statistics is the science of collecting, analyzing, and interpreting data.
                
                The mean is the average of a dataset, calculated by summing all values 
                and dividing by the number of values.
                
                Standard deviation measures the dispersion of data points 
                around the mean, indicating how spread out the data is.
                """
                st.session_state.sample_text = sample_text
        
        if 'sample_text' in st.session_state:
            st.text_area("Sample Document", st.session_state.sample_text, height=200)
            if st.button("Load Sample"):
                # Process sample text
                doc_structure = self.doc_processor.process_document(
                    st.session_state.sample_text, 
                    "Sample Document"
                )
                st.session_state.current_document = {
                    'structure': doc_structure,
                    'text': st.session_state.sample_text,
                    'filename': "Sample Document"
                }
                st.session_state.document_loaded = True
                st.rerun()
    
    def display_document_explorer(self):
        """Main document explorer interface"""
        doc = st.session_state.current_document
        
        # Create two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header(doc['filename'])
            
            # Document navigation
            tab1, tab2, tab3 = st.tabs(["📄 Full Text", "🧭 Hierarchy", "📊 Concepts"])
            
            with tab1:
                # Display document with selection capability
                selected_text = st.text_area(
                    "Document Text",
                    doc['text'],
                    height=400,
                    key="document_text"
                )
                
                # Get selected text (simplified - in production would use JS)
                selected = st.text_input(
                    "Selected text to explain (copy-paste here):",
                    placeholder="Select text from above and paste here..."
                )
                
                if selected and len(selected) > 10:
                    self.display_explanation_panel(selected)
            
            with tab2:
                # Display document hierarchy
                st.subheader("Document Structure")
                if 'structure' in doc and 'hierarchy' in doc['structure']:
                    hierarchy = doc['structure']['hierarchy']
                    
                    for section in hierarchy.get('sections', []):
                        with st.expander(f"📑 {section.metadata.get('title', 'Section')}"):
                            st.write(section.text[:500] + "...")
                            
                            # Show paragraphs in this section
                            paragraphs = [p for p in hierarchy.get('paragraphs', []) 
                                         if p.parent_id == section.chunk_id]
                            for para in paragraphs[:3]:  # Show first 3 paragraphs
                                st.caption(f"Paragraph: {para.text[:200]}...")
            
            with tab3:
                # Display concept graph
                st.subheader("Concept Knowledge Graph")
                if st.session_state.concept_graph:
                    fig = st.session_state.concept_graph.visualize_graph(max_nodes=20)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show concept list
                    if 'concepts' in doc and 'concepts' in doc['concepts']:
                        concepts = doc['concepts']['concepts'][:20]  # Top 20
                        st.write("**Key Concepts:**")
                        for concept_data in concepts:
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"• {concept_data['concept']}")
                            with col_b:
                                st.write(f"Score: {concept_data['score']:.2f}")
        
        with col2:
            self.display_user_panel()
    
    def display_explanation_panel(self, selected_text: str):
        """Display explanation and interaction panel"""
        st.subheader("🤔 Explanation Request")
        
        # Explanation settings
        col_settings1, col_settings2 = st.columns(2)
        
        with col_settings1:
            depth = st.radio(
                "Explanation Depth",
                ["Adaptive", "Beginner", "Intermediate", "Advanced"],
                horizontal=True
            )
        
        with col_settings2:
            include_context = st.checkbox("Include document context", True)
            generate_questions = st.checkbox("Generate follow-up questions", True)
        
        # Get user knowledge
        user_profile = st.session_state.user_model.get_user_profile()
        
        # Prepare context
        context = st.session_state.current_document['structure']
        
        # Generate explanation
        if st.button("Generate Explanation", type="primary"):
            with st.spinner("Generating adaptive explanation..."):
                explanation_result = self.explainer.generate_explanation(
                    selected_text=selected_text,
                    context=context,
                    user_knowledge=user_profile,
                    depth_preference=depth.lower()
                )
                
                # Display explanation
                st.markdown("### 📝 Explanation")
                st.write(explanation_result['explanation'])
                
                # Display metadata
                with st.expander("Explanation Details"):
                    st.write(f"**Style:** {explanation_result['style']['tone']}")
                    st.write(f"**Known concepts used:** {', '.join(explanation_result['known_concepts_used'][:3])}")
                    st.write(f"**New concepts explained:** {', '.join(explanation_result['unknown_concepts_explained'][:3])}")
                
                # Follow-up questions
                if generate_questions and explanation_result['follow_up_questions']:
                    st.markdown("### ❓ Follow-up Questions")
                    for i, question in enumerate(explanation_result['follow_up_questions'], 1):
                        st.write(f"{i}. {question}")
                
                # Knowledge gap analysis
                st.markdown("### 🎯 Knowledge Gap Analysis")
                
                # Extract main concept
                main_concept = self.explainer._extract_main_concept(selected_text)
                
                # Find prerequisites
                if st.session_state.concept_graph:
                    prerequisites = st.session_state.concept_graph.find_prerequisites(
                        main_concept,
                        set(st.session_state.user_model.get_known_concepts(threshold=0.7))
                    )
                    
                    if prerequisites:
                        st.warning(f"**Prerequisites needed for '{main_concept}':**")
                        for prereq in prerequisites[:3]:
                            st.write(f"- {prereq['concept']} (gap: {1 - prereq.get('current_knowledge', 0):.1%})")
                        
                        # Generate micro-lesson for biggest gap
                        if st.button("Generate Micro-Lesson for Biggest Gap"):
                            gap_result = self.explainer.generate_prerequisite_material(
                                main_concept, prerequisites
                            )
                            st.info(f"**Micro-lesson for {gap_result['gap_concept']}:**")
                            st.write(gap_result.get('micro_lesson', gap_result.get('message', '')))
                    else:
                        st.success("No significant knowledge gaps detected!")
                
                # Voice explanation option
                if st.session_state.voice_active:
                    if st.button("🎤 Hear Explanation"):
                        self.voice_interface.speak_explanation(explanation_result['explanation'])
    
    def display_user_panel(self):
        """Display user profile and knowledge panel"""
        st.header("👤 Learning Profile")
        
        if st.session_state.user_model:
            user_profile = st.session_state.user_model.get_user_profile()
            
            # Knowledge metrics
            st.metric("Mastered Concepts", len(user_profile['known_concepts']))
            st.metric("Learning Concepts", len(user_profile['learning_concepts']))
            st.metric("Knowledge Gaps", len(user_profile['unknown_concepts']))
            
            # Knowledge distribution
            st.subheader("Knowledge Distribution")
            
            # Create donut chart
            import plotly.graph_objects as go
            
            labels = ['Mastered', 'Learning', 'Unknown']
            values = [
                len(user_profile['known_concepts']),
                len(user_profile['learning_concepts']),
                len(user_profile['unknown_concepts'])
            ]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.3,
                marker_colors=['#00CC96', '#FFA15A', '#EF553B']
            )])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Known concepts
            with st.expander("✅ Mastered Concepts"):
                for concept in user_profile['known_concepts'][:5]:
                    st.write(f"• {concept['concept']} ({concept['knowledge']:.0%})")
            
            # Learning concepts
            with st.expander("📚 Currently Learning"):
                for concept in user_profile['learning_concepts'][:5]:
                    st.write(f"• {concept['concept']} ({concept['knowledge']:.0%})")
            
            # Knowledge gaps
            with st.expander("⚠️ Knowledge Gaps"):
                for concept in user_profile['unknown_concepts'][:5]:
                    st.write(f"• {concept['concept']} ({concept['knowledge']:.0%})")
            
            # Learning recommendations
            st.subheader("🎯 Recommended Learning")
            
            if 'concepts' in st.session_state.current_document:
                # Get top concept from document
                top_concept = st.session_state.current_document['concepts']['concepts'][0]['concept']
                
                # Get learning path
                learning_path = st.session_state.user_model.recommend_learning_path(
                    top_concept,
                    st.session_state.concept_graph
                )
                
                if learning_path:
                    st.write(f"**Path to understand '{top_concept}':**")
                    for i, step in enumerate(learning_path[:3], 1):
                        st.write(f"{i}. {step['concept']} (priority: {step['priority']:.2f})")
                else:
                    st.info("You're ready to tackle the main concepts!")
            
            # Voice interface
            if st.session_state.voice_active:
                st.subheader("🎤 Voice Interface")
                
                if st.button("Ask a Question"):
                    query = self.voice_interface.listen_for_question(timeout=5)
                    if query:
                        st.write(f"**You asked:** {query}")
                        
                        # Process query
                        result = self.voice_interface.process_voice_query(
                            query,
                            st.session_state.current_document['structure']
                        )
                        
                        if result['success']:
                            st.write(f"**Explanation:** {result['explanation'][:200]}...")
                            
                            if st.button("Hear Response"):
                                self.voice_interface.speak_explanation(result['explanation'])
        
        else:
            st.info("Upload a document to start building your knowledge profile.")

def run():
    """Run the Streamlit DocExplainer application"""
    app = DocExplainerApp()
    app.run()
    
# Run the application
if __name__ == "__main__":
    app = DocExplainerApp()
    app.run()