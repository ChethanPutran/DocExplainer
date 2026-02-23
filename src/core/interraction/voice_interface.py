import speech_recognition as sr
import pyttsx3
import queue
import threading
from typing import Optional, Callable
import sounddevice as sd
import numpy as np
import whisper
from src.core.explanation_engine.adaptive_explainer import AdaptiveExplainer
from src.core.knowlege_modelling.knowledge_tracing import BayesianKnowledgeTracer

class VoiceInterface:
    """
    Voice interface for asking questions and receiving explanations
    """
    
    def __init__(self, explainer: AdaptiveExplainer, user_model: BayesianKnowledgeTracer):
        self.explainer = explainer
        self.user_model = user_model
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.whisper_model = whisper.load_model("base")  # Small model for speed
        
        # Configure TTS
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.9)
        
        # Audio queue for real-time processing
        self.audio_queue = queue.Queue()
        self.is_listening = False
        
    def listen_for_question(self, timeout: int = 5) -> Optional[str]:
        """
        Listen for user question with timeout
        """
        print("Listening for question... (speak now)")
        
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen with timeout
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                
                # Convert speech to text
                try:
                    # Try Google Speech Recognition first
                    text = self.recognizer.recognize_google(audio)
                    print(f"You said: {text}")
                    return text
                except sr.UnknownValueError:
                    # Fall back to Whisper
                    print("Google didn't understand, trying Whisper...")
                    return self._whisper_transcribe(audio)
                
        except sr.WaitTimeoutError:
            print("Listening timeout")
            return None
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None
    
    def _whisper_transcribe(self, audio) -> str:
        """Transcribe audio using Whisper"""
        # Convert AudioData to numpy array
        audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        result = self.whisper_model.transcribe(audio_data, fp16=False)
        return result['text']
    
    def speak_explanation(self, explanation: str):
        """Convert text explanation to speech"""
        print("Speaking explanation...")
        
        # Clean up explanation for speech
        speech_text = self._prepare_for_speech(explanation)
        
        # Speak
        self.tts_engine.say(speech_text)
        self.tts_engine.runAndWait()
    
    def _prepare_for_speech(self, text: str) -> str:
        """Prepare text for TTS by cleaning and adding pauses"""
        # Remove markdown and special characters
        text = re.sub(r'[#*_`]', '', text)
        
        # Add pauses after periods and commas
        text = text.replace('.', '. ')
        text = text.replace(',', ', ')
        
        # Limit length for TTS
        if len(text.split()) > 100:
            sentences = text.split('. ')
            text = '. '.join(sentences[:3]) + '. '  # First 3 sentences
        
        return text
    
    def process_voice_query(self, query: str, current_context: Dict) -> Dict:
        """
        Process voice query and generate voice response
        """
        # Classify query type
        query_type = self._classify_query(query)
        
        # Extract target from query
        target_text = self._extract_target_from_query(query, current_context)
        
        if not target_text:
            return {
                'success': False,
                'response': "I couldn't identify what you want me to explain. Please try again.",
                'query_type': query_type
            }
        
        # Get user knowledge state
        user_profile = self.user_model.get_user_profile()
        
        # Generate explanation
        explanation_result = self.explainer.generate_explanation(
            selected_text=target_text,
            context=current_context,
            user_knowledge=user_profile,
            depth_preference='adaptive'
        )
        
        # Update user model with this interaction
        self.user_model.update_from_interaction(
            concept=self._extract_main_concept(target_text),
            response_data={
                'correct': None,  # Unknown for voice queries
                'time_spent': 30,  # Estimate
                'explanation_depth': 'adaptive',
                'asked_question': True
            }
        )
        
        return {
            'success': True,
            'query': query,
            'query_type': query_type,
            'target': target_text[:100] + "...",
            'explanation': explanation_result['explanation'],
            'follow_up_questions': explanation_result['follow_up_questions'],
            'text_for_speech': self._prepare_for_speech(explanation_result['explanation'])
        }
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of voice query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['explain', 'what is', 'tell me about']):
            return 'explanation'
        elif any(word in query_lower for word in ['why', 'how come', 'reason']):
            return 'reasoning'
        elif any(word in query_lower for word in ['example', 'for instance']):
            return 'example'
        elif any(word in query_lower for word in ['simplify', 'easier', 'beginner']):
            return 'simplification'
        elif any(word in query_lower for word in ['again', 'repeat']):
            return 'repetition'
        else:
            return 'general'
    
    def _extract_target_from_query(self, query: str, context: Dict) -> str:
        """Extract target text from voice query"""
        # Simple keyword extraction
        keywords = ['explain', 'what is', 'tell me about', 'mean by', 'meaning of']
        
        for keyword in keywords:
            if keyword in query.lower():
                # Extract text after keyword
                parts = query.lower().split(keyword)
                if len(parts) > 1:
                    target_phrase = parts[1].strip()
                    
                    # Try to find this phrase in the document
                    if 'hierarchy' in context:
                        # Search in sentences
                        for chunk_type in ['sentences', 'paragraphs']:
                            if chunk_type in context['hierarchy']:
                                for chunk in context['hierarchy'][chunk_type]:
                                    if target_phrase in chunk.text.lower():
                                        return chunk.text
        
        # If no specific target found, return recent context
        if 'hierarchy' in context and 'sentences' in context['hierarchy']:
            sentences = context['hierarchy']['sentences']
            if sentences:
                return sentences[-1].text  # Most recent sentence
        
        return query  # Fallback to using the query itself
    
    def _extract_main_concept(self, text: str) -> str:
        """Extract main concept from text"""
        # Simple extraction - first noun phrase
        words = text.split()
        if len(words) > 0:
            return words[0]  # Very simple, would use NER in production
        return "unknown"
    
    def start_realtime_listening(self, callback: Callable):
        """Start real-time listening in a separate thread"""
        self.is_listening = True
        
        def listen_loop():
            while self.is_listening:
                query = self.listen_for_question(timeout=10)
                if query:
                    callback(query)
        
        thread = threading.Thread(target=listen_loop, daemon=True)
        thread.start()
    
    def stop_realtime_listening(self):
        """Stop real-time listening"""
        self.is_listening = False
