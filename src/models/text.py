from abc import abstractmethod
import re
# from sentence_transformers import SentenceTransformer
# import torch
# from transformers import AutoModelForTokenClassification, AutoTokenizer

class EmbeddingModel:
    DEFAULT_MODEL_NAME = 'all-MiniLM-L6-v2'

    def __init__(self,model_name: str = DEFAULT_MODEL_NAME) -> None:
        # self.model = SentenceTransformer(model_name)
        self.model = None

    def encode(self, text: str) -> list[float]:
        if self.model is None:
            raise ValueError("Embedding model is not initialized.")
        return self.model.encode(text).tolist()

class NERLLM:
    def __init__(self, llm_client = None) -> None:
        self.llm = llm_client

    def extract_concepts(self, text:  list[str]) -> list[str]:
        if not self.llm:
            raise ValueError("LLM client is not provided.")
        
        """Extract concepts using LLM"""
        prompt = f"""
        Extract the key concepts from the following text. 
        Return only the concepts as a comma-separated list.
        
        Text: {text[:2000]}
        
        Concepts:
        """
        
        try:
            response = self.llm.generate(prompt)
            concepts = [c.strip() for c in response.split(',')]
            return concepts
        except:
            return []
class TextModels:
    def __init__(self,llm_client = None) -> None:
        self.llm_client = llm_client
    def get_ner_model(self):
        return NERModel()
    def get_ner_regex(self):
        return NERRegex()
    def get_ner_llm(self):
        if self.llm_client is None:
            return None
        return NERLLM(self.llm_client)
    @abstractmethod   
    def extract_concepts(self, text: list[str]) -> list[str]:
        pass
    
class NERRegex():
    def extract_concepts(self, text:  list[str]) -> list[str]:
        """Extract concepts using linguistic patterns"""
        patterns = [
            r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Capitalized phrases
            r'(?:"([^"]+)"\s+(?:is|are|means|refers to))',  # Quoted definitions
            r'(?:\b(?:the|a|an)\s+([A-Za-z-]+\s+(?:of|in|for)\s+[A-Za-z-]+))',  # Noun phrases
        ]
        
        concepts = []
        for pattern in patterns:
            matches = re.finditer(pattern, " ".join(text))
            for match in matches:
                concept = match.group(1) if match.groups() else match.group(0)
                if concept and len(concept.split()) <= 5:  # Limit to 5 words
                    concepts.append(concept.strip())
        
        return list(set(concepts))

class NERModel():
    DEFAULT_TOKENIZER_NAME = "distilbert-base-cased" # Or "dslim/bert-base-NER"
    DEFAULT_MODEL_NAME = "elastic/distilbert-base-cased-finetuned-conll03-english" # Or "dslim/bert-base-NER"
    
    def __init__(self) -> None:
        # self.tokenizer = AutoTokenizer.from_pretrained(self.DEFAULT_TOKENIZER_NAME)
        # self.model = AutoModelForTokenClassification.from_pretrained(self.DEFAULT_MODEL_NAME)
        self.tokenizer = None
        self.model = None
        
    def extract_concepts(self, text: list[str]) -> list[str]:
        if self.tokenizer is None or self.model is None:
            raise ValueError("NER model or tokenizer is not initialized.")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = None
        import torch
        with torch.no_grad():
            outputs = self.model(**inputs).logits

        predictions = torch.argmax(outputs, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        concepts = []
        current_entity = []
        current_label = None
        
        for token, prediction in zip(tokens, predictions[0]):
            label = self.model.config.id2label[prediction.item()]
            
            if label.startswith("B-"):
                if current_entity:
                    concepts.append(" ".join(current_entity))
                current_entity = [token.replace("##", "")]
                current_label = label[2:]
            elif label.startswith("I-") and current_label == label[2:]:
                current_entity.append(token.replace("##", ""))
            else:
                if current_entity:
                    concepts.append(" ".join(current_entity))
                current_entity = []
                current_label = None
        
        if current_entity:
            concepts.append(" ".join(current_entity))
        
        return list(set([c for c in concepts if len(c) > 2]))
        