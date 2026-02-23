from typing import Dict, List, Tuple, Any
import re
import json
from src.core.knowlege_modelling.base import Concept, ConceptRelationship
from src.core.agent.prompts import relation_extractor_prompt, concept_extraction_template, concept_refinement_template, concept_canonicalization_template
from src.core.agent.llm_wrapper import LLMWrapper
from collections import defaultdict
from itertools import combinations
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import math

from dataclasses import dataclass, field
from typing import Set, Dict, Optional

@dataclass
class ConceptInvertedEntry:
    section_frequency: Dict[int, int] = field(default_factory=dict)
    paragraph_frequency: Dict[int, int] = field(default_factory=dict)
    first_occurrence: Optional[Tuple[int, int]] = None
    total_frequency: int = 0

class ConceptInvertedIndex:
    def __init__(self):
        self.index: Dict[int, ConceptInvertedEntry] = {}

    def add_occurrence(self, concept_id, section_id, section_order, paragraph_id):

        if concept_id not in self.index:
            self.index[concept_id] = ConceptInvertedEntry()

        entry = self.index[concept_id]

        entry.section_frequency[section_id] = \
            entry.section_frequency.get(section_id, 0) + 1

        entry.paragraph_frequency[paragraph_id] = \
            entry.paragraph_frequency.get(paragraph_id, 0) + 1

        entry.total_frequency += 1

        if entry.first_occurrence is None or section_order < entry.first_occurrence[1]:
            entry.first_occurrence = (section_id, section_order)

class ConceptCanonicalizer:
    def __init__(self):
        self.llm = LLMWrapper(concept_canonicalization_template)
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.concept_store = defaultdict(list)
        self.embedding_cache = {}

    def normalize(self, name: str) -> str:
        name = name.lower().strip()

        # Remove possessives
        name = re.sub(r"'s\b", "", name)

        # Remove parentheses content
        name = re.sub(r"\(.*?\)", "", name)

        # Remove trailing generic words
        name = re.sub(
            r"\b(model|models|method|methods|approach|approaches)\b", "", name)

        # Remove extra spaces
        name = re.sub(r"\s+", " ", name).strip()

        # Lemmatize (important)
        doc = self.nlp(name)
        tokens = []
        for token in doc:
            if not token.is_stop:
                tokens.append(token.lemma_)

        return " ".join(tokens)

    def merge_similar(self, concepts: List[str], threshold: float = 0.85):

        if len(concepts) <= 1:
            return {concepts[0]: concepts}

        embeddings = self.embedder.encode(concepts)

        similarity_matrix = cosine_similarity(embeddings)

        clustering = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=1 - threshold,
            n_clusters=None,
        )

        distance_matrix = 1 - similarity_matrix
        labels = clustering.fit_predict(distance_matrix)

        clusters = defaultdict(list)
        for concept, label in zip(concepts, labels):
            clusters[label].append(concept)

        canonical_map = {}

        for cluster in clusters.values():
            # Choose shortest name as canonical
            canonical = min(cluster, key=len)
            canonical_map[canonical] = cluster

        return canonical_map

    def llm_canonicalize(self, concepts: List[str]):
        try:
            response = self.llm.generate({"concepts": concepts})
            return json.loads(response)
        except Exception as e:
            print(f"LLM canonicalization failed: {e}")
            return {c: [c] for c in concepts}

    def canonicalize_concepts(self, raw_concepts: List[str]):
        normalized_map = defaultdict(list)
        # Step 1: Rule normalize
        for raw in raw_concepts:
            norm = self.normalize(raw)
            normalized_map[norm].append(raw)

        unique_normalized = list(normalized_map.keys())

        # Step 2: Embedding cluster
        embed_clusters = self.merge_similar(unique_normalized)

        # Step 3: LLM refine cluster representatives
        llm_input = list(embed_clusters.keys())

        llm_map = self.llm_canonicalize(llm_input)
        final_map = defaultdict(list)

        # LLM merge
        for new_canonical, old_canonicals in llm_map.items():
            for old in old_canonicals:
                final_map[new_canonical].extend(embed_clusters.get(old, []))

        # Fallback
        for old_canonical, aliases in embed_clusters.items():
            if not any(old_canonical in v for v in llm_map.values()):
                final_map[old_canonical].extend(aliases)

        return final_map


class ConceptExtractor:
    def __init__(self, text_model, concepts_per_para: int = 10) -> None:
        self.llm = LLMWrapper(concept_refinement_template)
        self.concept_embeddings = {}
        self.canonicalizer = ConceptCanonicalizer()
        self.global_concepts: Dict[str, Concept] = {}
        # Initialize different extraction backends from the text_model factory
        self.ner_model = text_model.get_ner_model()
        self.ner_regex = text_model.get_ner_regex()
        self.spacy_extractor = text_model.get_spacy_model()
        self.ner_llm = text_model.get_ner_llm()
        self.concepts_per_para = concepts_per_para
        self.inverted_index = ConceptInvertedIndex()
    
    def get_inverted_index(self)->ConceptInvertedIndex:
        return self.inverted_index


    def _get_or_create_concept(self, name: str) -> Concept:
        """Ensures that the same concept name always returns the same Concept object instance."""

        name_normalized = self.canonicalizer.normalize(name)

        # --- Step 1: Ensure embedding exists in cache ---
        if name_normalized not in self.canonicalizer.embedding_cache:
            self.canonicalizer.embedding_cache[name_normalized] = \
                self.canonicalizer.embedder.encode(name_normalized)

        embedding = self.canonicalizer.embedding_cache[name_normalized]

        # --- Step 2: Create concept if not exists ---
        if name_normalized not in self.global_concepts:
            self.global_concepts[name_normalized] = Concept(
                name=name_normalized,
                embedding=embedding
            )

        concept = self.global_concepts[name_normalized]

        # --- Step 3: Add alias ---
        if name not in concept.aliases:
            concept.aliases.append(name)

        return concept

    def _filter_concepts(
        self,
        concepts: List[Concept],
        context_text: str,
        top: int = 5
    ) -> List[Concept]:
        """
        Rank concepts using frequency, specificity, structural prominence,
        and definition patterns.
        """

        if not concepts:
            return []

        context_lower = context_text.lower()
        text_len = len(context_lower)

        scored_concepts = []

        for concept in concepts:

            raw_freq = 0
            positions = []

            # --- Frequency + Position from ALL aliases ---
            for alias in concept.aliases:
                alias_lower = alias.lower()
                start = 0

                while True:
                    pos = context_lower.find(alias_lower, start)
                    if pos == -1:
                        break
                    raw_freq += 1
                    positions.append(pos)
                    start = pos + len(alias_lower)

            if raw_freq == 0:
                continue

            # --- Frequency Score (log scaled) ---
            frequency_score = math.log1p(raw_freq)

            # --- Specificity (2–3 words preferred) ---
            num_words = len(concept.name.split())
            if num_words == 2:
                length_multiplier = 1.4
            elif num_words == 3:
                length_multiplier = 1.7
            elif num_words >= 4:
                length_multiplier = 0.7
            else:
                length_multiplier = 1.0

            # --- Position Score (earlier = more important) ---
            first_pos = min(positions)
            relative_pos = first_pos / text_len if text_len else 1.0
            position_score = math.exp(-2.5 * relative_pos)

            # --- Definition Bonus (check ALL aliases) ---
            definition_bonus = 1.0

            for alias in concept.aliases:
                alias_lower = alias.lower()
                patterns = [
                    f"{alias_lower} is",
                    f"{alias_lower} are",
                    f"{alias_lower} refers to",
                    f"{alias_lower}:"
                ]
                if any(p in context_lower for p in patterns):
                    definition_bonus = 2.0
                    break

            # --- Final Score ---
            score = (
                frequency_score
                * length_multiplier
                * position_score
                * definition_bonus
            )

            concept.score = round(score, 4)
            concept.frequency = raw_freq

            scored_concepts.append(concept)

        scored_concepts.sort(key=lambda x: x.score, reverse=True)

        return self._prune_subsets(scored_concepts[: top + 5], top)

    def _prune_subsets(self, concepts: List[Concept], top: int) -> List[Concept]:
        """Removes shorter concepts that are substrings of longer, higher-ranked ones."""
        final = []
        concepts.sort(key=lambda x: len(x.name), reverse=True)

        for i, concept in enumerate(concepts):
            is_subset = False
            for j, other in enumerate(concepts):
                if i != j and concept.name in other.name and concept.score < other.score:
                    is_subset = True
                    break
            if not is_subset:
                final.append(concept)

        final.sort(key=lambda x: x.score, reverse=True)
        return final[:top]

    def _llm_refine_concepts(self, candidates: List[str], context: str) -> List[str]:
        """
        Cleans candidate phrases.
        Splits compound concepts.
        Removes generic phrases.
        Returns atomic learning concepts.
        """
        response = self.llm.generate(
            {'candidates': candidates, 'context': context})
        parsed = json.loads(response)

        refined = []
        for item in parsed:
            concept = self._get_or_create_concept(item["name"])
            concept.definitions.append(item["definition"])
            refined.append(concept.name)

        return refined

    def extract(self, text: str, context: str, section_id: int, paragraph_id: int) -> List[Concept]:
        # NLP Layer
        noun_phrases = self.spacy_extractor.extract_noun_phrases(text)
        named_entities = self.spacy_extractor.extract_named_entities(text)

        ner_concepts = self.ner_model.extract_concepts(text)
        pattern_concepts = self.ner_regex.extract_concepts(text)

        # Merge raw candidates
        raw_candidates = list(
            set(noun_phrases + named_entities + ner_concepts + pattern_concepts)
        )

        # LLM REFINEMENT
        refined_concepts = self._llm_refine_concepts(raw_candidates, text)

        # FULL CANONICALIZATION PIPELINE
        canonical_map = self.canonicalizer.canonicalize_concepts(
            refined_concepts)

        all_concepts = []
        context_lower = context.lower()
        for canonical_name, aliases in canonical_map.items():
            concept = self._get_or_create_concept(canonical_name)

            for alias in aliases:
                if alias not in concept.aliases:
                    concept.aliases.append(alias)

                alias_lower = alias.lower()
                start = 0

                while True:
                    pos = context_lower.find(alias_lower, start)
                    if pos == -1:
                        break

                    concept.occurrences.append({
                        "section_id": section_id,
                        "paragraph_id": paragraph_id,
                        "char_start": pos,
                        "char_end": pos + len(alias_lower),
                        "snippet": context[max(0, pos-40):pos+40]
                    })

                    start = pos + len(alias_lower)
            all_concepts.append(concept)

        filtered_concepts = self._filter_concepts(
            all_concepts,
            context_text=text,
            top=self.concepts_per_para
        )

        for concept in filtered_concepts:
            self.inverted_index.add_occurrence(
                concept_id=concept.id,
                section_id=section_id,
                section_order=section_id,
                paragraph_id=paragraph_id
            )


        print(
            f"DEBUG: Extracted {len(all_concepts)} raw, kept {len(filtered_concepts)} after filtering."
        )

        return filtered_concepts


class RelationshipExtractor:
    def __init__(self) -> None:
        self.llm = LLMWrapper(relation_extractor_prompt)
        # Define high-value pedagogical relationships
        # ["is_a", "part_of", "depends_on", "enables", "implements", "results_in"]
        self.global_relations = {}
        self.relation_map = {
            "uses": "depends_on",
            "relies_on": "depends_on",
            "based_on": "depends_on",
            "built_on": "depends_on",
            "is_a": "is_a",
            "part_of": "part_of",
            "enables": "enables",
        }

    def _get_or_create_relationship(self, name: str) -> ConceptRelationship:
        """Ensures that the same relation name always returns the same Relation object instance."""
        name_normalized = name.lower().strip()
        if name_normalized not in self.global_relations:
            self.global_relations[name_normalized] = ConceptRelationship(
                relation=name_normalized)
        return self.global_relations[name_normalized]

    def _build_relations(self, text_concepts: List[Tuple[Concept, str, Concept]], context: str) -> List[ConceptRelationship]:
        relations = []
        for concept1, relation, concept2 in text_concepts:
            relation_type = self.relation_map.get(
                relation.lower().strip(),
                relation.lower().strip()
            )
            relations.append(
                ConceptRelationship(
                    concept1=concept1,
                    concept2=concept2,
                    relation=relation_type
                )
            )
        return relations

    def _extract_llm(self, concepts: List[Concept], text: str, context: str) -> List[Tuple[Concept, str, Concept]]:
        concept_names = [c.name for c in concepts]
        try:
            response = self.llm.generate(inputs={
                                         'text': text, 'context': context, 'concept_names': concept_names})
            parsed = json.loads(response)

            results = []

            name_to_obj = {c.name: c for c in concepts}

            for item in parsed:
                c1 = name_to_obj.get(item["source"])
                c2 = name_to_obj.get(item["target"])
                if c1 and c2:
                    results.append((c1, item["relation"], c2))
            return results
        except Exception as e:
            print(f"LLM Extraction failed: {e}")
            return []

    def extract(self, concepts: List[Concept], text: str, context: str) -> List[ConceptRelationship]:
        statistical_links = self._extract_statistical_relationships(
            concepts, text)
        llm_links = self._extract_llm(concepts, text, context)
        llm_relations = self._build_relations(llm_links, context)

        seen = set()
        stat_relations = []

        for _, rel_list in statistical_links:
            for _, rel in rel_list:
                key = tuple(sorted([rel.concept1.name, rel.concept2.name]))
                if key not in seen:
                    stat_relations.append(rel)
                    seen.add(key)
        return llm_relations + stat_relations

    def _extract_statistical_relationships(
        self, concepts: List[Concept], text: str
    ) -> List[Tuple[Concept, List[Tuple[Concept, ConceptRelationship]]]]:
        """Identifies relationships based on sentence-level co-occurrence and proximity."""
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        num_sentences = len(sentences)
        if num_sentences == 0:
            return []

        sentence_to_concepts = defaultdict(list)
        for concept in concepts:
            for idx, sent in enumerate(sentences):
                for alias in concept.aliases:
                    alias_lower = alias.lower()
                    sent_lower = sent.lower()
                    pattern = r"\b" + re.escape(alias_lower) + r"\b"
                    if re.search(pattern, sent_lower):
                        sentence_to_concepts[idx].append(concept)
                        break

        pair_map = defaultdict(list)
        for idx, concepts_in_sent in sentence_to_concepts.items():
            if len(concepts_in_sent) < 2:
                continue

            for c1, c2 in combinations(concepts_in_sent, 2):
                pair_key = tuple(sorted([c1, c2], key=lambda x: x.name))
                pair_map[pair_key].append(idx)

        nested_results = defaultdict(list)
        for (c1, c2), indices in pair_map.items():
            co_occurrence_count = len(indices)
            weight = min(1.0, (co_occurrence_count / num_sentences) * 2)
            snippets = [sentences[i] for i in indices[:3]]

            relationship = ConceptRelationship(
                concept1=c1,
                concept2=c2,
                relation="related_to",
                definition=f"Concepts appear together in {co_occurrence_count} sentence(s).",
                strength=weight,
                attributes={
                    "count": co_occurrence_count,
                    "indices": indices,
                    "snippets": snippets,
                    "method": "statistical_cooccurrence",
                },
            )

            nested_results[c1.name].append((c2, relationship))
            nested_results[c2.name].append((c1, relationship))

        final_output = []
        for concept in concepts:
            relations = nested_results.get(concept.name, [])
            final_output.append((concept, relations))

        return final_output


if __name__ == "__main__":
    from src.models.text import TextModels
    text_models = TextModels()
    ce = ConceptExtractor(text_models)

    text = """Section 1. Transformer-Based Models
                Transformers dominate text and code generation. They rely on self-attention mechanisms to
                model long-range dependencies in sequences. Transformers are highly scalable but computa-
                tionally expensive. Representative models include GPT-5, Gemini-3, and Claude-4.5.
                Section 2. Diffusion Models
                Diffusion models achieve state-of-the-art performance in image and video generation. They
                gradually denoise random noise into structured data. While producing high-quality outputs,
                inference can be slow. Examples include Stable Diffusion, Imagen,DALL-E 2 and Mid-
                journey."""

    concepts = ce.extract(
        text=text,
        context=text,
        section_id=1,
        paragraph_id=1
    )

    print(concepts)
