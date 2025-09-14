from spacy import Language
import spacy
import json
from spaczz.pipeline import SpaczzRuler
from .classifier_interface import ClassifierInterface

class EntityRuleClassifier(ClassifierInterface):
    def __init__(self, nlp: Language):
        self.nlp = nlp

    @staticmethod
    def train (entity_label_data_path):
        """Take labeled data and create a spaczz ruler model (fuzzy patterns)."""
        nlp = spacy.blank("en")

        # 1) Exact matches first
        exact = nlp.add_pipe("entity_ruler", config={"overwrite_ents": False})

        # 2) Fuzzy fallback second
        ruler = nlp.add_pipe("spaczz_ruler", config={"overwrite_ents": False})

        with open(entity_label_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            exact_patterns = EntityRuleClassifier.parse_exact_patterns(data)
            fuzzy_patterns = EntityRuleClassifier.parse_fuzzy_patterns(data)

        exact.add_patterns(exact_patterns)
        ruler.add_patterns(fuzzy_patterns)

        print("exact patterns:", len(exact.patterns))
        print("fuzzy patterns:", len(ruler.patterns))
        return nlp

    @staticmethod
    def parse_exact_patterns(data):
        """Build exact phrase patterns so perfect matches win and block fuzzy overlaps."""
        patterns = []
        for entity in data["entities"]:
            label = entity["label"]
            for phrase in entity["patterns"]:
                patterns.append({
                    "label": label,
                    "pattern": phrase,
                    "id": phrase
                })
        return patterns

    @staticmethod
    def parse_fuzzy_patterns(data):
        """
        Build spaczz patterns.
        - Multi-word phrases use token-level fuzzy to require each token to match.
        - Single words use fuzzy phrase.
        - Raise the minimum ratio to avoid spurious partial matches.
        """
        patterns = []
        for entity in data["entities"]:
            label = entity["label"]
            for phrase in entity["patterns"]:
                tokens = phrase.split()
                if len(tokens) > 1:
                    # Token-level fuzzy: each token must fuzzily match its counterpart.
                    token_pattern = [{"LOWER": {"FUZZY": t}} for t in tokens]
                    patterns.append({
                        "label": label,
                        "pattern": token_pattern,
                        "type": "token",
                        "id": phrase,
                        "kwargs": {"min_r": 90}  # tighten threshold
                    })
                else:
                    # Single-token fuzzy phrase
                    patterns.append({
                        "label": label,
                        "pattern": phrase,
                        "type": "fuzzy",
                        "id": phrase,
                        "kwargs": {"min_r": 90}
                    })
        return patterns
    
    def _extract_key_value(self, text, label):
        """Override in subclass to extract key part from matched entities based on label type"""
        return text

    def predict(self, text):
        doc = self.nlp(text)
        results = []
        for ent in doc.ents:
            ratio = getattr(ent._, "spaczz_ratio", None)
            if ratio is None:
                # From exact ruler (treat as perfect)
                ratio = 1.0
            matched_to = ent.ent_id_ or None  # set via pattern "id" above

            # Optional: use the canonical match for key extraction
            self._extract_key_value(matched_to or ent.text, ent.label_)

            results.append({
                "text": ent.text,
                "label": ent.label_,
                "confidence": ratio,
                "matched_to": matched_to
            })
        return results

    @classmethod
    def load(cls, entity_classifier_path: str):
        try:
            instance = cls(spacy.load(entity_classifier_path))
            return instance
        except Exception as e:
            print(f"Error loading entity classifier model: {e}")
            return None

    @staticmethod
    def save(nlp, entity_classifier_model_path: str):
        try:
            nlp.to_disk(entity_classifier_model_path)
        except Exception as e:
            print(f"Error saving entity classifier model: {e}")