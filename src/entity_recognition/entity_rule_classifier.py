from spacy import Language
import spacy
import json
from .data_classes import Prediction
from .interfaces.classifier_interface import ClassifierInterface

# This was an attempt to use spaCy's EntityRuler for rule-based entity recognition
# TODO: This code is not thoroughly tested, and may not work with the latest interfaces
# Uses spaCy's EntityRuler to create a rule-based entity recognition model
# IMPORTANT: This only works for relatively small data sets with predictable patterns
# This does not support fuzzy matching
# For larger data sets or more complex patterns, consider using a more advanced recognition method
class EntityRuleClassifier(ClassifierInterface):
    def __init__(self, nlp: Language):
        self.nlp = nlp

    @staticmethod
    def build_model (entity_label_data_path):
        """Build a spaCy NER model using the entity ruler component."""
        nlp = spacy.blank("en")
        # Add the entity ruler component using the string name
        ruler = nlp.add_pipe("entity_ruler")
        
        with open(entity_label_data_path, "r") as f:
            data = json.load(f)
            patterns = EntityRuleClassifier.parse_patterns(data)

        ruler.add_patterns(patterns)  # Wrap in a list
        
        return nlp

    @staticmethod
    def parse_patterns(data):
        patterns = []
        for entity in data["entities"]:
            label = entity["label"]
            for pattern in entity["patterns"]:
                # Handle multi-word phrases
                if " " in pattern:
                    pattern_tokens = [{"LOWER": word} for word in pattern.split()]
                else:
                    pattern_tokens = [{"LOWER": pattern}]
                patterns.append({"label": label, "pattern": pattern_tokens})
        return patterns

    def _extract_key_value(self, text, label):
        """Extract the key part from matched entities based on label type"""
        return text  # Default implementation, can be overridden in subclasses

    def predict(self, text):
        doc = self.nlp(text)
        if doc is not None:
            for ent in doc.ents:
                parsed_value = self._extract_key_value(ent.text, ent.label_)
                return [Prediction(ent.label_, parsed_value, 100)]
        return []

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