from spacy import Language
import spacy
import json
from .classifier_interface import ClassifierInterface

class EntityRuleClassifier(ClassifierInterface):
    def __init__(self, nlp: Language):
        self.nlp = nlp

    @staticmethod
    def train (entity_label_data_path):
        """Take labeled data and create an entity ruler model (not really training, just saving patterns)"""
        nlp = spacy.blank("en")
        # Add the entity ruler component using the string name
        ruler = nlp.add_pipe("entity_ruler")
        
        with open(entity_label_data_path, "r") as f:
            data = json.load(f)
            patterns = EntityRuleClassifier.parse_patterns(data)

        ruler.add_patterns(patterns)  # Wrap in a list
        print("ruler patterns:", ruler.patterns)
        
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

    def predict(self, text):
        print("parent predict called")
        doc = self.nlp(text)
        if doc is not None:
            return [(ent.text, ent.label_) for ent in doc.ents]
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