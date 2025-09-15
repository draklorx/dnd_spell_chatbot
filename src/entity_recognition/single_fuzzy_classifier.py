import json

from entity_recognition.data_classes import Prediction
from .interfaces.classifier_interface import ClassifierInterface
from rapidfuzz import fuzz

# Finds the best fuzzy match from a list of patterns for each entity label greater than a minimum score
# IMPORTANT: This only works for relatively small data sets with predictable patterns
# For larger data sets or more complex patterns, consider using a more advanced recognition method
class SingleFuzzyClassifier(ClassifierInterface):
    def __init__(self, entity_classifier_path):
        with open(entity_classifier_path, 'r') as f:
            data = json.load(f)
            self.entities = data["entities"]

    def _extract_key_value(self, text, label):
        """Extract the key part from matched entities based on label type"""
        return text  # Default implementation, can be overridden in subclasses


    def predict(self, text):
        """Predict the class of the given text."""
        predicted_entities = []
        for entity in self.entities:
            best_score = 0
            for pattern in entity["patterns"]:
                score = fuzz.partial_ratio(pattern.lower(), text.lower())
                
                if score > best_score:
                    best_score = score
                    label = entity["label"]
                    value = pattern
                if best_score == 100:
                    break
            parsed_value = self._extract_key_value(value, label)
            predicted_entities.append(Prediction(label=label, value=parsed_value, confidence=best_score))
        return predicted_entities