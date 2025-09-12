
from abc import ABC, abstractmethod

class ClassifierInterface(ABC):
    @abstractmethod
    def train(entity_label_data_path, entity_classifier_model_path):
        """Train the classifier with the provided data."""
        pass

    @abstractmethod
    def predict(self, text):
        """Predict the class of the given text."""
        pass