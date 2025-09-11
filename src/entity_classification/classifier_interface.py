
from abc import ABC, abstractmethod

class ClassifierInterface(ABC):
    @abstractmethod
    def train(self, data):
        """Train the classifier with the provided data."""
        pass

    @abstractmethod
    def predict(self, text):
        """Predict the class of the given text."""
        pass