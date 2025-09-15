
from abc import ABC, abstractmethod
from ..data_classes import Prediction

class ClassifierInterface(ABC):
    @abstractmethod
    def _extract_key_value(self, text, label):
        """Extract the key part from matched entities based on label type"""
        pass

    @abstractmethod
    def predict(self, text) -> list[Prediction]:
        """Predict the class of the given text."""
        pass