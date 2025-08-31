from abc import ABC, abstractmethod


class NerInterface(ABC):
    """Abstract base class for Named Entity Recognition implementations"""
    
    @abstractmethod
    def load_data(self, data_path: str) -> None:
        """Load data from a file path"""
        pass
    
    @abstractmethod
    def extract_entities(self, text: str) -> dict[str, str]:
        """Extract entities from input text"""
        pass
    
    @abstractmethod
    def intent_requires_ner(self, intent: str) -> bool:
        """Check if an intent requires NER processing"""
        pass

    @abstractmethod
    def substitute_entities(self, response: str, entities: dict[str, str]) -> str:
        """Substitute entity placeholders in response templates"""
        pass