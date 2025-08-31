from abc import ABC, abstractmethod


class ChatbotInterface(ABC):
    """Abstract base class for Chatbot implementations"""
    
    @abstractmethod
    def run(self) -> None:
        """Run the chatbot"""
        pass