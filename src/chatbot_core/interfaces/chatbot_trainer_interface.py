from abc import ABC, abstractmethod


class ChatbotTrainerInterface(ABC):
    """Abstract base class for Chatbot implementations"""
    
    @abstractmethod
    def train(self) -> None:
        """Train the chatbot"""
        pass