
from abc import ABC, abstractmethod

from embeddings.data_classes import ChunkedEntry, RawEntry

class ContextChunkerInterface(ABC):
    @abstractmethod
    def chunk_entries(self, entries: list[RawEntry]) -> list[ChunkedEntry]:
        """Break text into smaller overlapping chunks"""
        pass