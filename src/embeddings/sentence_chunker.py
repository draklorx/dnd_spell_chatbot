import re
import math
import nltk
from nltk.tokenize import sent_tokenize
from .data_classes import RawEntry, ChunkedEntry, Chunk, ChunkContext
from embeddings.context_chunker_interface import ContextChunkerInterface

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SentenceChunker(ContextChunkerInterface):
    def __init__(self, chunk_size=10):
        """Initialize the chunker with a specific chunk size."""
        
        if chunk_size <= 4:
            raise ValueError("Target chunk size must be greater than 4.")
        self.chunk_size = chunk_size

    def chunk_entries(self, raw_entries: list[RawEntry]) -> list[ChunkedEntry]:
        """Process a text for an entry with chunking."""
        chunked_entries = []
        for raw_entry in raw_entries:
            chunk_contexts = []
            sentences = self.clean_and_split_text(raw_entry.text)
            for sentence_position, sentence in enumerate(sentences):
                chunks = self._chunk_sentence(sentence)
                chunk_context = ChunkContext(text=sentence, position=sentence_position, chunks=chunks)
                chunk_contexts.append(chunk_context)
            chunked_entry = ChunkedEntry(name=raw_entry.name, chunk_contexts=chunk_contexts)
            chunked_entries.append(chunked_entry)
        return chunked_entries


    def clean_and_split_text(self, text):
        """Clean text and split into sentences."""
        if not text:
            return []
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'#+\s*', '', text)             # Headers
        
        # Split into sentences
        sentences = sent_tokenize(text)

        return [sentence.strip() for sentence in sentences]


    def _chunk_sentence(self, sentence):
        """
        Break a sentence into smaller chunks while preserving meaning.
        
        Args:
            sentence: The sentence to chunk
        
        Returns:
            List of strings representing the chunks
        """
        
        words = sentence.split()
        min_overlap = math.ceil(self.chunk_size * 0.15)

        # If sentence is short enough, return as single chunk
        if len(words) <= self.chunk_size:
            return [Chunk(sentence)]

        # Determine number of chunks that still meets minimum overlap requirements
        num_chunks = math.ceil((len(words) - min_overlap) / (self.chunk_size - min_overlap))

        chunks = []
        # Evenly distribute chunks over the words
        for chunk_idx in range(num_chunks):
            if chunk_idx == 0:
                chunk_text = ' '.join(words[0:self.chunk_size])
                chunks.append(Chunk(chunk_text))
            elif chunk_idx == num_chunks - 1:
                chunk_text = ' '.join(words[-self.chunk_size:])
                chunks.append(Chunk(chunk_text))
            else:
                chunk_center = (len(words) // (num_chunks -1)) * (chunk_idx)
                chunk_text = ' '.join(words[chunk_center - self.chunk_size//2 : chunk_center + self.chunk_size//2])
                chunks.append(Chunk(chunk_text))

        return chunks