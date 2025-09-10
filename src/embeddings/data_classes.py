from dataclasses import dataclass

@dataclass
class RawEntry:
    name: str
    text: str

@dataclass
class Chunk:
    text: str

@dataclass
class ChunkContext:
    text: str
    position: int
    chunks: list[Chunk]

@dataclass
class ChunkedEntry:
    name: str
    chunk_contexts: list[ChunkContext]