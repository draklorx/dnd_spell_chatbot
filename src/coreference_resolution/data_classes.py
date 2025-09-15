from dataclasses import dataclass
from enum import Enum

class Role(Enum):
    USER = "user"
    BOT = "bot"

@dataclass
class Message:
    text: str
    role: Role