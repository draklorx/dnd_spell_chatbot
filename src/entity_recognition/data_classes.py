from dataclasses import dataclass
from enum import Enum

@dataclass
class Prediction:
    label: str
    value: str
    confidence: float
