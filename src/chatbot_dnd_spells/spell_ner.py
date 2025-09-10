import json
from fuzzywuzzy import fuzz
from intents.interfaces import NerInterface
import re

class SpellNer(NerInterface):
    def __init__(self, spells_data_path: str):
        self.spells_data = {}
        self.spell_names = []
        # Add conversation context tracking
        self.conversation_context = {
            "last_spell": None,
            "last_entities": {}
        }
        self.load_data(spells_data_path)
        
    def load_data(self, spells_path: str):
        """Load spell data from JSON file"""
        with open(spells_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for spell in data['spells']:
            self.spells_data[spell['name'].lower()] = spell
            self.spell_names.append(spell['name'].lower())
    
    def extract_entities(self, text: str) -> dict[str, any]:
        """Extract spell names and attributes from text using fuzzy matching with context"""
        text_lower = text.lower()
        entities = {}
        
        # Check for pronouns that might refer to previous spell
        pronouns = ['it', 'its', 'that', 'their', 'this spell', 'that spell', 'the spell', 'this one']
        has_pronoun = any(
            re.search(r'\b' + re.escape(pronoun) + r'\b', text_lower)
            for pronoun in pronouns
        )
        
        # If pronoun is used and we have context from previous message
        if has_pronoun and self.conversation_context["last_spell"]:
            best_match = self.conversation_context["last_spell"]
            # Start with previous entities as a base
            entities = self.conversation_context["last_entities"].copy()
            best_score = 100  # Assume perfect match since we're using context
        else:
            best_match = None
            best_score = 0
            
            for spell_name in self.spell_names:
                if spell_name in text_lower:
                    best_match = spell_name
                    best_score = 100
                    break
                    
                score = fuzz.partial_ratio(spell_name, text_lower)
                if score > best_score:
                    best_match = spell_name
                    best_score = score

        spell_data = self.spells_data[best_match]
        entities['spell_name'] = spell_data['name']
        entities['spell_data'] = spell_data
        entities['confidence'] = best_score
        # Update conversation context for next message
        self.conversation_context["last_spell"] = best_match
        self.conversation_context["last_entities"] = entities.copy()
        
        return entities
    
    def intent_requires_ner(self, intent: str) -> bool:
        """Check if an intent requires NER processing"""
        ner_intents = ['info', 'level', 'school', 'casting_time', 'range', 'components', 'duration', 'classes']
        return intent in ner_intents