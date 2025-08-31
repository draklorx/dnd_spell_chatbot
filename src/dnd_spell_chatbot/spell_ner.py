import json
from fuzzywuzzy import fuzz
from chatbot_core import NerInterface

class SpellNer(NerInterface):
    def __init__(self, spells_data_path: str):
        self.spells_data = {}
        self.spell_names = []
        self.load_data(spells_data_path)
        
    def load_data(self, spells_path: str):
        """Load spell data from JSON file"""
        with open(spells_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for spell in data['spells']:
            self.spells_data[spell['name'].lower()] = spell
            self.spell_names.append(spell['name'].lower())
    
    def extract_entities(self, text: str) -> dict[str, any]:
        """Extract spell names and attributes from text using fuzzy matching"""
        text_lower = text.lower()
        entities = {}
        
        # Find the best matching spell name
        best_match = None
        best_score = 0
        
        for spell_name in self.spell_names:
            # Direct substring match
            if spell_name in text_lower:
                best_match = spell_name
                best_score = 100
                break
                
            # Fuzzy matching for partial matches
            score = fuzz.partial_ratio(spell_name, text_lower)
            if score > best_score and score > 70:  # Threshold for fuzzy matching
                best_match = spell_name
                best_score = score
        
        if best_match:
            spell_data = self.spells_data[best_match]
            entities['spell_name'] = spell_data['name']
            entities['spell_data'] = spell_data
            
            # Extract level information
            if any(word in text_lower for word in ['level', 'tier']):
                entities['level'] = spell_data['level']
                
            # Extract damage information
            if any(word in text_lower for word in ['damage', 'hurt', 'harm']):
                if spell_data.get('damage_dice'):
                    entities['damage_dice'] = spell_data['damage_dice']
                    entities['damage_type'] = spell_data['damage_type']
                    
            # Extract class information
            if any(word in text_lower for word in ['class', 'who can cast', 'caster']):
                entities['classes'] = spell_data['classes']
                
        return entities
    
    def intent_requires_ner(self, intent: str) -> bool:
        """Check if an intent requires NER processing"""
        # You could store this in the intents.json or hard-code it
        ner_intents = ['spell_info', 'spell_level', 'spell_damage', 'spell_classes']
        return intent in ner_intents
    
    def substitute_entities(self, response: str, entities: dict) -> str:
        """Substitute entity placeholders in responses"""
        for key, value in entities.items():
            if isinstance(value, list):
                value = ", ".join(value)
            response = response.replace(f"{{{key}}}", str(value))
        return response
    
    def get_spell_info(self, spell_name: str) -> dict:
        """Get full spell information by name"""
        return self.spells_data.get(spell_name.lower(), {})