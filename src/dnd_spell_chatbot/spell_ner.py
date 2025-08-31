import json
import re
from fuzzywuzzy import fuzz
from chatbot_core import NerInterface

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
        pronouns = ['it', 'this spell', 'that spell', 'the spell', 'this one']
        has_pronoun = any(pronoun in text_lower for pronoun in pronouns)
        
        # If pronoun is used and we have context from previous message
        if has_pronoun and self.conversation_context["last_spell"]:
            best_match = self.conversation_context["last_spell"]
            # Start with previous entities as a base
            entities = self.conversation_context["last_entities"].copy()
            best_score = 100  # Assume perfect match since we're using context
        else:
            # Your existing spell matching code
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

        if best_score > 80:
            # Extract additional information based on the query
            if any(word in text_lower for word in ['level', 'tier']):
                entities['level'] = spell_data['level']
                
            # Enhanced damage extraction with description parsing
            if any(word in text_lower for word in ['damage', 'hurt', 'harm']):
                # Extract damage info from description
                description = spell_data.get('description', '')
                damage_pattern = r'(\d+d\d+)\s+(\w+)\s+damage'
                match = re.search(damage_pattern, description.lower())
                if match:
                    entities['damage_dice'] = match.group(1)
                    entities['damage_type'] = match.group(2)
                    
                # Also check for higher level casting info
                if 'higherLevelSlot' in spell_data:
                    entities['higher_level_info'] = spell_data['higherLevelSlot']
                    
            # Extract class information
            if any(word in text_lower for word in ['class', 'who can cast', 'caster']):
                entities['classes'] = spell_data['classes']
                        
        return entities
    
    def substitute_entities(self, response: str, entities: dict) -> str:
        """Substitute entity placeholders with enhanced context awareness"""
        # Special case for damage questions to include upcast information
        if '{damage_dice}' in response and 'damage_dice' in entities and 'higher_level_info' in entities:
            spell_name = entities.get('spell_name', 'The spell')
            damage_dice = entities.get('damage_dice', '')
            damage_type = entities.get('damage_type', 'damage')
            higher_info = entities.get('higher_level_info', '')
            
            # Create an enhanced response with upcasting information
            return f"{spell_name} deals {damage_dice} {damage_type} damage. {higher_info}"
        
        # Standard placeholder replacement
        for key, value in entities.items():
            if isinstance(value, list):
                value = ", ".join(value)
            if key != 'spell_data' and key != 'higher_level_info':  # Skip complex objects
                response = response.replace(f"{{{key}}}", str(value))
                
        return response
    
    def intent_requires_ner(self, intent: str) -> bool:
        """Check if an intent requires NER processing"""
        ner_intents = ['spell_info', 'spell_level', 'spell_damage', 'spell_classes']
        return intent in ner_intents
    
    def get_spell_info(self, spell_name: str) -> dict:
        """Get full spell information by name"""
        return self.spells_data.get(spell_name.lower(), {})