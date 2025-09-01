from pathlib import Path
from chatbot_core import Assistant, ChatbotInterface
from .spell_ner import SpellNer
import re

class Chatbot(ChatbotInterface):
    
    def __init__(self):
        # Get paths relative to this file's location
        current_dir = Path(__file__).parent
        artifacts_dir = current_dir / 'artifacts'
        self.intents_path = current_dir / 'data' / 'intents.json'
        self.spells_path = current_dir / 'data' / 'spells.json'
        self.model_path = artifacts_dir / 'chatbot_model.pth'
        self.dimensions_path = artifacts_dir / 'dimensions.json'
        self.exceptions_path = current_dir / 'logs' / 'exceptions.log'
        self.function_mappings = {}

        self.ner = SpellNer(self.spells_path)

        self.assistant = Assistant(
            self.intents_path,
            self.exceptions_path
        )
        self.assistant.parse_intents()

        try:
            self.assistant.load_model(self.model_path, self.dimensions_path)
        except FileNotFoundError:
            self.assistant.train_and_save(self.model_path, self.dimensions_path)

    @staticmethod
    def substitute_spell_data(response: str, entities: dict) -> str:
        """Substitute entity placeholders found in the response with values from spell data"""
        # Find all placeholders in the response {key}
        placeholders = re.findall(r'\{(\w+)\}', response)
        
        for key in placeholders:
            value = ""
            if key in entities["spell_data"]:
                # Handle special cases
                if key == "components":
                    value = ", ".join(value)
                    material = entities["spell_data"].get("material", '')
                    value += f" ({material})" if material else ''
                elif isinstance(value, list):
                    value = ", ".join(value)
                else:
                    value = entities["spell_data"][key]
            elif key == "casting_time":
                casting_time = entities["spell_data"].get("castingTime", "")
                value = casting_time if casting_time else entities["spell_data"].get("actionType", "")

            response = response.replace(f"{{{key}}}", str(value))
                
        return response
    
    def run(self):
        print("Welcome to the DnD Spell Chatbot!")
        print("Type '/retrain' to retrain the model or '/quit' to exit.")

        while True:
            message = input('You:')

            if message == "/retrain":
                self.assistant.train_and_save(self.model_path, self.dimensions_path)
                continue

            if message == "/quit":
                exit()

            predicted_intent, response = self.assistant.process_message(message)

            # Extract spell entities if this intent requires NER
            entities = {}
            if self.ner.intent_requires_ner(predicted_intent):
                entities = self.ner.extract_entities(message)
                
                if entities["confidence"] < 80:
                    return f"I couldn't find that spell in my grimoire. Did you mean {entities['spell_name']}?"
                response = self.substitute_spell_data(response, entities)

            print(response)
            
            # Handle function mappings
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()