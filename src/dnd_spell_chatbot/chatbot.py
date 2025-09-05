from pathlib import Path
from chatbot_core import Assistant, ChatbotInterface, ModelData, Trainer
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
        self.model_data_path = artifacts_dir / 'model_data.json'
        self.exceptions_path = current_dir / 'logs' / 'exceptions.log'
        self.function_mappings = {}

        self.ner = SpellNer(self.spells_path)


    @staticmethod
    def substitute_spell_data(response: str, entities: dict) -> str:
        """Substitute entity placeholders found in the response with values from spell data"""
        # Find all placeholders in the response {key}
        placeholders = re.findall(r'\{(\w+)\}', response)
        
        for key in placeholders:
            if key in entities["spell_data"]:
                value = entities["spell_data"][key]
                # Handle special cases
                if key == "components":
                    value = ", ".join(value)
                    material = entities["spell_data"].get("material", '')
                    value += f" ({material})" if material else ''
                elif isinstance(value, list):
                    value = ", ".join(value)
            elif key == "casting_time":
                casting_time = entities["spell_data"].get("castingTime", "")
                value = casting_time if casting_time else entities["spell_data"].get("actionType", "")
            else:
                raise ValueError(f"Unknown placeholder '{key}' in response.")

            response = response.replace(f"{{{key}}}", str(value))
                
        return response
    
    def load(self):
        print("Loading model...")
        print(self.model_path)
        print(self.model_data_path)
        model = ModelData.load_model(self.model_path, self.model_data_path)

        self.assistant = Assistant(
            model,
            self.exceptions_path
        )

    def train(self):
        trainer = Trainer(self.intents_path)
        trainer.train_and_save(self.model_path, self.model_data_path, self.intents_path)

        self.assistant = Assistant(
            trainer.model,
            self.exceptions_path
        )

    def run(self):
        print("Welcome to the DnD Spell Chatbot!")
        print("Type '/retrain' to retrain the model or '/quit' to exit.")

        while True:
            message = input('You:')

            if message == "/retrain":
                self.train()
                continue

            if message == "/quit":
                exit()

            predicted_intent, response = self.assistant.process_message(message)

            # Extract spell entities if this intent requires NER
            entities = {}
            if self.ner.intent_requires_ner(predicted_intent):
                entities = self.ner.extract_entities(message)
                
                if entities["confidence"] < 65:
                    response = f"That spell does not exist in my grimoire. I'm limited to SRD data. If your spell is not in the SRD, I won't be able to help."
                elif entities["confidence"] < 80:
                    response = f"I couldn't find that spell in my grimoire. Did you mean {entities['spell_name']}?"
                else:
                    response = self.substitute_spell_data(response, entities)

            print(response)
            
            # Handle function mappings
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()