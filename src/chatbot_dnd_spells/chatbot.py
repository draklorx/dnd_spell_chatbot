from pathlib import Path
from .spell_entity_classifier import SpellEntityClassifier
from intents.assistant import Assistant
from intents.models import ModelData
from intents.interfaces import ChatbotInterface
from chatbot_dnd_spells.chatbot_config import ChatbotConfig
from .spell_searcher import SpellSearcher
import re

class Chatbot(ChatbotInterface):
    
    def __init__(self):
        # Get paths relative to this file's location
        current_dir = Path(__file__).parent
        self.config = ChatbotConfig(current_dir)
        self.function_mappings = {}
        self.entity_classifier = SpellEntityClassifier.load(self.config.entity_classifier_model_path)

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
        intent_classifier = ModelData.load_model(self.config.model_path, self.config.model_data_path)

        self.assistant = Assistant(
            intent_classifier,
            self.config.exceptions_path
        )

        self.vector_searcher = SpellSearcher(self.config.spells_db_path)

    def run(self):
        print("Welcome to the DnD Spell Chatbot!")
        print("Type '/debug' to enter debug mode or '/quit' to exit.")

        while True:
            message = input('You:')

            if message == "/debug":
                self.debug = True
                self.assistant.debug = True
                self.vector_searcher.debug = True
                print("Debug mode enabled.")
                continue

            if message == "/quit":
                exit()

            predicted_intent, response = self.assistant.process_message(message)

            # Extract spell entities if this intent requires NER
            entities = self.entity_classifier.predict(message)
            print("entities", entities)
            # if entities["confidence"] < 65:
            #     response = f"That spell does not exist in my grimoire. I'm limited to SRD data. If your spell is not in the SRD, I won't be able to help."
            # elif entities["confidence"] < 80:
            #     response = f"I couldn't find that spell in my grimoire. Did you mean {entities['spell_name']}?"
            # else:
            #     if predicted_intent:
            #         response = self.substitute_spell_data(response, entities)
            #     else:
            #         response = self.vector_searcher.search(message, entities['spell_name'], 0.45, 0.5, 3)


            print(response)
            print() # add a blank line for readability
            
            # Handle function mappings
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()