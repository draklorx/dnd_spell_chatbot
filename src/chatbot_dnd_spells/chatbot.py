from pathlib import Path
from .spell_entity_classifier import SpellEntityClassifier
from intents.assistant import Assistant
from intents.models import ModelData
from intents.interfaces import ChatbotInterface
from chatbot_dnd_spells.chatbot_config import ChatbotConfig
from .spell__vector_searcher import SpellVectorSearcher
from coreference_resolution import ChatContext
from coreference_resolution.coreference_resolver import CoreferenceResolver
from entity_recognition import Prediction
import re

class Chatbot(ChatbotInterface):
    
    def __init__(self):
        # Get paths relative to this file's location
        current_dir = Path(__file__).parent
        self.config = ChatbotConfig(current_dir)
        self.function_mappings = {}
        self.entity_classifier = SpellEntityClassifier(self.config.processed_entity_label_data_path)
        self.chat_context = ChatContext()
        self.coreference_resolver = CoreferenceResolver(self.chat_context)

    def substitute_spell_data(self, response: str) -> str:
        """Substitute entity placeholders found in the response with values from spell data"""
        # Find all placeholders in the response {key}
        placeholders = re.findall(r'\{(\w+)\}', response)
        
        spell_name = self.chat_context.get_context("SPELL")

        spell_data = self.chat_context.fetch_data(self.config.processed_spell_data_path, "name", spell_name.value)

        for key in placeholders:
            if key in spell_data:
                value = spell_data[key]
                # Handle special cases
                if key == "components":
                    value = ", ".join(value)
                    material = spell_data.get("material", '')
                    value += f" ({material})" if material else ''
                elif isinstance(value, list):
                    value = ", ".join(value)
            elif key == "casting_time":
                casting_time = spell_data.get("castingTime", "")
                value = casting_time if casting_time else spell_data.get("actionType", "")
            else:
                raise ValueError(f"Unknown placeholder '{key}' in response.")

            response = response.replace(f"{{{key}}}", str(value))
                
        return response
    
    def _extract_entities_from_response(self, response: str):
        """Extract entities from bot responses to update context"""
        # This could be expanded to extract entities from bot responses
        # For now, we'll focus on user message processing
        pass
    
    def load(self):
        intent_classifier = ModelData.load_model(self.config.model_path, self.config.model_data_path)

        self.assistant = Assistant(
            intent_classifier,
            self.config.exceptions_path
        )

        self.vector_searcher = SpellVectorSearcher(self.config.spells_db_path)

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

            # First, resolve any coreferences in the message
            resolved_entities = self.coreference_resolver.resolve_coreferences(message)
            
            # Update chat context with resolved entities
            for entity_type, entity_value in resolved_entities.items():
                # Create a prediction object for the resolved entity
                resolved_prediction = Prediction(entity_type, entity_value, 95.0)  # High confidence for resolved entities
                self.chat_context.update_context(resolved_prediction)

            predicted_intent, response = self.assistant.process_message(message)

            # Extract spell entities if this intent requires entity recognition (only if no coreferences were resolved)
            if not resolved_entities:
                self.chat_context.clear_contexts()
                predictions = self.entity_classifier.predict(message)
                for prediction in predictions:
                    if prediction.confidence >= 80:
                        if prediction.label != "LEVEL":
                            self.chat_context.update_context(prediction)
                        elif (prediction.confidence >= 90):
                            self.chat_context.update_context(prediction)

            # Determine if we're querying for a spell list
            if predicted_intent == "spell_query":
                # TODO implement spell list querying
                pass
            else:
                spell = self.chat_context.get_context("SPELL")
                if spell is None or spell.confidence < 80:
                    if not predicted_intent:
                        response = "I'm not sure what you mean. Could you please rephrase?"
                    else:
                        response = "I'm sorry, I can't find that spell in my grimoire. Could you try again?"
                elif spell.confidence < 90:
                    response = f"I couldn't find that spell in my grimoire. Did you mean {spell.value}?"
                else:
                    if not predicted_intent:
                        response = self.vector_searcher.search(message, spell.value, rec_score=0.45, min_score=0.5, max_results=3)
                    else:
                        response = self.substitute_spell_data(response)

            print(response)
            print() # add a blank line for readability
            
            # Add conversation to chat history
            self.chat_context.add_to_chat_history(message, response)
            
            # Handle function mappings
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()