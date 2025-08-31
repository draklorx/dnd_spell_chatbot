from pathlib import Path
from chatbot_core import Assistant, ChatbotInterface
from .spell_ner import SpellNer

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

        spell_ner = SpellNer(self.spells_path)
        self.assistant = Assistant(
            self.intents_path,
            self.exceptions_path,
            ner=spell_ner
        )
        self.assistant.parse_intents()

        try:
            self.assistant.load_model(self.model_path, self.dimensions_path)
        except FileNotFoundError:
            self.assistant.train_and_save(self.model_path, self.dimensions_path)

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

            print(self.assistant.process_message(message))