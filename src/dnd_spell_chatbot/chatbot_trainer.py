import json
from embeddings import Embedder
from pathlib import Path
from chatbot_core import Trainer
from chatbot_core.interfaces import ChatbotTrainerInterface

class ChatbotTrainer(ChatbotTrainerInterface):
    
    def __init__(self):
        # Get paths relative to this file's location
        current_dir = Path(__file__).parent
        artifacts_dir = current_dir / 'artifacts'
        self.intents_path = current_dir / 'data' / 'intents.json'
        self.model_path = artifacts_dir / 'chatbot_model.pth'
        self.model_data_path = artifacts_dir / 'model_data.json'
        self.spells_path = current_dir / 'data' / 'spells.json'
        self.spells_db_path = artifacts_dir / 'spells.db'

    def train(self):
        """
        Train the chatbot model.
        """
        print ("Starting training process...")
        trainer = Trainer(self.intents_path)
        trainer.train_and_save(self.model_path, self.model_data_path, self.intents_path)
            
        """Load entries from JSON and process them."""

        with open(self.spells_path, 'r', encoding='utf-8') as f:
            spells_data = json.load(f)

        entries = []

        # map spells to entries using their name, description, higherLevelSlot, and cantripUpgrade
        for entry in spells_data['spells']:
            entry_text = entry['description']
            if 'higherLevelSlot' in entry and entry['higherLevelSlot']:
                entry_text += " " + entry['higherLevelSlot']
            if 'cantripUpgrade' in entry and entry['cantripUpgrade']:
                entry_text += " " + entry['cantripUpgrade']
            entries.append({
                'name': entry['name'],
                'text': entry_text,
            })

        # Load and process the spells
        embedder = Embedder(self.spells_db_path)
        embedder.process_entries(entries)
            
        embedder.close()
