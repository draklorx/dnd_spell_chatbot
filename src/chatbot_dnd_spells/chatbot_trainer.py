import json
from pathlib import Path
from chatbot_dnd_spells.chatbot_config import ChatbotConfig
from embeddings import Embedder, SentenceChunker
from embeddings.data_classes import RawEntry
from intents import Trainer
from intents.interfaces import ChatbotTrainerInterface
from .spell_entity_classifier import SpellEntityClassifier

class ChatbotTrainer(ChatbotTrainerInterface):
    
    def __init__(self):
        # Get paths relative to this file's location
        current_dir = Path(__file__).parent
        self.config = ChatbotConfig(current_dir)

    def preprocess_data(self):
        print("Starting data preprocessing...")
        from .data_processor import DataProcessor
        processor = DataProcessor(
            self.config.raw_spell_data_path,
            self.config.raw_entity_label_data_path,
            self.config.processed_spell_data_path,
            self.config.processed_entity_label_data_path
        )
        processor.process_data()
        print("Data preprocessing complete.")

    def train_intents(self):
        print ("Starting intent training process...")
        trainer = Trainer(self.config.intents_path)
        trainer.train_and_save(self.config.model_path, self.config.model_data_path, self.config.intents_path)
        print ("Intent training complete.")

    def train_spell_embeddings(self):
        """Load entries from JSON and process them."""

        with open(self.config.processed_spell_data_path, 'r', encoding='utf-8') as f:
            spells_data = json.load(f)

        entries = []

        # map spells to entries using their name, description, higherLevelSlot, and cantripUpgrade
        for entry in spells_data['spells']:
            entry_text = entry['description']
            if 'higherLevelSlot' in entry and entry['higherLevelSlot']:
                entry_text += " " + entry['higherLevelSlot']
            if 'cantripUpgrade' in entry and entry['cantripUpgrade']:
                entry_text += " " + entry['cantripUpgrade']
            entries.append(RawEntry(entry['name'], entry_text))

        # Load and process the spells
        chunker = SentenceChunker()
        embedder = Embedder(self.config.spells_db_path)
        chunked_entries = chunker.chunk_entries(entries)
        embedder.process_entries(chunked_entries)

    def train_entity_classifier(self):
        print("Starting entity classifier training process...")
        nlp = SpellEntityClassifier.train(self.config.processed_entity_label_data_path)
        SpellEntityClassifier.save(nlp, self.config.entity_classifier_model_path)
        print("Entity classifier training complete.")

    def train(self):
        """
        Train the chatbot model.
        """
        while True:
            print ("What would you like to train?")
            print ("1. Data Preprocessing")
            print ("2. Intent Classifier")
            print ("3. Spell Embeddings")
            print ("A. All of the above")
            print ("Q. Quit")
            choice = input("You: ").strip()
            if choice == '1':
                self.preprocess_data()
            elif choice == '2':
                self.train_intents()
            elif choice == '3':
                self.train_spell_embeddings()
            elif choice.lower() == 'a':
                self.preprocess_data()
                self.train_intents()
                self.train_spell_embeddings()
            elif choice.lower() == 'q':
                print("Exiting training.")
                exit()
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 'q' to quit.")
                continue
