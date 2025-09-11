from pathlib import Path

class ChatbotConfig():
    
    def __init__(self, base_dir: Path):
        # Get paths relative to this file's location
        artifacts_dir = base_dir / 'artifacts'
        self.intents_path = base_dir / 'intents' / 'intents.json'
        
        # spaCy model paths
        self.model_path = artifacts_dir / 'spacy_intent_model'
        self.model_data_path = artifacts_dir / 'spacy_model_data.json'
        
        # Legacy PyTorch model paths (keeping for backup)
        self.pytorch_model_path = artifacts_dir / 'chatbot_model.pth'
        self.pytorch_model_data_path = artifacts_dir / 'model_data.json'
        
        self.spells_path = base_dir / 'data' / 'spells.json'
        self.spells_db_path = artifacts_dir / 'spells.db'
        self.exceptions_path = base_dir / 'intents' / 'exceptions.txt'

