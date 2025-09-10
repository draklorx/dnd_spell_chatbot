from pathlib import Path

class ChatbotConfig():
    
    def __init__(self, base_dir: Path):
        # Get paths relative to this file's location
        artifacts_dir = base_dir / 'artifacts'
        self.intents_path = base_dir / 'intents' / 'intents.json'
        self.model_path = artifacts_dir / 'chatbot_model.pth'
        self.model_data_path = artifacts_dir / 'model_data.json'
        self.spells_path = base_dir / 'data' / 'spells.json'
        self.spells_db_path = artifacts_dir / 'spells.db'
        self.exceptions_path = base_dir / 'intents' / 'exceptions.txt'

