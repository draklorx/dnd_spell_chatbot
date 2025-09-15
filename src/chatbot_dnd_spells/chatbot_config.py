from pathlib import Path

class ChatbotConfig():
    
    def __init__(self, base_dir: Path):
        # Get paths relative to this file's location
        
        # intents
        self.intents_path = base_dir / 'intents' / 'intents.json'
        self.exceptions_path = base_dir / 'intents' / 'exceptions.txt'
        
        # raw data paths
        raw_data_dir = base_dir / 'data_raw'
        self.raw_spell_data_path = raw_data_dir / 'spells.json'
        self.raw_entity_label_data_path = raw_data_dir / 'entities.json'

        # processed data paths
        self.processed_data_dir = base_dir / 'data_processed'
        self.processed_spell_data_path = self.processed_data_dir / 'spells.json'
        self.processed_entity_label_data_path = self.processed_data_dir / 'entities.json'


        self.artifacts_dir = base_dir / 'artifacts'
        self.entity_classifier_model_path = self.artifacts_dir / 'entity_classifier_model'
        self.spells_db_path = self.artifacts_dir / 'spells.db'
        self.model_data_path = self.artifacts_dir / 'model_data.json'
        self.model_path = self.artifacts_dir / 'intent_model'

