from .intent_classifier import IntentClassifier

class Trainer:
    def __init__(self, intents_path):
        self.intent_classifier = IntentClassifier()
        self.intents_path: str = intents_path

    def train_model(self, n_iter=20):
        """Train the spaCy intent classifier"""
        self.intent_classifier.train(self.intents_path, n_iter=n_iter)

    def train_and_save(self, model_path, model_data_path, intents_path):
        """Train and save the spaCy model"""
        self.train_model(n_iter=20)
        self.intent_classifier.save_model(model_path, model_data_path)
        print("Model retrained and saved.")
