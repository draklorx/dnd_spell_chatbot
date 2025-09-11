import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import json
import os
from pathlib import Path

class IntentClassifier:
    def __init__(self):
        self.nlp = None
        self.intents = []
        self.intents_responses = {}
        
    def create_model(self, intents: list[str]):
        """Create a new spaCy model with text categorizer component"""
        # Create blank English model
        self.nlp = spacy.blank("en")
        
        # Add text categorizer to the pipeline with correct config
        textcat = self.nlp.add_pipe("textcat")
        
        # Add intent labels to the text categorizer
        for intent in intents:
            textcat.add_label(intent)
            
        self.intents = intents
        
    def prepare_training_data(self, intents_path: str):
        """Parse intents.json and prepare training data for spaCy"""
        training_data = []
        
        if os.path.exists(intents_path):
            with open(intents_path, 'r') as f:
                intents_data = json.load(f)
                
            for intent in intents_data['intents']:
                tag = intent['tag']
                
                if tag not in self.intents:
                    self.intents.append(tag)
                    self.intents_responses[tag] = intent['responses']
                
                # Create training examples for each pattern
                for pattern in intent['patterns']:
                    # Clean pattern of entity placeholders for training
                    clean_pattern = self._clean_pattern(pattern)
                    
                    # Create label dict with all intents set to False except current one
                    cats = {intent_label: False for intent_label in self.intents}
                    cats[tag] = True
                    
                    training_data.append((clean_pattern, {"cats": cats}))
                    
        return training_data
    
    def _clean_pattern(self, pattern: str) -> str:
        """Remove entity placeholders like {name} from patterns for training"""
        import re
        # Replace {name} placeholders with generic tokens
        pattern = re.sub(r'\{name\}', 'SPELL_NAME', pattern)
        pattern = re.sub(r'\{[^}]+\}', 'ENTITY', pattern)
        return pattern
    
    def train(self, intents_path: str, n_iter: int = 20):
        """Train the spaCy intent classifier"""
        print("Preparing training data...")
        training_data = self.prepare_training_data(intents_path)
        
        # Create model if not already created
        if self.nlp is None:
            self.create_model(self.intents)
            
        print(f"Training on {len(training_data)} examples...")
        
        # Get the textcat component
        textcat = self.nlp.get_pipe("textcat")
        
        # Convert training data to spaCy format
        examples = []
        for text, annotations in training_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        
        # Initialize the model
        self.nlp.initialize(lambda: examples)
        
        # Training loop
        for i in range(n_iter):
            losses = {}
            random.shuffle(examples)
            
            # Batch training examples
            batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
            
            for batch in batches:
                self.nlp.update(batch, losses=losses)
                
            print(f"Epoch {i+1}/{n_iter} - Loss: {losses.get('textcat', 0):.4f}")
    
    def predict(self, text: str) -> tuple[str | None, float]:
        """Predict intent for given text"""
        if self.nlp is None:
            raise ValueError("Model not trained or loaded")
            
        doc = self.nlp(text)
        
        # Get the highest scoring category
        scores = doc.cats
        if not scores:
            return None, 0.0
            
        predicted_intent = max(scores, key=scores.get)
        confidence = scores[predicted_intent]
        
        return predicted_intent, confidence
    
    def get_response(self, intent: str) -> str:
        """Get a random response for the given intent"""
        if intent in self.intents_responses and self.intents_responses[intent]:
            return random.choice(self.intents_responses[intent])
        return "I'm not sure how to respond to that."
    
    def save_model(self, model_path: str, model_data_path: str):
        """Save the trained spaCy model and metadata"""
        if self.nlp is None:
            raise ValueError("No model to save")
            
        # Save spaCy model - use parent directory and create spacy_intent_model subdirectory
        base_dir = Path(model_path).parent
        model_dir = base_dir / "spacy_intent_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(model_dir)
        
        # Save metadata
        metadata = {
            'intents': self.intents,
            'intents_responses': self.intents_responses
        }
        
        with open(model_data_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Model saved to {model_dir}")
        print(f"Metadata saved to {model_data_path}")
    
    def load_model(self, model_path: str, model_data_path: str):
        """Load a trained spaCy model and metadata"""
        # Load spaCy model - look in the spacy_intent_model subdirectory
        base_dir = Path(model_path).parent
        model_dir = base_dir / "spacy_intent_model"
        self.nlp = spacy.load(model_dir)
        
        # Load metadata
        with open(model_data_path, 'r') as f:
            metadata = json.load(f)
            
        self.intents = metadata['intents']
        self.intents_responses = metadata['intents_responses']
        
        print(f"Model loaded from {model_path}")
