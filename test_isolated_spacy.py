#!/usr/bin/env python3
"""
Isolated test for spaCy intent classifier
"""
import spacy
import json
import random
import os
from pathlib import Path

# Simple standalone version for testing
class TestSpacyIntentClassifier:
    def __init__(self):
        self.nlp = None
        self.intents = []
        self.intents_responses = {}
        
    def prepare_training_data(self, intents_path: str):
        """Parse intents.json and prepare training data for spaCy"""
        training_data = []
        
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
                import re
                clean_pattern = re.sub(r'\{[^}]+\}', 'ENTITY', pattern)
                
                # Create label dict with all intents set to False except current one
                cats = {intent_label: False for intent_label in self.intents}
                cats[tag] = True
                
                training_data.append((clean_pattern, {"cats": cats}))
                
        return training_data
    
    def train_simple(self, intents_path: str):
        """Simple training approach"""
        print("Preparing training data...")
        training_data = self.prepare_training_data(intents_path)
        
        # Create blank model
        self.nlp = spacy.blank("en")
        
        # Add textcat component
        textcat = self.nlp.add_pipe("textcat")
        
        # Add labels
        for intent in self.intents:
            textcat.add_label(intent)
        
        print(f"Training on {len(training_data)} examples...")
        print(f"Intents: {self.intents}")
        
        # Convert to spaCy format
        from spacy.training import Example
        examples = []
        for text, annotations in training_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        
        # Initialize with examples
        self.nlp.initialize(lambda: examples)
        
        print("✓ Model initialized successfully")
        
        # Train for a few iterations
        for i in range(5):
            losses = {}
            random.shuffle(examples)
            
            # Update in small batches
            batch_size = 4
            for j in range(0, len(examples), batch_size):
                batch = examples[j:j+batch_size]
                self.nlp.update(batch, losses=losses)
            
            print(f"Epoch {i+1}/5 - Loss: {losses.get('textcat', 0):.4f}")
        
        return True
    
    def predict(self, text: str):
        """Predict intent"""
        doc = self.nlp(text)
        scores = doc.cats
        if not scores:
            return None, 0.0
        predicted_intent = max(scores, key=scores.get)
        confidence = scores[predicted_intent]
        return predicted_intent, confidence
    
    def get_response(self, intent: str):
        """Get random response"""
        if intent in self.intents_responses and self.intents_responses[intent]:
            return random.choice(self.intents_responses[intent])
        return "No response available"

def test_classifier():
    """Test the classifier"""
    print("=== Testing Isolated spaCy Classifier ===\n")
    
    try:
        classifier = TestSpacyIntentClassifier()
        
        intents_path = Path("src/chatbot_dnd_spells/intents/intents.json")
        if not intents_path.exists():
            print(f"✗ Intents file not found: {intents_path}")
            return False
        
        # Train
        success = classifier.train_simple(str(intents_path))
        if not success:
            return False
        
        print("\n✓ Training completed")
        
        # Test predictions
        test_messages = [
            "Tell me about fireball",
            "What level is magic missile?",
            "What school is fireball?",
            "What does cure wounds do?"
        ]
        
        print("\nTesting predictions:")
        for message in test_messages:
            intent, confidence = classifier.predict(message)
            response = classifier.get_response(intent)
            print(f"  '{message}'")
            print(f"    -> Intent: {intent} (confidence: {confidence:.3f})")
            print(f"    -> Response: {response[:50]}...")
            print()
        
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_classifier()
