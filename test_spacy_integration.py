#!/usr/bin/env python3
"""
Test script for spaCy intent classification
"""
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import directly to avoid circular imports
from intents.models.spacy_intent_classifier import SpacyIntentClassifier

def test_spacy_classifier():
    """Test the spaCy intent classifier"""
    print("Testing spaCy Intent Classifier...")
    
    # Initialize classifier
    classifier = SpacyIntentClassifier()
    
    # Path to intents file
    intents_path = src_path / "chatbot_dnd_spells" / "intents" / "intents.json"
    
    if not intents_path.exists():
        print(f"Error: Intents file not found at {intents_path}")
        return False
    
    try:
        # Train the classifier
        print("Training classifier...")
        classifier.train(str(intents_path), n_iter=5)  # Use fewer iterations for testing
        
        # Test some predictions
        test_messages = [
            "Tell me about fireball",
            "What level is magic missile?",
            "What does cure wounds do?",
            "How does healing work?",
            "What's the range of lightning bolt?"
        ]
        
        print("\nTesting predictions:")
        for message in test_messages:
            intent, confidence = classifier.predict(message)
            print(f"Message: '{message}'")
            print(f"  Predicted Intent: {intent}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Response: {classifier.get_response(intent)}")
            print()
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_spacy_classifier()
    if success:
        print("✓ spaCy integration test completed successfully!")
    else:
        print("✗ spaCy integration test failed!")
        sys.exit(1)
