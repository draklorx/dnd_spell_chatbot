#!/usr/bin/env python3
"""
Test the spaCy intent classifier implementation
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_spacy_classifier():
    """Test the custom spaCy intent classifier"""
    print("Testing custom spaCy Intent Classifier...")
    
    # Import here to avoid circular imports
    from intents.models.spacy_intent_classifier import SpacyIntentClassifier
    
    try:
        # Initialize classifier
        classifier = SpacyIntentClassifier()
        
        # Path to intents file
        intents_path = Path("src/chatbot_dnd_spells/intents/intents.json")
        
        if not intents_path.exists():
            print(f"✗ Intents file not found at {intents_path}")
            return False
        
        print("✓ Classifier initialized")
        print("✓ Intents file found")
        
        # Train the classifier (just 3 iterations for testing)
        print("Training classifier...")
        classifier.train(str(intents_path), n_iter=3)
        
        print("✓ Training completed")
        
        # Test some predictions
        test_messages = [
            "Tell me about fireball",
            "What level is magic missile?",
            "What does cure wounds do?",
        ]
        
        print("\nTesting predictions:")
        for message in test_messages:
            intent, confidence = classifier.predict(message)
            response = classifier.get_response(intent)
            print(f"  '{message}' -> {intent} ({confidence:.3f})")
        
        print("✓ Predictions working")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing spaCy Intent Classifier ===\n")
    
    success = test_spacy_classifier()
    
    if success:
        print("\n✓ All tests passed! spaCy intent classifier is working.")
    else:
        print("\n✗ Tests failed. Check the errors above.")
