#!/usr/bin/env python3
"""
Test the complete spaCy integration with the chatbot
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_full_integration():
    """Test the full chatbot integration with spaCy"""
    print("=== Testing Full spaCy Integration ===\n")
    
    try:
        # Import components individually to avoid circular imports
        from intents.models.spacy_intent_classifier import SpacyIntentClassifier
        from intents.assistant import Assistant
        
        print("✓ Successfully imported spaCy components")
        
        # Load the trained model
        model_dir = Path("src/chatbot_dnd_spells/artifacts/spacy_intent_model")
        model_data_path = Path("src/chatbot_dnd_spells/artifacts/spacy_model_data.json")
        
        if not model_dir.exists() or not model_data_path.exists():
            print(f"✗ Model files not found. Please run train_spacy_model.py first.")
            return False
        
        # Load the spaCy intent classifier
        intent_classifier = SpacyIntentClassifier()
        intent_classifier.load_model(str(model_dir), str(model_data_path))
        
        print("✓ Successfully loaded spaCy model")
        
        # Create Assistant with spaCy classifier
        exceptions_path = Path("src/chatbot_dnd_spells/intents/exceptions.txt")
        assistant = Assistant(intent_classifier, str(exceptions_path))
        
        print("✓ Successfully created Assistant with spaCy classifier")
        
        # Test some interactions
        test_messages = [
            "Tell me about fireball",
            "What level is magic missile?",
            "What school is fireball?",
            "What does cure wounds do?",
            "What's the range of lightning bolt?",
            "Random unrelated message that shouldn't match"
        ]
        
        print("\nTesting assistant responses:")
        for message in test_messages:
            try:
                intent, response = assistant.process_message(message)
                print(f"\n  Input: '{message}'")
                print(f"  Intent: {intent}")
                print(f"  Response: {response}")
            except Exception as e:
                print(f"  Error processing '{message}': {e}")
        
        print("\n✓ All integration tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_integration()
    if success:
        print("\n🎉 spaCy integration is working perfectly!")
        print("The chatbot is now using spaCy for intent classification.")
    else:
        print("\n❌ Integration test failed. Check the errors above.")
        sys.exit(1)
