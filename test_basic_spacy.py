#!/usr/bin/env python3
"""
Simple test for spaCy integration
"""
import spacy
import json
from pathlib import Path

def test_basic_spacy():
    """Test basic spaCy functionality"""
    print("Testing basic spaCy functionality...")
    
    try:
        # Create a blank model
        nlp = spacy.blank("en")
        
        # Add text categorizer
        textcat = nlp.add_pipe("textcat")
        
        # Add some test labels
        textcat.add_label("info")
        textcat.add_label("level")
        
        print("✓ spaCy model created successfully!")
        
        # Test tokenization
        doc = nlp("Tell me about fireball")
        print(f"✓ Tokenization works: {[token.text for token in doc]}")
        
        return True
        
    except Exception as e:
        print(f"✗ spaCy test failed: {e}")
        return False

def test_intents_file():
    """Test reading the intents file"""
    print("\nTesting intents file...")
    
    intents_path = Path("src/chatbot_dnd_spells/intents/intents.json")
    
    if not intents_path.exists():
        print(f"✗ Intents file not found at {intents_path}")
        return False
    
    try:
        with open(intents_path, 'r') as f:
            intents_data = json.load(f)
        
        print(f"✓ Intents file loaded successfully!")
        print(f"  Found {len(intents_data['intents'])} intent categories")
        
        for intent in intents_data['intents'][:3]:  # Show first 3
            print(f"    - {intent['tag']}: {len(intent['patterns'])} patterns")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to read intents file: {e}")
        return False

if __name__ == "__main__":
    print("=== spaCy Integration Test ===\n")
    
    test1_success = test_basic_spacy()
    test2_success = test_intents_file()
    
    if test1_success and test2_success:
        print("\n✓ All tests passed! spaCy integration is ready.")
    else:
        print("\n✗ Some tests failed. Check the errors above.")
