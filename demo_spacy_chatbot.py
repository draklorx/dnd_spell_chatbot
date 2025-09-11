#!/usr/bin/env python3
"""
Demo script showing the spaCy-powered DnD Spell Chatbot
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_chatbot():
    """Demonstrate the spaCy-powered chatbot"""
    print("🎲 === DnD Spell Chatbot with spaCy ===")
    print("Now powered by spaCy for intent classification!\n")
    
    try:
        from intents.models.spacy_intent_classifier import SpacyIntentClassifier
        from intents.assistant import Assistant
        
        # Load the spaCy model
        model_dir = Path("src/chatbot_dnd_spells/artifacts/spacy_intent_model")
        model_data_path = Path("src/chatbot_dnd_spells/artifacts/spacy_model_data.json")
        
        intent_classifier = SpacyIntentClassifier()
        intent_classifier.load_model(str(model_dir), str(model_data_path))
        
        # Create assistant
        exceptions_path = Path("src/chatbot_dnd_spells/intents/exceptions.txt")
        assistant = Assistant(intent_classifier, str(exceptions_path))
        assistant.debug = True  # Show debug info
        
        print("✓ spaCy model loaded successfully!")
        print("✓ Assistant ready!")
        print(f"Available intents: {intent_classifier.intents}")
        print("\n" + "="*60)
        
        # Demo conversations
        demo_messages = [
            "Tell me about fireball",
            "What level is magic missile?",
            "What school is fireball?", 
            "What does cure wounds do?",
            "What's the range of lightning bolt?",
            "What components does shield require?",
            "How long does bless last?",
            "What classes can cast heal?",
            "This is a random message that shouldn't match anything"
        ]
        
        for i, message in enumerate(demo_messages, 1):
            print(f"\n[Demo {i}] User: {message}")
            intent, response = assistant.process_message(message)
            print(f"Response: {response}")
            
            if intent:
                print(f"💡 Intent detected: {intent}")
            else:
                print("💡 No specific intent detected")
        
        print("\n" + "="*60)
        print("🎉 Demo completed!")
        print("\nKey improvements with spaCy:")
        print("  ✓ Better natural language understanding")
        print("  ✓ More robust intent classification")
        print("  ✓ Easier to train and extend")
        print("  ✓ Production-ready performance")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_chatbot()
    if not success:
        print("\n❌ Please ensure the spaCy model is trained by running:")
        print("  python train_spacy_model.py")
        sys.exit(1)
