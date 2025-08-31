import random
from pathlib import Path
from chatbot_assistant import ChatbotAssistant
from spell_ner import SpellNer

if __name__ == "__main__":
    # Get paths relative to this file's location
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    spell_ner = SpellNer(current_dir / 'data' / 'spells.json') if current_dir / 'data' / 'spells.json' else None
    assistant = ChatbotAssistant(
        current_dir / 'data' / 'intents.json',
        ner=spell_ner
    )
    assistant.parse_intents()

    try:
        assistant.load_model(
            project_root / 'artifacts' / 'chatbot_model.pth',
            project_root / 'artifacts' / 'dimensions.json'
        )
    except FileNotFoundError:
        assistant.train_and_save()

    print("Welcome to the DnD Spell Chatbot!")
    print("Type '/retrain' to retrain the model or '/quit' to exit.")

    while True:
        message = input('You:')

        if message == "/retrain":
            assistant.train_and_save()
            continue

        if message == "/quit":
            exit()

        print(assistant.process_message(message))