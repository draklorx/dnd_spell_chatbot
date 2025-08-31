import random
from pathlib import Path
from chatbot_assistant import ChatbotAssistant

def end_conversation(assistant):
    print(random.choice(assistant.intents_responses["farewell"]))
    exit()

if __name__ == "__main__":
    # Get paths relative to this file's location
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    
    assistant = ChatbotAssistant(
        current_dir / 'data' / 'intents.json',
        function_mappings={"farewell": lambda: end_conversation(assistant)}
    )
    assistant.parse_intents()

    try:
        assistant.load_model(
            project_root / 'artifacts' / 'chatbot_model.pth',
            project_root / 'artifacts' / 'dimensions.json'
        )
    except FileNotFoundError:
        assistant.train_and_save()

    while True:
        message = input('You:')

        if message == "/retrain":
            assistant.train_and_save()
            continue

        print(assistant.process_message(message))