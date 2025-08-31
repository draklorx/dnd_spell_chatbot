import random
from ChatbotAssistant import ChatbotAssistant

def end_conversation(assistant):
    print(random.choice(assistant.intents_responses["farewell"]))
    exit()

if __name__ == "__main__":
    assistant = ChatbotAssistant('intents.json', function_mappings = {
        "farewell": lambda: end_conversation(assistant)
    })
    assistant.parse_intents()

    try:
        assistant.load_model('chatbot_model.pth', 'dimensions.json')
    except FileNotFoundError:
        assistant.train_and_save()


    while True:
        message = input('You:')

        if (message == "/retrain"):
            assistant.train_and_save()
            continue

        print(assistant.process_message(message))