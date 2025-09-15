import json
from .data_classes import Message, Role
from entity_recognition.data_classes import Prediction


class ChatContext:
    def __init__(self):
        self.context = {}
        self.chat_history: list[Message] = []

    def update_context(self, value: Prediction):
        self.context[value.label] = value

    def get_context(self, label):
        return self.context.get(label, None)

    def clear_contexts(self):
        self.context = {}

    def add_to_chat_history(self, user_message, bot_response):
        self.chat_history.append(Message(user_message, Role.USER))
        self.chat_history.append(Message(bot_response, Role.BOT))

    def fetch_data(self, file_path, key, value):
        """Fetch data from a JSON file based on a key-value pair."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data['spells']:
                    if item.get(key).lower() == value.lower():
                        return item
            return {}
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
            return {}

    def get_chat_history(self):
        return self.chat_history