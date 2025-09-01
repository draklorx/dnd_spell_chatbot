import os
from dnd_spell_chatbot.chatbot import Chatbot

def intents_updated(model_path, model_data_path, intents_path) -> bool:
    """Check if the intents file has been modified since the model was last trained"""
    # If model doesn't exist, it needs to be trained
    if not model_path.exists() or not model_data_path.exists():
        return True
        
    # Compare modification timestamps
    intents_mtime = os.path.getmtime(intents_path)
    model_mtime = os.path.getmtime(model_path)
    
    # Return True if intents file is newer than model file
    return intents_mtime > model_mtime

if __name__ == "__main__":
    try:
        chatbot = Chatbot()
        if intents_updated(chatbot.model_path, chatbot.model_data_path, chatbot.intents_path):
            chatbot.train()
        else:
            chatbot.load()
    except Exception as e:
        print(f"Error occurred during initialization: {e}")
        exit()

    try:
        chatbot.run()
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Attempting to restart")
        chatbot.run()