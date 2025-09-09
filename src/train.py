import json
from dnd_spell_chatbot import ChatbotTrainer
from embeddings import Embedder

if __name__ == "__main__":
    try:
        trainer = ChatbotTrainer()
        trainer.train()
    except Exception as e:
        print(f"Error occurred during setup & training: {e}")
        exit()
