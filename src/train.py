from chatbot_dnd_spells import ChatbotTrainer

if __name__ == "__main__":
    trainer = ChatbotTrainer()
    # setup artifacts directory
    trainer.config.artifacts_dir.mkdir(exist_ok=True)
    trainer.train()
