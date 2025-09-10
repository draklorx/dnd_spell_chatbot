import random

import torch
import torch.nn.functional as F
from .utils.data_preprocessor import DataPreprocessor
from chatbot_dnd_spells.colors import YELLOW, RESET

class Assistant:
    def __init__(self, model, exceptions_path):
        self.model_data = model
        self.exceptions_path = exceptions_path
        self.debug = False

    def write_exception(self, input_message, predicted_tag, confidence):
        with open(self.exceptions_path, "a") as f:
            f.write(f"Message: {input_message}, Predicted Tag: {predicted_tag}, Confidence: {confidence}\n")

    def process_message(self, input_message) -> tuple[str | None, str]:
        words = DataPreprocessor.tokenize_and_lemmatize(input_message)
        bag = DataPreprocessor.bag_of_words(words, self.model_data.vocabulary)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        self.model_data.intent_classifier.eval()
        with torch.no_grad():
            logits = self.model_data.intent_classifier(bag_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence = torch.max(probabilities).item()
        
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_intent = self.model_data.intents[predicted_class_index]

        # Only respond if confidence is high enough
        if (self.debug):
            print(f"{YELLOW}Debug: Predicted {predicted_intent} (confidence: {confidence:.3f}){RESET}")
        if confidence < 0.8:
            self.write_exception(input_message, predicted_intent, confidence)
            return (None, "I'm not sure what you mean. Can you rephrase?")

        # Generate response with entity substitution
        if self.model_data.intents_responses[predicted_intent]:
            return (predicted_intent, random.choice(self.model_data.intents_responses[predicted_intent]))

        return (None, "I'm not sure how to respond to that.")
