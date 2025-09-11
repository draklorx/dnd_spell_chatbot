import random
from .models.spacy_intent_classifier import SpacyIntentClassifier

# Define colors directly to avoid circular imports
YELLOW = "\033[93m"
RESET = "\033[0m"

class Assistant:
    def __init__(self, model, exceptions_path):
        self.intent_classifier = model
        self.exceptions_path = exceptions_path
        self.debug = False

    def write_exception(self, input_message, predicted_tag, confidence):
        with open(self.exceptions_path, "a") as f:
            f.write(f"Message: {input_message}, Predicted Tag: {predicted_tag}, Confidence: {confidence}\n")

    def process_message(self, input_message) -> tuple[str | None, str]:
        # Use spaCy intent classifier for prediction
        predicted_intent, confidence = self.intent_classifier.predict(input_message)

        # Only respond if confidence is high enough
        if self.debug:
            print(f"{YELLOW}Debug: Predicted {predicted_intent} (confidence: {confidence:.3f}){RESET}")
            
        if confidence < 0.6:  # Lowered from 0.8 for better spaCy performance
            self.write_exception(input_message, predicted_intent, confidence)
            return (None, "I'm not sure what you mean. Can you rephrase?")

        # Generate response
        if predicted_intent:
            response = self.intent_classifier.get_response(predicted_intent)
            return (predicted_intent, response)

        return (None, "I'm not sure how to respond to that.")
