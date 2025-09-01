import os
import json
import random

import nltk
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .chatbot_model import ChatbotModel

# Download if needed (NLTK is smart about not re-downloading)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

class Assistant:
    def __init__(self, intents_path, exceptions_path, function_mappings=None):
        self.model = None
        self.intents_path: str = intents_path
        self.exceptions_path: str = exceptions_path

        # training data representing lemmatized patterns from intents.json and the tag they are associated with
        self.documents: list[tuple[list[str], str]] = []
        # a sorted list of unique lemmatized words generated from every pattern in the training data
        self.vocabulary: list[str] = []
        # a list of every tag in intents.json
        self.intents: list[str] = []
        # a dictionary of responses mapping each intent tag to its responses
        self.intents_responses: dict[str, list[str]] = {}

        self.function_mappings = function_mappings

        self.X = None
        self.y = None

        
    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        
        words = nltk.word_tokenize(text)
        # Keep and lemmatize words that contain at least one alphanumeric character
        words = [lemmatizer.lemmatize(word.lower()) for word in words if any(c.isalnum() for c in word)]
        
        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]
    
    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss

            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, model_data_path):
        torch.save(self.model.state_dict(), model_path)

        with open(model_data_path, 'w') as f:
            json.dump({
                'input_size': self.X.shape[1],
                'output_size': len(self.intents),
                'intents': self.intents,
                'intents_responses': self.intents_responses,
                'vocabulary': self.vocabulary
            }, f)

    def load_model(self, model_path, model_data_path):
        with open(model_data_path, 'r') as f:
            data = json.load(f)
        
        # Load all necessary data from the saved file
        self.intents = data['intents']
        self.intents_responses = data['intents_responses'] 
        self.vocabulary = data['vocabulary']
        
        self.model = ChatbotModel(data['input_size'], data['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def train_and_save(self, model_path, model_data_path):
        self.parse_intents()
        self.prepare_data()
        self.train_model(batch_size=8, lr=0.001, epochs=100)
        
        self.save_model(
            model_path,
            model_data_path
        )
        print("Model retrained and saved.")

    def write_exception(self, input_message, predicted_tag, confidence):
        with open(self.exceptions_path, "a") as f:
            f.write(f"Message: {input_message}, Predicted Tag: {predicted_tag}, Confidence: {confidence}\n")

    def process_message(self, input_message) -> tuple[str | None, str]:
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(bag_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence = torch.max(probabilities).item()
        
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        # Only respond if confidence is high enough
        if confidence < 0.7:
            self.write_exception(input_message, predicted_intent, confidence)
            return (None, "I'm not sure what you mean. Can you rephrase?")

        # Generate response with entity substitution
        if self.intents_responses[predicted_intent]:
            return (predicted_intent, random.choice(self.intents_responses[predicted_intent]))

        return (None, "I'm not sure how to respond to that.")
