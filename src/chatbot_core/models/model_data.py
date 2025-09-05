import os
import json
import numpy as np
import torch
from .intent_classifier import IntentClassifier
from ..utils.data_preprocessor import DataPreprocessor

class ModelData:
    def __init__(self):
        self.intent_classifier = None

        # training data representing lemmatized patterns from intents.json and the tag they are associated with
        self.documents: list[tuple[list[str], str]] = []
        # a sorted list of unique lemmatized words generated from every pattern in the training data
        self.vocabulary: list[str] = []
        # a list of every tag in intents.json
        self.intents: list[str] = []
        # a dictionary of responses mapping each intent tag to its responses
        self.intents_responses: dict[str, list[str]] = {}

        self.X = None
        self.y = None

    def parse_intents(self, intents_path):
        if os.path.exists(intents_path):
            with open(intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = DataPreprocessor.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = DataPreprocessor.bag_of_words(words, self.vocabulary)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)
        
    def save_model(self, model_path, model_data_path):
        torch.save(self.intent_classifier.state_dict(), model_path)

        with open(model_data_path, 'w') as f:
            json.dump({
                'input_size': self.X.shape[1],
                'output_size': len(self.intents),
                'intents': self.intents,
                'intents_responses': self.intents_responses,
                'vocabulary': self.vocabulary
            }, f)

    @staticmethod
    def load_model(model_path, model_data_path):
        model_data = ModelData()
        with open(model_data_path, 'r') as f:
            data = json.load(f)
        
        # Load all necessary data from the saved file
        model_data.intents = data['intents']
        model_data.intents_responses = data['intents_responses']
        model_data.vocabulary = data['vocabulary']

        model_data.intent_classifier = IntentClassifier(data['input_size'], data['output_size'])
        model_data.intent_classifier.load_state_dict(torch.load(model_path, weights_only=True))

        return model_data
