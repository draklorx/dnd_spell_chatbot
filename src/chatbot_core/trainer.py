import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from .models.intent_classifier import IntentClassifier
from .models.model_data import ModelData

class Trainer:
    def __init__(self, intents_path):
        self.model_data = ModelData();
        self.intents_path: str = intents_path

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.model_data.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.model_data.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model_data.intent_classifier = IntentClassifier(self.model_data.X.shape[1], len(self.model_data.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_data.intent_classifier.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_data.intent_classifier(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss

            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")


    def train_and_save(self, model_path, model_data_path, intents_path):
        self.model_data.parse_intents(intents_path)
        self.model_data.prepare_data()
        self.train_model(batch_size=8, lr=0.001, epochs=100)

        self.model_data.save_model(
            model_path,
            model_data_path
        )
        print("Model retrained and saved.")
