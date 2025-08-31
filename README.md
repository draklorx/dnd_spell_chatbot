# D&D Spell Chatbot

A fantasy-themed conversational AI chatbot built with PyTorch and NLTK that simulates interactions with a tavern keeper at the Drunken Dragon Inn. The bot can engage in natural conversations about drinks, tavern rumors, and general fantasy-themed topics.

## Features

- **Natural Language Processing**: Built using PyTorch neural networks with NLTK for text preprocessing
- **Fantasy Roleplay**: Immersive tavern keeper persona with D&D-themed responses
- **Intent Recognition**: Handles multiple conversation intents including greetings, drink orders, rumors, and farewells
- **Model Persistence**: Automatically saves and loads trained models

## Project Structure

```
dnd_spell_chatbot/
├── src/
│   └── dnd_spell_chatbot/
│       ├── main.py                    # Main application entry point
│       ├── chatbot_model.py           # Neural network model definition
│       ├── chatbot_assistant.py       # Chatbot logic and conversation handling
│       └── data/
│           └── intents.json           # Training data with conversation patterns and responses
├── artifacts/                         # Generated model files directory
│   ├── chatbot_model.pth             # Trained PyTorch model (generated)
│   └── dimensions.json               # Model dimensions and vocabulary data (generated)
├── logs/                             # Application logs directory
│   └── exceptions.log                # Low confidence message log (generated)
├── pyproject.toml                    # Project dependencies and configuration
├── uv.lock                          # Dependency lock file
└── README.md                        # This file
```

## Requirements

- Python 3.13+
- PyTorch 2.8.0+
- NLTK 3.9.1+
- NumPy 2.3.2+

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd dnd_spell_chatbot
```

2. Install dependencies using uv (recommended) or pip:

```bash
# Using uv
uv sync

# Or using pip
pip install torch nltk numpy
```

## Usage

Run the chatbot application from the project root:

```bash
python src/dnd_spell_chatbot/main.py
```

The chatbot will automatically:

- Download required NLTK data on first run
- Train and save a model if no pre-trained model exists
- Load the existing model for subsequent runs

You can then begin conversing with the tavern keeper. Try phrases like:

- "Hello there"
- "What do you have to drink?"
- "Any interesting rumors?"
- "Goodbye"

### Special Commands

- Type `/retrain` to retrain the model with the current training data

## Training Data

The bot is trained on fantasy tavern scenarios defined in [src/dnd_spell_chatbot/data/intents.json](src/dnd_spell_chatbot/data/intents.json), including:

- **Greetings**: Welcome messages from the tavern keeper
- **Drink Orders**: Information about available beverages
- **Food Requests**: Tavern meal offerings
- **Gossip**: Fantasy world rumors and stories
- **Farewells**: Polite goodbye messages

## Model Architecture

The chatbot uses a neural network implemented in [`ChatbotModel`](src/dnd_spell_chatbot/chatbot_model.py) with:

- Input layer for bag-of-words representation
- Two hidden layers (128 and 64 neurons) with ReLU activation and dropout
- Output layer for intent classification
- Trained weights automatically saved to `artifacts/chatbot_model.pth`

## Confidence Threshold

The chatbot uses a confidence threshold of 0.9. Messages with lower confidence are logged to `logs/exceptions.log` and receive a generic "I'm not sure what you mean" response.

## License

This project is open source. Please check the license file for more details.

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed correctly
- **Model Loading Issues**: Delete the `artifacts/` directory to force model retraining
- **NLTK Errors**: The application automatically downloads required NLTK data

For additional training data, check `logs/exceptions.log` for messages that the bot couldn't understand with high confidence.
