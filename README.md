# D&D Spell Chatbot

A specialized conversational AI assistant built with PyTorch and NLTK that helps users retrieve and understand information about Dungeons & Dragons spells. This NLP-powered chatbot can recognize spell names, provide descriptions, and answer spell-related questions.
Code originally sourced from NeuralNine's [`Youtube`](https://www.youtube.com/watch?v=a040VmmO-AY&ab_channel=NeuralNine) and [`github repository`](https://github.com/NeuralNine/youtube-tutorials/tree/main/AI%20Chatbot%20PyTorch).

## Features

- **Natural Language Processing**: Built using PyTorch neural networks with NLTK for text preprocessing
- **Named Entity Recognition**: Custom NER system to identify D&D spell names in user queries built with fuzzywuzzy
- **Context-Aware Responses**: Dynamic responses that incorporate spell data into conversation, as well as referencing previous discussed spell
- **Intent Classification**: Handles multiple conversation intents related to spell information
- **Model Persistence**: Automatically saves and loads trained models

## Project Structure

```
dnd_spell_chatbot/
├── src/
│   ├── main.py                       # Main application entry point
│   ├── chatbot_core/                 # Core chatbot framework
│   │   ├── __init__.py
│   │   ├── assistant.py              # Core NLP processing and model management
│   │   ├── chatbot_interface.py      # Interface definition for chatbots
│   │   ├── chatbot_model.py          # Neural network model definition
│   │   └── ner_interface.py          # Interface for named entity recognition
│   └── dnd_spell_chatbot/            # D&D specific implementation
│       ├── __init__.py
│       ├── chatbot.py                # D&D specific chatbot implementation
│       ├── spell_ner.py              # Spell name entity recognition
│       ├── artifacts/                # Generated model files
│       │   ├── chatbot_model.pth     # Trained PyTorch model (generated)
│       │   └── model_data.json       # Saved model data so we don't have to rebuild it (generated)
│       ├── data/                     # Training and reference data
│       │   ├── intents.json          # Training data with patterns and responses
│       │   └── spells.json           # D&D spell information database
│       └── logs/                     # Application logs directory
│           └── exceptions.log        # Low confidence message log
├── pyproject.toml                    # Project dependencies and configuration
├── uv.lock                           # Dependency lock file
└── README.md                         # This file
```

## Installation

1. Clone the repository:

```bash
git clone git@github.com:draklorx/dnd_spell_chatbot.git
cd dnd_spell_chatbot
```

2. Install dependencies using uv (recommended) or pip:

```bash
# Using uv
uv sync

# Or using pip
python -m venv
# windows
./.venv/Scripts/activate
# linux/mac
source ./.venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the chatbot application from the project root using uv (recommended) or inside venv (if installed with pip):

```bash
# Using uv
uv run src/main.py

# Or using venv after you've activated the environment
python src/main.py
```

The chatbot will automatically:

- Download required NLTK data on first run
- Train and save a model if no pre-trained model exists
- Load the existing model for subsequent runs

You can then begin asking about D&D spells. Try phrases like:

- "Tell me about Fireball"
- "How does Magic Missile work?"
- "What level is Cure Wounds?"
- "What's the casting time for Wish?"

### Special Commands

- Type `/retrain` to retrain the model with the current training data
- Type `/quit` to exit the application

## Training Data

The bot is trained on D&D spell-focused conversation patterns defined in [`src/dnd_spell_chatbot/data/intents.json`](src/dnd_spell_chatbot/data/intents.json), including:

- **Spell Information**: A complete rundown of the spell
- **Specific Spell Details**: Answering specific questions about the spell like casting time, range, or level

The spell information is retrieved from [`src/dnd_spell_chatbot/data/spells.json`](src/dnd_spell_chatbot/data/spells.json) which contains a comprehensive database of D&D spells. This data was sourced from [`dmcb / srd-5.2-spells.json`](https://gist.github.com/dmcb/4b67869f962e3adaa3d0f7e5ca8f4912)

## Model Architecture

The chatbot uses a neural network implemented in [`ChatbotModel`](src/chatbot_core/chatbot_model.py) with:

- Input layer for bag-of-words representation
- Hidden layers with ReLU activation and dropout for regularization
- Output layer for intent classification
- Trained weights automatically saved to [`src/dnd_spell_chatbot/artifacts/chatbot_model.pth`](src/dnd_spell_chatbot/artifacts/chatbot_model.pth)

## Confidence Threshold

The chatbot uses a confidence threshold of 0.7. Messages with lower confidence are logged to `logs/exceptions.log` and receive a clarification response.

## License

This project is open source. Please check the [`LICENSE.md`](LICENSE.md) file for more details.

## Issues / Future Development

Find out what's next on the [`github issues board`](https://github.com/draklorx/dnd_spell_chatbot/issues).
