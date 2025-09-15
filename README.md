# D&D Spell Chatbot

A specialized conversational AI assistant built with PyTorch and NLTK that helps users retrieve and understand information about Dungeons & Dragons spells. This NLP-powered chatbot can recognize spell names, provide descriptions, and answer spell-related questions. It is built on top of an embeddings database and can retrieve specific parts of spell descriptions based on the user's query.
Code originally sourced from NeuralNine's [`Youtube`](https://www.youtube.com/watch?v=a040VmmO-AY&ab_channel=NeuralNine) and [`github repository`](https://github.com/NeuralNine/youtube-tutorials/tree/main/AI%20Chatbot%20PyTorch).

## Features

-   **Natural Language Processing**: Built using PyTorch neural networks with NLTK for text preprocessing
-   **Named Entity Recognition**: Custom NER system to identify D&D spell names in user queries built with fuzzywuzzy
-   **Context-Aware Responses**: Dynamic responses that incorporate spell data into conversation, as well as referencing previous discussed spell
-   **Intent Classification**: Handles multiple conversation intents related to spell information
-   **Model Persistence**: Automatically saves and loads trained models
-   **Vector DB**: A vector DB is included to pull the most relevant sentences from the spell's description matching the user's query

## Project Structure

```
dnd_spell_chatbot/
├── src/
│   ├── main.py                       # Main application entry point
│   ├── train.py                      # Model training script
│   ├── chatbot_dnd_spells/           # D&D specific implementation
│   │   ├── __init__.py
│   │   ├── chatbot.py                # D&D specific chatbot implementation
│   │   ├── chatbot_config.py         # Configuration settings
│   │   ├── chatbot_trainer.py        # Training functionality
│   │   ├── data_processor.py         # Data processing utilities
│   │   ├── spell__vector_searcher.py # Vector search for spells
│   │   └── spell_entity_classifier.py # Spell entity classification
│   ├── coreference_resolution/       # Coreference resolution module
│   ├── embeddings/                   # Vector DB code
│   ├── entity_recognition/           # Entity recognition module
│   ├── intents/                      # Intent classification module
│   └── utils/                        # Utility functions
├── pyproject.toml                    # Project dependencies and configuration
├── uv.lock                           # Dependency lock file
├── requirements.txt                  # Python dependencies
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

Before you begin you should run the trainer to setup the db and run the training

```bash
# Using uv
uv run src/train.py

# Or using venv after you've activated the environment
python src/train.py
```

Run the chatbot application from the project root using uv (recommended) or inside venv (if installed with pip):

```bash
# Using uv
uv run src/main.py

# Or using venv after you've activated the environment
python src/main.py
```

The trainer will automatically:

-   Download required NLTK data on first run
-   Train and save a model
-   Create tables for vector DB and create embeddings for every spell passed in

The chatbot will automatically

-   Load the existing model
-   Detect if there's been changes to the training data and request that you train first
-   Try to answer the user's question based on intents first, and then failover to a vector DB search

You can then begin asking about D&D spells. Try phrases like:

-   "Tell me about Fireball"
-   "How does Magic Missile work?"
-   "What level is Cure Wounds?"
-   "What's the casting time for Wish?"

### Special Commands

-   Type `/quit` to exit the application

## Training Data

The bot is trained on D&D spell-focused conversation patterns defined in training data files, including:

-   **Spell Information**: A complete rundown of the spell
-   **Specific Spell Details**: Answering specific questions about the spell like casting time, range, or level

The spell information is retrieved from spell database files which contain a comprehensive database of D&D spells.

## Model Architecture

The chatbot uses a neural network with:

-   Input layer for bag-of-words representation
-   Hidden layers with ReLU activation and dropout for regularization
-   Output layer for intent classification
-   Trained weights automatically saved to model files

## Confidence Threshold

The chatbot uses a confidence threshold of 0.7. Messages with lower confidence are logged to `logs/exceptions.log` and receive a clarification response.

## License

This project is open source. Please check the [`LICENSE.md`](LICENSE.md) file for more details.

## Issues / Future Development

Find out what's next on the [`github issues board`](https://github.com/draklorx/dnd_spell_chatbot/issues).
