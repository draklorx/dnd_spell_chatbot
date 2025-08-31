# D&D Spell Chatbot

A fantasy-themed conversational AI chatbot built with PyTorch and NLTK that simulates interactions with a tavern keeper at the Drunken Dragon Inn. The bot can engage in natural conversations about drinks, tavern rumors, and general fantasy-themed topics.

## Features

- **Natural Language Processing**: Built using PyTorch neural networks with NLTK for text preprocessing
- **Fantasy Roleplay**: Immersive tavern keeper persona with D&D-themed responses
- **Intent Recognition**: Handles multiple conversation intents including greetings, drink orders, rumors, and farewells
- **Pre-trained Model**: Comes with a trained model ([chatbot_model.pth](chatbot_model.pth)) ready for immediate use

## Project Structure

```
dnd_spell_chatbot/
├── main.py                 # Main application entry point
├── ChatbotModel.py         # Neural network model definition
├── ChatbotAssistant.py     # Chatbot logic and conversation handling
├── intents.json           # Training data with conversation patterns and responses
├── chatbot_model.pth      # Pre-trained PyTorch model
├── dimensions.json        # Model dimensions and vocabulary data
├── pyproject.toml         # Project dependencies and configuration
└── README.md              # This file
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

3. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

Run the chatbot application:

```bash
python main.py
```

The chatbot will start and you can begin conversing with the tavern keeper. Try phrases like:
- "Hello there"
- "What do you have to drink?"
- "Any interesting rumors?"
- "Goodbye"

## Training Data

The bot is trained on fantasy tavern scenarios defined in [intents.json](intents.json), including:

- **Greetings**: Welcome messages from the tavern keeper
- **Drink Orders**: Information about available beverages
- **Rumors**: Fantasy world gossip and stories
- **Farewells**: Polite goodbye messages

## Model Architecture

The chatbot uses a neural network implemented in [ChatbotModel.py](ChatbotModel.py) with:
- Input layer for tokenized text
- Hidden layers for pattern recognition
- Output layer for intent classification
- Trained weights stored in [chatbot_model.pth](chatbot_model.pth)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add new intents to [intents.json](intents.json) if needed
5. Retrain the model if necessary
6. Submit a pull request

## License

This project is open source. Please check the license file for more details.

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed correctly
- **Model Loading Issues**: Verify that [chatbot_model.pth](chatbot_model.pth) exists and isn't corrupted
- **NLTK Errors**: Make sure required NLTK data packages are downloaded

For more issues, check [exceptions.log](exceptions.log) for detailed error messages.