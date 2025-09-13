# AI Chatbot with PyTorch

A neural network-based chatbot built with PyTorch that can understand natural language and provide intelligent responses across multiple domains including greetings, payments, travel, fitness, cooking, and more.

## 🤖 About the Project

This chatbot uses a simple yet effective neural network architecture to classify user intents and provide appropriate responses. The model is trained on a comprehensive dataset of conversation patterns and can handle various topics from everyday conversations to specific domain queries.

## ✨ Features

- **Intent Classification**: Recognizes 20+ different conversation intents
- **Natural Language Processing**: Uses NLTK for text preprocessing and tokenization
- **Neural Network**: Custom PyTorch neural network for intent prediction
- **Bag of Words**: Advanced text representation for better understanding
- **Confidence Threshold**: Only responds when confident about the intent (>75%)
- **Extensible Design**: Easy to add new intents and responses
- **Command Line Interface**: Interactive chat experience

## 🧠 Supported Conversation Topics

The chatbot can assist with:

- **General Conversations**: Greetings, goodbyes, thanks
- **Payments**: Payment methods and options
- **Travel**: Car rental and travel inquiries
- **Pet Care**: Pet grooming and care tips
- **Sports**: Sports updates and news
- **Entertainment**: Movie and book recommendations
- **Education**: Study tips and learning advice
- **Finance**: Investment guidance
- **Career**: Job hunting and resume tips
- **Health & Fitness**: Exercise and wellness advice
- **Cooking**: Recipes and cooking help
- **Travel Planning**: Vacation ideas and destinations
- **Language Learning**: Tips for learning new languages
- **Mental Health**: Stress management and well-being
- **Home Improvement**: DIY projects and organization
- **Childcare**: Baby care and parenting tips

## 🛠️ Project Structure

```
chatbot-project/
│
├── chat.py                   # Main chatbot interface
├── train.py                  # Model training script
├── model.py                  # Neural network architecture
├── nltk_utils.py            # Text preprocessing utilities
├── intents.json             # Training data and responses
├── data.pth                 # Trained model weights
└── README.md               # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- PyTorch
- NLTK
- NumPy

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chatbot-project
   ```

2. **Install dependencies**
   ```bash
   pip install torch nltk numpy
   ```

3. **Download NLTK data**
   ```python
   import nltk
   nltk.download('punkt')
   ```

### Usage

#### Running the Chatbot

```bash
python chat.py
```

The chatbot will start an interactive session:
```
Let's chat! (type 'quit' to exit)
You: Hello
Bots: Hi there! How can I help you?
You: Can you recommend a movie?
Bots: How about watching a classic like 'Inception' or 'The Dark Knight'? What genre do you like?
You: quit
```

#### Training the Model

If you want to retrain the model with new data:

```bash
python train.py
```

This will:
- Load intents from `intents.json`
- Preprocess the text data
- Train the neural network
- Save the trained model to `data.pth`

## 🔧 Model Architecture

### Neural Network

The chatbot uses a simple feedforward neural network:

```python
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer
        self.relu = nn.ReLU()                          # Activation
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer
```

### Text Processing Pipeline

1. **Tokenization**: Splits text into individual words
2. **Stemming**: Reduces words to their root form
3. **Bag of Words**: Converts text to numerical vectors
4. **Intent Classification**: Predicts the most likely intent
5. **Response Selection**: Randomly selects from appropriate responses

### Training Parameters

- **Epochs**: 1000
- **Batch Size**: 8
- **Learning Rate**: 0.001
- **Hidden Layer Size**: 8
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

## 📝 Customizing the Chatbot

### Adding New Intents

1. **Edit `intents.json`**:
   ```json
   {
     "tag": "your_new_intent",
     "patterns": [
       "Example user input 1",
       "Example user input 2"
     ],
     "responses": [
       "Bot response 1",
       "Bot response 2"
     ]
   }
   ```

2. **Retrain the model**:
   ```bash
   python train.py
   ```

3. **Test your new intent**:
   ```bash
   python chat.py
   ```

### Adjusting Confidence Threshold

In `chat.py`, modify line 46 to change the confidence threshold:
```python
if prob.item() > 0.75:  # Change 0.75 to your desired threshold
```

## 📊 Performance

- **Training Time**: ~30 seconds on CPU
- **Model Size**: ~50KB
- **Response Time**: <100ms
- **Accuracy**: >90% on training intents

## 🔍 How It Works

1. **Input Processing**: User input is tokenized and stemmed
2. **Feature Extraction**: Text is converted to bag-of-words vector
3. **Intent Prediction**: Neural network classifies the intent
4. **Confidence Check**: Only responds if confidence > 75%
5. **Response Selection**: Randomly picks from matching responses

## 🐛 Troubleshooting

### Common Issues

**NLTK Data Missing**:
```bash
python -c "import nltk; nltk.download('punkt')"
```

**Model Not Found**:
```bash
python train.py  # Retrain the model
```

**Low Confidence Responses**:
- Add more training patterns to `intents.json`
- Lower the confidence threshold in `chat.py`
- Retrain the model

## 🚀 Future Enhancements

- [ ] Web interface with Flask/FastAPI
- [ ] Context awareness for multi-turn conversations
- [ ] Integration with external APIs
- [ ] Voice input/output capabilities
- [ ] Sentiment analysis
- [ ] Multi-language support
- [ ] Database integration for conversation history

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Add new intents or improve the model architecture
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## 📚 Technologies Used

- **PyTorch**: Deep learning framework
- **NLTK**: Natural language processing
- **NumPy**: Numerical computations
- **JSON**: Data storage format
- **Python**: Core programming language

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- PyTorch community for the excellent deep learning framework
- NLTK team for natural language processing tools
- Open source community for inspiration and resources

## 📞 Contact

For questions, suggestions, or issues, please open an issue in the repository.

---

*Start chatting and experience the power of AI conversation! 🤖💬*
