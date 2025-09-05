import nltk

nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

class DataPreprocessor:    
    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words if any(c.isalnum() for c in word)]
        return words
    
    @staticmethod
    def bag_of_words(words, vocabulary):
        return [1 if word in words else 0 for word in vocabulary]