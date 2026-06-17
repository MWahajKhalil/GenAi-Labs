

from nltk import corpus
paragraph = """Born in Lahore, he graduated from Keble College, Oxford. He began his international cricket career in a 1971 Test series against England. He advocated for neutral umpiring during his captaincy. He led Pakistan to its first-ever Test series victories in India and England during 1987. Playing until 1992, he captained the Pakistan national cricket team for most of the 1980s and early 1990s. In addition to achieving the all-rounder's triple of scoring 3,000 runs and taking 300 wickets in Tests, he holds the world record for the most wickets as a captain in Test cricket, along with the second-best bowling figures in an innings. Moreover, he has won the most Player of the Series awards in Test cricket for Pakistan and ranks fourth overall in Test history. In 2009, he was inducted into the ICC Cricket Hall of Fame."""



#-----tokenization-----
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt') 
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


def tokenization(paragraph):
    tokenized_sentence = nltk.sent_tokenize(paragraph)
    tokenized_words = nltk.word_tokenize(paragraph)
    return tokenized_sentence, tokenized_words








sentence, words = tokenization(paragraph)


# lemmatization


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()



import re
corpus =[]
for i in range(len(sentence)):
    review = re.sub('[^a-zA-Z]', ' ', sentence[i])
    print(review)
    review = review.lower()
    corpus.append(review)




#stemming 

stemmer = PorterStemmer()
for i in range(len(corpus)):
    stemmed_word = []
    for word in corpus[i].split():
        if word not in stopwords.words('english'):
            stemmed_word.append(stemmer.stem(word))
    




#lemmitization
lemmatizer = WordNetLemmatizer()
for i in range(len(corpus)):
    lemma_word = []
    words = nltk.word_tokenize(corpus[i])
    for word in words:
        if word not in stopwords.words('english'):
            lemma_word.append(lemmatizer.lemmatize(word))
    print(lemma_word)
    




#bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus)

print(cv.vocabulary_)
print("\n\n")
print(x.toarray())



# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
y = tfidf.fit_transform(corpus)
print("\n\n")
print(y.toarray())


#Text similarity using cosine similarity
import numpy as np

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    return dot_product / (norm_vector1 * norm_vector2)

# Compareing the first sentence (i = 0) and the second sentence (i= 1)
print("\nCosine Similarity (Sentence 1 vs Sentence 2):")
print(cosine_similarity(y.toarray()[0], y.toarray()[1]))


#Word embeddings
import pandas as pd
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

print(model.wv.similarity('imran', 'khan'))



#CONTINOUS BAG OF WORDs
def generate_cbow_data(corpus_data, window_size=2):
    """
    Generate Continuous Bag of Words (CBOW) training data.
    Takes a list of sentences (where each sentence is a string) and window_size.
    Returns pairs of (context_words, target_word).
    """
    cbow_data = []
    for sentence in corpus_data:
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence) if isinstance(sentence, str) else sentence
        
        for i, target in enumerate(words):
            context = []
            # Previous words (window_size before)
            for j in range(max(0, i - window_size), i):
                context.append(words[j])
            # Next words (window_size after)
            for j in range(i + 1, min(len(words), i + window_size + 1)):
                context.append(words[j])
            
            cbow_data.append((context, target))
    return cbow_data

#generate_cbow_data(corpus)
cbow_pairs = generate_cbow_data(corpus, window_size=2)
for pair in cbow_pairs[:5]:
    print(f"Context: {pair[0]} -> Target: {pair[1]}")


# --- SKIP-GRAM ---
def generate_skipgram_data(corpus_data, window_size=2):
    """
    Generate Skip-Gram training data.
    Takes a list of sentences (where each sentence is a string) and window_size.
    Returns pairs of (target_word, context_word).
    """
    skipgram_data = []
    for sentence in corpus_data:
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence) if isinstance(sentence, str) else sentence
        
        for i, target in enumerate(words):
            # Previous words (window_size before)
            for j in range(max(0, i - window_size), i):
                skipgram_data.append((target, words[j]))
            # Next words (window_size after)
            for j in range(i + 1, min(len(words), i + window_size + 1)):
                skipgram_data.append((target, words[j]))
            
    return skipgram_data

print("\n--- Skip-Gram Data (first 5 pairs) ---")
skipgram_pairs = generate_skipgram_data(corpus, window_size=2)
for pair in skipgram_pairs[:5]:
    print(f"Target: {pair[0]} -> Context Word: {pair[1]}")

#i have implemented a function to generate n-grams from a given corpus of text. The function takes in the corpus data and the value of n (the number of items in the n-gram) and returns a list of n-grams. Each n-gram is represented as a tuple of words. The function uses the NLTK library to tokenize the sentences and words in the corpus, and then constructs the n-grams based on the specified window size.
def generate_ngrams(corpus_data, n=3):
    ngrams = []
    for sentence in corpus_data:
        words = nltk.word_tokenize(sentence)
        for i in range(len(words) - n + 1): #here n is the number of items in the n-gram, so we need to ensure that we have enough words to create an n-gram. The loop runs until len(words) - n + 1 to avoid index out of range errors when creating n-grams. for example, if we have a sentence with 5 words and we want to create trigrams (n=3), the loop will run until index 2 (5 - 3 + 1 = 3) to create the trigrams (0,1,2), (1,2,3), and (2,3,4). If we tried to run the loop until len(words), we would encounter an index out of range error when trying to access words[i+n-1] for the last few iterations.
            ngrams.append(tuple(words[i:i+n]))
    return ngrams

print("\n--- N-grams (n=3) ---")
n_grams = generate_ngrams(corpus, n=3)
for gram in n_grams[:5]: #here we are printing the first 5 n-grams generated from the corpus. Each n-gram is represented as a tuple of words. The function generate_ngrams takes in the corpus data and the value of n (the number of items in the n-gram) and returns a list of n-grams. The loop iterates through the corpus, tokenizes each sentence into words, and constructs the n-grams based on the specified window size (n). The first 5 n-grams are printed to give an example of the output generated by the function.  
    print(gram) 



#part of speech tagging
from nltk import pos_tag
tokens = nltk.word_tokenize(paragraph)
pos_tags = pos_tag(tokens)
print(pos_tags)

#named entity recognition
from nltk import ne_chunk
named_entities = ne_chunk(pos_tags)
print(named_entities)   




#dependency parsing
from nltk.parse import CoreNLPParser
parser = CoreNLPParser(url='http://localhost:9000')
sentences = nltk.sent_tokenize(paragraph)
for sentence in sentences:
    parse_tree = next(parser.raw_parse(sentence))
    print(parse_tree)   


#sentiment analysis     
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sentences = nltk.sent_tokenize(paragraph)
for sentence in sentences:
    sentiment = sia.polarity_scores(sentence)
    print(f"Sentence: {sentence}\nSentiment: {sentiment}\n")    

#these above code snippets demonstrate various NLP techniques using the NLTK library, including tokenization, stemming, lemmatization, bag of words, TF-IDF, cosine similarity, word embeddings, CBOW and Skip-Gram data generation, n-grams, part of speech tagging, named entity recognition, dependency parsing, and sentiment analysis. Each technique is applied to a sample paragraph about Imran Khan's cricket career
#all important techniques are covered in this code snippet and it serves as a comprehensive guide for anyone looking to understand and implement NLP techniques using NLTK. REMAINING ARE SOME TECHNIQUES LIKE TOPIC MODELING, LANGUAGE MODELING, MACHINE TRANSLATION, TEXT GENERATION, etc. WHICH CAN BE IMPLEMENTED USING OTHER LIBRARIES LIKE Gensim, Hugging Face Transformers, etc.


### Text Classification with NLTK (using Naive Bayes)  
def get_words_in_categories(word_list):
    positive_words = ['happy', 'joyful', 'excellent', 'wonderful', 'amazing', 'great', 'good', 'positive']
    negative_words = ['sad', 'unhappy', 'terrible', 'awful', 'bad', 'negative']
    neutral_words = ['the', 'is', 'in', 'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'because', 'as', 'that', 'which', 'this', 'it', 'its', 'to', 'for', 'of', 'on', 'at', 'by', 'with', 'about', 'from', 'into', 'through', 'during', 'before', 'after', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
    return [word for word in word_list if word.lower() in positive_words + negative_words + neutral_words]


def classifier():
    positive_words = get_words_in_categories(['happy', 'joyful', 'excellent', 'wonderful', 'amazing', 'great', 'good', 'positive'])
    negative_words = get_words_in_categories(['sad', 'unhappy', 'terrible', 'awful', 'bad', 'negative'])
    neutral_words = get_words_in_categories(['the', 'is', 'in', 'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'because', 'as', 'that', 'which', 'this', 'it', 'its', 'to', 'for', 'of', 'on', 'at', 'by', 'with', 'about', 'from', 'into', 'through', 'during', 'before', 'after', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
    return positive_words, negative_words, neutral_words    



#text classification
positive_words, negative_words, neutral_words = classifier()

from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

def create_classifier():
    # Sample training data (you would use a larger, more diverse dataset in practice)
    training_data = [
        (['happy', 'joyful', 'excellent', 'wonderful', 'amazing', 'great', 'good', 'positive'], 'positive'),
        (['sad', 'unhappy', 'terrible', 'awful', 'bad', 'negative'], 'negative'),
        (['the', 'is', 'in', 'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'because', 'as', 'that', 'which', 'this', 'it', 'its', 'to', 'for', 'of', 'on', 'at', 'by', 'with', 'about', 'from', 'into', 'through', 'during', 'before', 'after', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'], 'neutral'),
    ]
    
    # Create a Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(training_data)
    return classifier



#text classification example
classifier = create_classifier()
text = input("Enter the Text to get Sentiment Analysis: ")
text_tokenized = word_tokenize(text)
text_filtered = [word for word in text_tokenized if word.lower() not in stop_words]
text_stemmed = stemming(text)
text_lemmatized = lemmatize(text)

#calculate sentiment scores for positive, negative, and neutral words
positive_score = sum(1 for word in text_stemmed if word.lower() in positive_words)
negative_score = sum(1 for word in text_stemmed if word.lower() in negative_words)
neutral_score = sum(1 for word in text_stemmed if word.lower() in neutral_words)

#determine sentiment
if positive_score > negative_score and positive_score > neutral_score:
    sentiment = 'positive'
elif negative_score > positive_score and negative_score > neutral_score:
    sentiment = 'negative'
else:
    sentiment = 'neutral'

print("Sentiment:", sentiment)


