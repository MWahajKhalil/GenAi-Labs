

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


#N GRAM
def generate_ngrams(corpus_data, n=3):
    ngrams = []
    for sentence in corpus_data:
        words = nltk.word_tokenize(sentence)
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i+n]))
    return ngrams

print("\n--- N-grams (n=3) ---")
n_grams = generate_ngrams(corpus, n=3)
for gram in n_grams[:5]:
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

    