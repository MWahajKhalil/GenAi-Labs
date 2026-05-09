

from nltk import corpus
paragraph = """Born in Lahore, he graduated from Keble College, Oxford. He began his international cricket career in a 1971 Test series against England. He advocated for neutral umpiring during his captaincy. He led Pakistan to its first-ever Test series victories in India and England during 1987. Playing until 1992, he captained the Pakistan national cricket team for most of the 1980s and early 1990s. In addition to achieving the all-rounder's triple of scoring 3,000 runs and taking 300 wickets in Tests, he holds the world record for the most wickets as a captain in Test cricket, along with the second-best bowling figures in an innings. Moreover, he has won the most Player of the Series awards in Test cricket for Pakistan and ranks fourth overall in Test history. In 2009, he was inducted into the ICC Cricket Hall of Fame."""



#-----tokenization-----
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
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
print(x[0].toarray())


