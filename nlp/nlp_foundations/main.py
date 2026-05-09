
#this is jsut for my own practice and definitions as well as fucntions

# writing a function for NLP tokenization
def tokenize(text):
    return text.split()



#stemming is the process of reducing a word to its root form. 

stop_words = ['the', 'is', 'in', 'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'because', 'as', 'that', 'which', 'this', 'it', 'its', 'to', 'for', 'of', 'on', 'at', 'by', 'with', 'about', 'from', 'into', 'through', 'during', 'before', 'after', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']


def stemming(text):
    # Tokenize the text
    tokenized_text = tokenize(text)
    
    # Remove stop words (using list comprehension to avoid skipping elements)
    filtered_text = [word for word in tokenized_text if word.lower() not in stop_words]
    
    # Apply basic stemming rules (suffix stripping)
    stemmed_text = []
    for word in filtered_text:
        # A simple rule-based stemmer
        if word.endswith('ing') and len(word) > 4:
            stemmed_text.append(word[:-3])
        elif word.endswith('ed') and len(word) > 3:
            stemmed_text.append(word[:-2])
        elif word.endswith('es') and len(word) > 3:
            stemmed_text.append(word[:-2])
        elif word.endswith('s') and len(word) > 2 and not word.endswith('ss'):
            stemmed_text.append(word[:-1])
        elif word.endswith('ational') and len(word) > 7:
            stemmed_text.append(word[:-5])
        elif word.endswith('tional') and len(word) > 4:
            stemmed_text.append(word[:-4])
        elif word.endswith('er') and len(word) > 2:
            stemmed_text.append(word[:-2])
        elif word.endswith('ness') and len(word) > 4:
            stemmed_text.append(word[:-4])
        elif word.endswith('ly') and len(word) > 2:
            stemmed_text.append(word[:-2])
        
        else:
            stemmed_text.append(word)
            
    return stemmed_text



#lemmatization is the process of reducing a word to its root form BUT it also considers the context of the word and the part of speech of the word and removes the affixes.

def lemmatize(text):
    # Tokenize the text
    tokenized_text = tokenize(text)
    
    # Remove stop words
    filtered_text = [word for word in tokenized_text if word.lower() not in stop_words]
    
    # Dictionary for basic irregular verbs/nouns
    lemma_dict = {
        'am': 'be', 'is': 'be', 'are': 'be', 'was': 'be', 'were': 'be',
        'has': 'have', 'had': 'have', 'having': 'have',
        'does': 'do', 'did': 'do', 'doing': 'do',
        'mice': 'mouse', 'geese': 'goose',
        'better': 'good', 'best': 'good',
        'worse': 'bad', 'worst': 'bad',
        'children': 'child', 'men': 'man', 'women': 'woman',
        'feet': 'foot', 'teeth': 'tooth'
    }
    
    lemmatized_text = []
    for word in filtered_text:
        lower_word = word.lower()
        # Dictionary lookup for irregular forms
        if lower_word in lemma_dict:
            lemma = lemma_dict[lower_word] #checks if the word is in the dictionary
            lemmatized_text.append(lemma.capitalize() if word.istitle() else lemma)
        # Basic rule-based lemmatization for regular forms
        elif lower_word.endswith('ies'):
            lemmatized_text.append(word[:-3] + 'y')
        elif lower_word.endswith('es') and (lower_word.endswith('ses') or lower_word.endswith('xes') or lower_word.endswith('ches') or lower_word.endswith('shes')):
            lemmatized_text.append(word[:-2])
        elif lower_word.endswith('s') and len(word) > 2 and not lower_word.endswith('ss'):
            lemmatized_text.append(word[:-1])
        else:
            lemmatized_text.append(word)
            
    return lemmatized_text





# Bag of Words (BoW) is a way to represent text as numbers.
# It creates a vocabulary of all unique words and counts how many times each word appears.

def create_bow(text):
    # First, we clean the text using our lemmatize function
    cleaned_words = lemmatize(text)
    
    # Create an empty dictionary to hold our word counts
    bow = {}
    
    # Count the frequency of each word
    for word in cleaned_words:
        # Convert to lowercase to ensure consistency
        word = word.lower()
        if word in bow:
            bow[word] += 1
        else:
            bow[word] = 1
            
    return bow


input_text = input("Enter the Text to process (Stemming, Lemmatization, and BoW): ")

final_text_stemmed = stemming(input_text)
print("\n--- Stemmed ---")
print(final_text_stemmed)

final_text_lemmatized = lemmatize(input_text)
print("\n--- Lemmatized ---")
print(final_text_lemmatized)

bow_result = create_bow(input_text)
print("\n--- Bag of Words (BoW) ---")
print(bow_result)
