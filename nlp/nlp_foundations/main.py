

# writing a function for NLP tokenization
def tokenize(text):
    return text.split()






# Tokenization is the process of breaking down text into smaller units such as words, phrases, or symbols called tokens.


#writing a function for stemming 

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
        else:
            stemmed_text.append(word)
            
    return stemmed_text


input_text = input("Enter the Text to get Stemming: ")


final_text = stemming(input_text)
print(final_text)



