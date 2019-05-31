# Import the NLTK library
import nltk
#import pandas as pd
import os

data_folder=r'C:\Users\K1774755\Desktop\New folder\SharePoint\Cognitive Impairment in Schiz - Doc\Courses\CUSMUMH\week 7 - NLP_courses_and_tutorials with nltk & spacy\data'
input_text=os.path.join(data_folder,'brexit.txt')
# Open our text file and read the text into a string
file=open(input_text,encoding="utf8")
text = file.read()
print(text)
# First step is to split the text into sentences

# Use the NLTK default
sentences = nltk.sent_tokenize(text)

# We could also choose from the many available sentence tokenizers
tokenizer = nltk.tokenize.PunktSentenceTokenizer()
sentences_ws = tokenizer.tokenize(text)

print(sentences_ws)

# Hmmmm...thqt first sentence looks a bit funny...those newlines ('\n' should be sentence boundaries)
# Let's do some cleaning up - split sentences multiple newlines (\n\n)
# We can use a regular expression to ddo this - first import Python's regular expression library
import re

clean_sentences = [] # declare an empty list to store our clean sentences

for sentence_ws in sentences_ws:
    # Concatenate all elements of re.split (which returns a list) to our clean sentence list
    clean_sentences += re.split('\n\n+', sentence_ws)

print(clean_sentences)

# Ahhh, much better! Next we want to split the text into tokens

tokenized_sentences = []

# Iterate over all clean sentences in our list
for sentence in clean_sentences:
    # Use NLTK's default word tokenizer
    tokenized_sentence = nltk.word_tokenize(sentence)
    # Append the results to our list of tokenized sentences
    tokenized_sentences.append(tokenized_sentence)

# Alternatively we could use a list comprehension (one of the many great things about Python!)
tokenized_sentences_lc = [nltk.word_tokenize(sentence) for sentence in clean_sentences]

# Just to be sure these lists are equivalent! We can check a condition with the assert keyword
# It with throw an exception if the condition is not met (returns False), or do nothing otherwise (returns True)
assert tokenized_sentences == tokenized_sentences_lc

print(tokenized_sentences)

# Next we want to add part-of-speech tags
# Download required resources - the next step will raise an exception if this is not installed.
# Just uncomment the following line to install the tagger
# nltk.download('averaged_perceptron_tagger')
# Again, we can do this with a list comprehension and NLTK's default tagger method
pos_tagged_sentences = [nltk.pos_tag(token_sequence) for token_sequence in tokenized_sentences]

print(pos_tagged_sentences)

# Next we want to identify named entities in the text
# Again, if the following step raises an exception,
# just download the missing resource by uncommenting the following lines and executing this cell
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# Detect named entities
chunked_sentences = [nltk.ne_chunk(sentence) for sentence in pos_tagged_sentences]

for chunked_sentence in chunked_sentences:
    print(chunked_sentence)

# Now we want to collect all the named entity mentions and their types
ne_names = {}  # Declare an empty dictionary - this is basically Python's version of a hash map


# We can declare a function to search the chunk trees to collect all the named entity names
def extract_entity_names(t):
    entity_names = []

    # "label" is the name used for attributes of Tree objects in NLTK (http://www.nltk.org/_modules/nltk/tree.html)
    if hasattr(t, 'label') and t.label():
        if t.label() in ['GE', 'LOCATION', 'ORGANIZATION', 'PERSON']:
            # Dictionary get(attr_name, default_value) allows us
            # to retrieve a default value if the key is not found in the dictionary
            tmp = ne_names.get(t.label(), [])
            # join is part of the Python library that creates a string out of list
            # elements using the specified separator
            tmp.append(' '.join([child[0] for child in t]))
            ne_names[t.label()] = tmp
        else:
            for child in t:
                # extend is another list method that adds individual elements of a list to the
                # list
                entity_names.extend(extract_entity_names(child))

    return entity_names


for chunked_sentence in chunked_sentences:
    extract_entity_names(chunked_sentence)

# Pretty print is a Python module we can use to make the output nice to read
from pprint import pprint

pprint(ne_names)  # Try comparing with print(ne_names) to see the difference