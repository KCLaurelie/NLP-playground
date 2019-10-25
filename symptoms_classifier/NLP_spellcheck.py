import os
import re
from collections import Counter
from spellchecker import SpellChecker
from symptoms_classifier.NLP_text_cleaning import clean_string
from symptoms_classifier.NLP_embedding import tokenize_text_series
root_folder = os.getcwd() + '\\symptoms_classifier'
words_dic = os.path.join(root_folder, 'big.txt')  # vocabulary to use for spelling mistakes etc


# this is slow and shit
def autocorrect_sentences(sentences):
    # sentences = pd.read_excel(r'C:\Users\K1774755\Downloads\phd\attention_sentences.xlsx')['clean_text']
    spell = SpellChecker()
    tokens = tokenize_text_series(sentences, tokenization_type='clean', remove_contractions=True)
    tokens = [item for sublist in tokens for item in sublist]
    mispelled = spell.unknown(list(set(tokens)))
    print(len(mispelled), 'words mispelled in the text')

    for idx, word in enumerate(mispelled):
        print(idx, word)
        # Get the one `most likely` answer
        correct_word = spell.correction(word)
        print('replacing', word, 'with', correct_word)
        sentences = sentences.replace(word, correct_word)

    return sentences


def autocorrect(text, clean_text=True):
    if clean_text:
        text = clean_string(text, remove_punctuation=False)

    # find those words that may be misspelled
    words = re.sub(r'[^\w\s]', ' ', text).split()

    spell = SpellChecker()
    misspelled = spell.unknown(words)

    for word in misspelled:
        # Get the one `most likely` answer
        correct_word = spell.correction(word)
        text = text.replace(word, correct_word)

    return text


def all_words(text): return re.findall(r'\w+', text.lower())


# WORDS = Counter(all_words(open(words_dic).read()))
#
#
# def prob(word, n=sum(WORDS.values())):
#     "Probability of `word`."
#     return WORDS[word] / n
#
#
# def correction(word):
#     "Most probable spelling correction for word."
#     return max(candidates(word), key=prob)
#
#
# def candidates(word):
#     "Generate possible spelling corrections for word."
#     return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]
#
#
# def known(words):
#     "The subset of `words` that appear in the dictionary of WORDS."
#     return set(w for w in words if w in WORDS)
#
#
# def edits1(word):
#     "All edits that are one edit away from `word`."
#     letters = 'abcdefghijklmnopqrstuvwxyz'
#     splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
#     deletes = [L + R[1:] for L, R in splits if R]
#     transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
#     replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
#     inserts = [L + c + R for L, R in splits for c in letters]
#     return set(deletes + transposes + replaces + inserts)
#
#
# def edits2(word):
#     "All edits that are two edits away from `word`."
#     return (e2 for e1 in edits1(word) for e2 in edits1(e1))