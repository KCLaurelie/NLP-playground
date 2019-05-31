# As usual import the library we want to use (after installation)
import spacy

# Load the English model
nlp = spacy.load(r'C:\Users\K1774755\AppData\Local\Continuum\anaconda3\lib\site-packages\en_core_web_sm\en_core_web_sm-2.0.0')

# text = open('data/brexit.txt').read() # We could load a text from file, but let's use the text string declared below as input this time
text = "The patient denies having very suicidal thoughts. This is not a test. It doesn't have a meaning. Brexit was never a good idea. There is no good deal for the UK."

# Run our text through the default processing pipeline
doc = nlp(text)
# See what it looks like (using Python's string formatter to make it readable)!
heading = '{:10}{:10}{:10}{:10}{:10}{:10}{:10}'.format('INDEX','WORD','LEMMA','TAG1','TAG2','HEAD','REL')
print(heading)

hlines = '{:10}{:10}{:10}{:10}{:10}{:10}{:10}'.format('-----','----','-----','----','----','----','---')
print(hlines)

for token in doc:
    line = '{:10}{:10}{:10}{:10}{:10}{:10}{:10}'.format(str(token.i), token.text, token.lemma_, token.tag_, token.pos_, str(token.head.i), token.dep_)
    print(line)

# It's also possible to view the syntactic dependency structure graphically
from spacy import displacy

displacy.render(doc, style='dep', jupyter=True, options={'distance': 100})

# We get a bunch of warnings, but who cares!

# A basic negation and suicidality detection exercise...
# Add a new 'negated' attribute to negated verbs and nouns
# and a 'suicidality' attribute to mentions that match a pattern,
# also check if these mentions are within some kind of negative scope (here the verb 'deny')...

for token in doc:

    # Verbal heads
    if token.dep_ == 'neg':
        # Add a custom annotation to the head token
        doc[token.head.i].doc.user_data[(token.head.i, 'negated')] = 'TRUE'

    # Nominal heads
    if token.dep_ == 'det' and token.lemma_ == 'no':
        # Add a custom annotation to the head token
        doc[token.head.i].doc.user_data[(token.head.i, 'negated')] = 'TRUE'

    # We can match the tokens that govern any word containing a particular substring of interest (e.g. the patient
    # denies having suicidal thoughts)
    import re

    if re.search('^suicid', token.lemma_, flags=re.I) is not None:

        # We can define a function "locally" to walk up the dependency tree and add an annotation
        def walk_up_and_annotate(cur_token):
            # Add an annotation
            doc[cur_token.i].doc.user_data[(cur_token.i, 'suicidality')] = 'TRUE'
            # cur_token.dep_ is the dependency label, cur_token.pos_ is the part-of-speech
            if cur_token.dep_ in ['ccomp', 'xcomp'] or cur_token.pos_ == 'VERB':
                return cur_token
            return walk_up_and_annotate(cur_token.head)

            # ...and another one to walk back down again


        def walk_down_and_annotate(cur_token):
            doc[cur_token.i].doc.user_data[(cur_token.i, 'negated')] = 'TRUE'
            # In a "standard" dependency tree, each node has exactly one head/governing node,
            # but a head may govern one or more child nodes
            for child in cur_token.children:
                walk_down_and_annotate(child)


        head = walk_up_and_annotate(token)

        # Annotate the negation if need be
        if head.head.lemma_ == 'deny':
            walk_down_and_annotate(head)
# Now let's see if it worked...
# We add the new columns in the output to show the values of our newly annotated attributes

heading = '{:10}{:10}{:10}{:10}{:10}{:10}{:10}{:10}{:10}'.format('INDEX', 'WORD', 'LEMMA', 'TAG1', 'TAG2', 'HEAD',
                                                                 'REL', 'NEGATED', 'SUICIDALITY')
print(heading)

hlines = '{:10}{:10}{:10}{:10}{:10}{:10}{:10}{:10}{:10}'.format('-----', '----', '-----', '----', '----', '----', '---',
                                                                '-------', '-----------')
print(hlines)

for token in doc:
    # Retrieve our custom annotations if there are any, else return a default value '_'
    negated = token.doc.user_data.get((token.i, 'negated'), '_')
    suicidality = token.doc.user_data.get((token.i, 'suicidality'), '_')

    line = '{:10}{:10}{:10}{:10}{:10}{:10}{:10}{:10}{:10}'.format(str(token.i), token.text, token.lemma_, token.tag_,
                                                                  token.pos_, str(token.head.i), token.dep_, negated,
                                                                  suicidality)
    print(line)

