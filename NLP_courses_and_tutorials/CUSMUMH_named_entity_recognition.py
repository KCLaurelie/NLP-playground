import spacy
from spacy import displacy
from spacy.gold import GoldParse
from spacy.scorer import Scorer

import json
import io

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import sklearn
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import scipy

import random
import os

from collections import OrderedDict
data_folder = r"""C:\Users\K1774755\King's College London\Cognitive Impairment in Schizophrenia - Documents\Courses\CUSMUMH\week 8 - unsupervized NLP_courses_and_tutorials"""

# 1: corpus
# We have prepared training and test data in a json format.
with io.open(os.path.join(data_folder, 'chunking_trainingdata_CAT.json'), encoding='utf8') as f:
    train_data = json.load(f, object_pairs_hook=OrderedDict)

# Let's take a look at a random document and its annotations. The json format contains the text itself, and then the start and stop offsets for each entity. What are the instances we want to learn?
train_data[14][1]
# Let's save a document to look at after training.
example_text, _ = train_data[44]

# 2: Training a named entity model with spaCy
# We can use spaCy to train our own named entity recognition model using their training algorithm. First we need to load a spaCy English language model, so that we can sentence- and word tokenize.
nlp = spacy.load(r'C:\Users\K1774755\AppData\Local\Continuum\anaconda3\lib\site-packages\en_core_web_sm\en_core_web_sm-2.0.0')

#What nlp preprocessing parts does this model contain?
nlp.pipe_names

#We have our own named entities that we want to develop a model for
#Let's add these entity labels to the spaCy ner pipe.
if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
# otherwise, get it so we can add labels
else:
    ner = nlp.get_pipe("ner")
labels = set()
for _, annotations in train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])
        labels.add(ent[2])

#We don't want to retrain the other pipeline steps, so let's keep those.

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
other_pipes = ["tagger", "parser"]
other_pipes

#train the model
#Let's train our clinical concept ner model. Let's set the number of training iterations.
n_iter=(10)
with nlp.disable_pipes(*other_pipes):  # only train NER
#with nlp.disable_pipes(other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = spacy.util.minibatch(train_data, size=spacy.util.compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

#We have now added a clinical concept entity recognizer in the spaCy nlp model! Let's look at an example document and the predicted entities from the new model.

example_text
doc2 = nlp(example_text)
colors = {'ANATOMY': 'lightyellow',
          'DISEASESYNDROME': 'pink',
          'SIGNSYMPTOM': 'lightgreen'}
displacy.render(doc2, style='ent', jupyter=True, options={'colors':colors})

# 3: Evaluation
# How do we know how good this model is? Let's compare with the 'gold standard' test data.

scorer = Scorer()
with io.open('chunking_testdata_CAT.json', encoding='utf8') as f:
    test_data = json.load(f, object_pairs_hook=OrderedDict)

for text, entity_offsets in test_data:
    doc = nlp.make_doc(text)
    gold = GoldParse(doc, entities=entity_offsets.get('entities'))
    doc = nlp(text)
    scorer.score(doc, gold)
print('Precision: ',scorer.scores['ents_p'])
print('Recall: ',scorer.scores['ents_r'])
print('F1: ',scorer.scores['ents_f'])
#print(scorer.scores)