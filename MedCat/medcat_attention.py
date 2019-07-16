from medcat.cat import CAT
from medcat.utils.vocab import Vocab
from medcat.prepare_cdb import PrepareCDB
from medcat.cdb import CDB
import os

root_path = r'C:\Users\K1774755\PycharmProjects\toy-models\MedCat'
vocab = Vocab()

# Load the vocab model you just downloaded
vocab.load_dict(os.path.join(root_path, 'med_ann_norm_dict.dat'))

# If you have an existing CDB
cdb = CDB()
# cdb.load_dict(os.path.join(root_path, 'simple_cdb.csv'))


# If you need a special CDB you can build one from a .csv file
preparator = PrepareCDB(vocab=vocab)
csv_paths = [os.path.join(root_path, 'simple_cdb.csv')]#, '<another one>', ...]
csv_paths = [os.path.join(root_path, 'attention_cdb.csv')]
cdb = preparator.prepare_csvs(csv_paths)

# Save the new CDB for later
cdb.save_dict(os.path.join(root_path, 'simple_cdb.cdb'))

# To annotate documents we do
doc = "My simple document with kidney failure"
cat = CAT(cdb=cdb, vocab=vocab)
cat.train = False
doc_spacy = cat(doc)
# Entities are in
doc_spacy._.ents
# Or to get a json
doc_json = cat.get_json(doc)

# To have a look at the results:
from spacy import displacy
# Note that this will not show all entites, but only the longest ones
displacy.serve(doc_spacy, style='ent')

# To run cat on a large number of documents
data = [] # [(<doc_id>, <text>), (<doc_id>, <text>), ...]
docs = cat.multi_processing(data)

# Training and Fine-tuning
# To fine-tune or train everything from the ground up (excluding word-vectors), you can use the following:

# Loadinga CDB or creating a new one is as above.

# To run the training do
f = open("<some file with a lot of medical text>", 'r')
# If you want fine tune set it to True, old training will be preserved
cat.run_training(f, fine_tune=False)