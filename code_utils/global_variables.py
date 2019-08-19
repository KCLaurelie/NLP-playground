import sys
import os
import csv

csv.field_size_limit(200000000)

if os.path.exists(r'T:\aurelie_mascio'):  # working on SLaM machine
    root_path = r'T:\aurelie_mascio'
    os.environ['R_HOME'] = r'T:\aurelie_mascio\software\R-3.6.0'
else:  # working on normal machine
    root_path = r'C:\Users\K1774755\AppData\Local\Continuum\anaconda3\envs'
    os.environ['R_HOME'] = r'C:\Program Files\R\R-3.6.0'

# for SLaM
code_path_slam = os.path.join(root_path, 'python_github')
if os.path.exists(code_path_slam):
    os.chdir(code_path_slam)
    sys.path.append(os.path.join(code_path_slam, 'python_pkg'))
    import nltk.data
    nltk.data.path.append(r'T:\aurelie_mascio\software')

# Spacy environment
spacy_lib = os.path.join(root_path, r'spacy\Lib\site-packages')
spacy_en_path = os.path.join(spacy_lib, r'en_core_web_sm\en_core_web_sm-2.1.0')
sys.path.append(spacy_lib)

