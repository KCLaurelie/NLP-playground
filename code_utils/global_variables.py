import sys
import os

if os.path.exists(r'T:\aurelie_mascio'):
    root_path = r'T:\aurelie_mascio'
else:
    root_path = r'C:\Users\K1774755\AppData\Local\Continuum\anaconda3\envs'

code_path = os.path.join(root_path, 'python_github')
spacy_lib = os.path.join(root_path, r'spacy\Lib\site-packages')
spacy_en_path = os.path.join(spacy_lib, r'en_core_web_sm\en_core_web_sm-2.1.0')

if os.path.exists(code_path):
    os.chdir(code_path)

sys.path.append(spacy_lib)

