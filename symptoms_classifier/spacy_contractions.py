import sys
import os
spacy_lib = r'C:\Users\K1774755\AppData\Local\Continuum\anaconda3\envs\spacy\Lib\site-packages'
sys.path.append(spacy_lib)
spacy_en_path = os.path.join(spacy_lib, r'en_core_web_sm\en_core_web_sm-2.1.0')

import spacy
from spacy.attrs import ORTH, LEMMA, NORM, TAG


nlp = spacy.load(spacy_en_path)

TOKENIZER_EXCEPTIONS = {
# do
    "don't": [
        {ORTH: "do", LEMMA: "do"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "doesn't": [
        {ORTH: "does", LEMMA: "do"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "didn't": [
        {ORTH: "did", LEMMA: "do"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
# can
    "can't": [
        {ORTH: "ca", LEMMA: "can"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "couldn't": [
        {ORTH: "could", LEMMA: "can"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
# have
    "I've'": [
        {ORTH: "I", LEMMA: "I"},
        {ORTH: "'ve'", LEMMA: "have", NORM: "have", TAG: "VERB"}],
    "haven't": [
        {ORTH: "have", LEMMA: "have"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "hasn't": [
        {ORTH: "has", LEMMA: "have"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "hadn't": [
        {ORTH: "had", LEMMA: "have"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
# will/shall will be replaced by will
    "I'll'": [
        {ORTH: "I", LEMMA: "I"},
        {ORTH: "'ll'", LEMMA: "will", NORM: "will", TAG: "VERB"}],
    "he'll'": [
        {ORTH: "he", LEMMA: "he"},
        {ORTH: "'ll'", LEMMA: "will", NORM: "will", TAG: "VERB"}],
    "she'll'": [
        {ORTH: "she", LEMMA: "she"},
        {ORTH: "'ll'", LEMMA: "will", NORM: "will", TAG: "VERB"}],
    "it'll'": [
        {ORTH: "it", LEMMA: "it"},
        {ORTH: "'ll'", LEMMA: "will", NORM: "will", TAG: "VERB"}],
    "won't": [
        {ORTH: "wo", LEMMA: "will"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
    "wouldn't": [
        {ORTH: "would", LEMMA: "will"},
        {ORTH: "n't", LEMMA: "not", NORM: "not", TAG: "RB"}],
# be
    "I'm'": [
        {ORTH: "I", LEMMA: "I"},
        {ORTH: "'m'", LEMMA: "be", NORM: "am", TAG: "VERB"}]
}