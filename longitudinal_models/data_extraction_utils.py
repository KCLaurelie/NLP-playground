# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 08:26:08 2019

@author: AMascio
"""
import os
import pandas as pd
import longitudinal_models.general_utils as gutils
import nltk.data
from nltk import tokenize

root_path = r'T:\aurelie_mascio'
CRIS_data_path = root_path + '\\CRIS data'
os.chdir(root_path + r'\python_scripts')  # directory with python library
headers_dict_file = root_path + r'\python_scripts\CRIS_data_dict.csv'
nltk.data.path.append(root_path + '\\software')


# GENERATE RANDOM SAMPLE OF FILES
def generate_sample(patients_file=r'T:\aurelie_mascio\F20 corpus\patients_static_data.csv',
                    sample_size=1000,
                    random_state=1,
                    sample_cols=['brcid', 'len_text', 'date_of_birth', 'first_language', 'gender', 'employment',
                                 'occupation', 'religion', 'marital_status', 'ethnicity', 'secondary_diag']):
    patients_data = gutils.super_read_csv(patients_file)  # in case file saved in stupid format
    patients_data.fillna('not known', inplace=True)
    summary = patients_data.describe(include='all')
    print(summary)
    sample = patients_data[sample_cols].sample(n=sample_size, random_state=random_state)
    return sample


##############################################################################
## EXTRACT TEXTS FROM CSV FILES
############################################################################## 

test_path = root_path + '\\text to annotate\\eHOST\\10383159'
text_file = test_path + '\\MHA Tribunal Report 02.04.15.txt'
file_path = root_path + r'\CRIS data\F20_all_texts_fromUI.csv'


def extract_texts_from_csv(file_path=root_path + r'\multimorbidity\data_multimorbidity.csv',
                           output_path=root_path + r'\multimorbidity\corpus',
                           usecols=['BrcId', 'CN_Doc_ID', 'doc_date', 'Attachment_Text'],
                           # ['BRC_ID','cATCN10','cATDa0', 'cATAt03'],
                           read_from_SQL_export=False):
    if read_from_SQL_export:  # file exported using SSIS package method
        TextFileReader = pd.read_csv(file_path, sep='|', header=0, encoding='ansi', quoting=1, escapechar='\\',
                                     engine='python', usecols=usecols, chunksize=10000)
    else:
        TextFileReader = pd.read_csv(file_path, usecols=usecols, chunksize=100000, low_memory=False, header=0,
                                     encoding='utf8', engine='c', error_bad_lines=False)
    # TextFileReader= gutils.super_read_csv(csv_file,usecols=usecols,clean_results=False)
    # test=TextFileReader.get_chunk(3)
    # row_count = sum(1 for row in TextFileReader)
    # text_data = pd.concat(TextFileReader, ignore_index=True)
    for chunk in TextFileReader:
        for record in chunk.itertuples():
            tmp_text = record[usecols[0]]
            title = str(record[usecols[0]]) + '_' + str(record[usecols[2]]) + '_' + str(record[usecols[1]])
            with open(os.path.join(output_path, title + '.txt'), "w") as text_file:
                text_file.write(tmp_text)
    return 0


def extract_sentences_from_corpus(corpus_path=root_path + r'\CRIS texts annotations\F20 all texts',
                                  keywords=['attention', 'concentration']):
    res = pd.DataFrame(columns=['document', 'sentence'])
    # r=root, d=directories, f = files
    for r, d, f in os.walk(corpus_path):
        for file in f:
            if '.txt' in file:
                sentences = extract_sentences_from_textfile(file, keywords)
                res_tmp = pd.DataFrame(sentences, columns=['sentence'])
                res_tmp['document'] = os.path.basename(file)
                res = res.append(res_tmp, ignore_index=True, sort=False)
    res = res.drop_duplicates(subset=['sentence'])
    return res


def extract_sentences_from_CRIS_frontend_output(CRIS_file,
                                                usecols=['BRC_ID', 'cATDa0', 'cATSu02', 'cATAt03'],
                                                res_cols=['brc_id', 'doc_date', 'doc_type', 'sentence'],
                                                keywords=['attention', 'concentration']):
    # CRIS_file=file_path+"Attention\\results_F20_honos_attention_long.csv"
    res = pd.DataFrame(columns=res_cols)
    TextFileReader = pd.read_csv(CRIS_file, usecols=usecols, chunksize=100000, low_memory=False, header=0,
                                 encoding='utf8', engine='c', error_bad_lines=False)
    clean_data = pd.concat(TextFileReader, ignore_index=True)
    clean_data = clean_data.dropna(subset=['cATAt03']).drop_duplicates()
    # clean_data=clean_CRIS_frontend_headers(clean_data,headers_dict_file)
    _iter = 0
    for record in clean_data.itertuples():
        _iter += 1
        sentences = text_to_sentences(record.cATAt03, keywords=keywords)
        print(_iter, len(sentences))
        if len(sentences) > 0:
            res_tmp = pd.DataFrame(sentences, columns=['sentence'])
            res_tmp['brc_id'] = record.BRC_ID
            res_tmp['doc_date'] = record.cATDa0
            res_tmp['doc_type'] = record.cATSu02
            res = res.append(res_tmp, ignore_index=True, sort=False)
    res = res.drop_duplicates(subset=['sentence'])
    return res.sort_values(by=['brc_id', 'doc_date'])


def extract_sentences_from_textfile(text_file, keywords=['example', 'concentration']):
    with open(text_file, 'r') as myfile: data = myfile.read()
    res = text_to_sentences(data, keywords=keywords)
    return res


def text_to_sentences(text, keywords=['example', 'concentration']):
    res = []
    sentences = tokenize.sent_tokenize(text)
    for sentence in sentences:
        if any(map(lambda word: word in sentence, keywords)):
            res.append(sentence)
    return res


##############################################################################
## FUNCTIONS TO LOAD BIG CSV FILES
############################################################################## 
def clean_CRIS_frontend_headers(df, headers_dict_file=headers_dict_file):
    headers_dict = pd.read_csv(headers_dict_file)
    headers_dict = dict(zip(headers_dict['code'], headers_dict['text']))
    df.columns = [headers_dict.get(item, item) for item in list(df.columns)]
    df.columns = df.columns.str.lower()
    return df


def save_chunks_to_csv(TextFileReader, output_file="", remove_nan_cols=['observation'], drop_duplicates=True):
    res_df = pd.DataFrame()
    _iter = 0
    for chunk in TextFileReader:
        try:
            if len(remove_nan_cols) > 0: chunk = chunk.dropna(subset=remove_nan_cols)
            if drop_duplicates: chunk = chunk.drop_duplicates()
            res_df = pd.concat([res_df, chunk], ignore_index=True)
            print("processing chunk nb", _iter)
        except:
            print("issue with chunk nb", _iter)
        _iter += 1
        # res_df=res_df.sort_values(by=['BrcID', 'Document_Date'])
    if 'csv' in output_file: res_df.to_csv(output_file)
    return res_df


# res=save_chunks_to_csv(TextFileReader, col_to_clean=['BrcID'])

def save_chunks_to_csv2(TextFileReader, output_file,
                        cols_to_keep=[],
                        drop_duplicates=True):
    for chunk in TextFileReader:
        if len(cols_to_keep) > 0: chunk = chunk[cols_to_keep]
        if drop_duplicates: chunk = chunk.drop_duplicates()
        # print(newChunk.columns)
        # print("Chunk -> File process")
        with open(output_file, 'a') as f:
            chunk.to_csv(f, header=False, sep='\t', index=False)
            print("Chunk appended to the file")
    return 0
