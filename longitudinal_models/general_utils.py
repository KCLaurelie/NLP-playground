# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:46:34 2019

@author: AMascio
"""
import os
import pandas as pd
import numpy as np
import datetime
import sys
import csv
from collections import OrderedDict
import pyximport

pyximport.install()
sys.maxsize
csv.field_size_limit(200000000)

root_path = r'T:\aurelie_mascio'
CRIS_data_path = root_path + '\\CRIS data'
SQL_path = root_path + '\\SQL queries'
try:
    os.chdir(root_path + r'\python_scripts')  # directory with python library
except:
    pass
headers_dict_file = root_path + r'\python_scripts\CRIS_data_dict.csv'
patients_data_file = CRIS_data_path + r'\F20_patients_documents_details_from_DB.csv'


##############################################################################
# GENERAL UTILS FUNCTIONS
############################################################################## 
# standardize dataframe data
def clean_df(df, to_numeric=True, filter_col=None, filter_value=None, threshold_col=None, threshold_value=None):
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'brc_id': 'brcid'})
    if 'brcid' in df.columns: df.dropna(subset=['brcid'], inplace=True)
    date_cols = [x for x in df.columns if 'date' in x]
    df[date_cols] = df[date_cols].apply(pd.to_datetime, errors='ignore')
    if to_numeric:
        non_date_cols = [col for col in df.columns if df[col].dtype != 'datetime64[ns]']
        df[non_date_cols] = df[non_date_cols].apply(pd.to_numeric, errors='ignore')
    if filter_col is not None: df = df[df[filter_col].isin(filter_value)]
    if threshold_col is not None: df = df.loc[df[threshold_col] >= threshold_value]
    df.drop_duplicates(inplace=True)
    return df


def get_duplicate_columns(df):
    '''
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])

    return list(duplicateColumnNames)


def drop_duplicate_columns(df):
    new_df = df.drop(columns=get_duplicate_columns(df))
    return new_df


## COUNT NUMBER WORDS IN TEXT FILE
def count_words(file_name):
    num_words = 0
    with open(file_name, 'r') as f:
        for line in f:
            words = line.split()
            num_words += len(words)
    return num_words


## read csv file 
# file_path=r'T:\aurelie_mascio\multimorbidity\corpus\multimorbidity_attachments_discharge_docs.csv'
def super_read_csv(file_path, usecols=None, clean_results=True, filter_col=None, filter_value=None, threshold_col=None,
                   threshold_value=None,
                   read_from_SQL_export=False):
    # f = open(csv_file, encoding='utf-8-sig')
    # df = pd.read_csv(csv_file, sep='|', header=0, encoding='cp1252', quoting=1, escapechar='\\', engine='python')

    if os.stat(file_path).st_size / 1e9 >= 1.5:  # large csv files need to be read by chunks
        # 1. READ THE FILE IN CHUNKS
        if read_from_SQL_export:  # file exported using SSIS package method
            TextFileReader = pd.read_csv(file_path, sep='|', header=0, encoding='ansi', quoting=1, escapechar='\\',
                                         engine='python', usecols=usecols, chunksize=10000)
        else:
            TextFileReader = pd.read_csv(file_path, usecols=usecols, chunksize=10000, header=0, engine='python',
                                         error_bad_lines=False)
            # TextFileReader=pd.read_csv(file_path,usecols=usecols,chunksize=10000,low_memory=False,header=0,encoding='utf8',engine='c',error_bad_lines=False)
        # test=TextFileReader.get_chunk(3)
        # row_count = sum(1 for row in TextFileReader)

        # 2. RE-ASSEMBLE THE FILE CHUNKS IN A DATAFRAME
        try:  # if we're lucky we can load all that in 1 go
            res = pd.concat(TextFileReader, ignore_index=True)
            if clean_results: res = clean_df(res, filter_col=filter_col, filter_value=filter_value,
                                             threshold_col=threshold_col, threshold_value=threshold_value)
        except:  # if we're less lucky let's concatenate each chunk individually
            res = pd.DataFrame()
            for chunk in TextFileReader:
                if clean_results: chunk = clean_df(chunk, filter_col=filter_col, filter_value=filter_value,
                                                   threshold_col=threshold_col, threshold_value=threshold_value)
                res = pd.concat([res, chunk], ignore_index=True)

    else:  # normal sized csv files
        if read_from_SQL_export:  # file exported using SSIS package method
            res = pd.read_csv(file_path, sep='|', header=0, encoding='ansi', quoting=1, escapechar='\\',
                              engine='python', usecols=usecols)
        else:
            try:  # file exported directly from SQL (not using SSIS export method)
                res = pd.read_csv(file_path, header=0, usecols=usecols, engine='python', error_bad_lines=False)
            except:  # file was likely created from excel
                res = pd.read_csv(file_path, header=0, usecols=usecols, engine='python', error_bad_lines=False,
                                  encoding='ISO-8859-1')
        if clean_results: res = clean_df(res, filter_col=filter_col, filter_value=filter_value,
                                         threshold_col=threshold_col, threshold_value=threshold_value)
    return res


## LOAD SYMPTOMS APP RESULTS (headers specific to CRIS NLP apps)
def load_symptoms_df(symptoms_file,
                     obs_filter=['positive'],
                     confidence_threshold=0.5,
                     usecols=['BrcID', 'observation', 'Document_Date', 'confidence']):
    res_df = pd.DataFrame()
    TextFileReader = pd.read_csv(symptoms_file, usecols=usecols, chunksize=10000, header=0, engine='python',
                                 error_bad_lines=False)
    for chunk in TextFileReader:
        if len(obs_filter) > 0: chunk = chunk[chunk.observation.isin(obs_filter)]
        res_df = pd.concat([res_df, chunk.loc[chunk.confidence > confidence_threshold]], ignore_index=True)
    res_df = clean_df(res_df)
    res_df['year'] = res_df.document_date.dt.year
    return res_df


##############################################################################
## RANDOM STUFF
##############################################################################

def cut_with_na(to_bin, bins, labels, na_category='not known'):
    to_bin = pd.to_numeric(to_bin, errors='coerce')
    res = pd.cut(pd.Series(to_bin),
                 bins=bins,
                 labels=labels
                 ).values.add_categories(na_category)
    res = res.fillna(na_category)
    return res


# source: https://www.nhs.uk/common-health-questions/lifestyle/what-is-the-body-mass-index-bmi/    
def bmi_category(score, na_category='not known'):
    res = cut_with_na(to_bin=score,
                      bins=[-np.inf, 18.5, 25, 30, np.inf],
                      labels=['underweight', 'normal', 'preobese', 'obese'],
                      na_category=na_category)
    return res


# source: https://www.cdc.gov/bloodpressure/measure.htm
def blood_pressure(systolic, diastolic, na_category='not known'):
    ratio = pd.to_numeric(systolic, errors='coerce') / pd.to_numeric(diastolic, errors='coerce')
    res = cut_with_na(to_bin=ratio,
                      bins=[-np.inf, 120. / 80, 140. / 90, np.inf],
                      labels=['normal', 'prehypertension', 'hypertension'],
                      na_category=na_category)
    return res


# source: https://www.diabetes.co.uk/fasting-plasma-glucose-test.html
def diabetes(plasma_glucose, na_category='not known'):
    res = cut_with_na(to_bin=plasma_glucose,
                      bins=[-np.inf, 5.5, 7, np.inf],
                      labels=['normal', 'prediabetic', 'diabetic'],
                      na_category=na_category)
    return res


def convert_num_to_bucket(nb, bucket_size=0.5, convert_to_str=True):
    lower_bound = np.ceil(nb / bucket_size) * bucket_size - bucket_size
    upper_bound = np.ceil(nb / bucket_size) * bucket_size
    res = [lower_bound, upper_bound]
    if convert_to_str: res = str(res)
    return res


##############################################################################
## DATE FUNCTIONS FOR LONGITUDINAL MODELLING
############################################################################## 
# dates = ["2014-10-10", "2016-01-07"]

def monthlist_short(dates):
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    return OrderedDict(
        ((start + datetime.timedelta(_)).strftime(r"%b-%y"), None) for _ in range((end - start).days)).keys()


def monthlist_fast(dates):
    start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
    total_months = lambda dt: dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start) - 1, total_months(end)):
        y, m = divmod(tot_m, 12)
        mlist.append(datetime(y, m + 1, 1).strftime("%b-%y"))
    return mlist


##############################################################################
## GENERATE LONGITUDINAL DATA FROM RAW DATA (HOSPITAL STAYS, MEASURES...)
##############################################################################
# test_file=r'T:\aurelie_mascio\CRIS data\F20_ward_stays_data_fromDB.csv'
# patients_data=r'T:\aurelie_mascio\CRIS data\F20_patients_details_from_DB.csv'
# test_df=pd.read_csv(test_file,header=0)
# df=test_df.head().copy()
# df['dob']=datetime.datetime(1986, month=8, day=13)

## FUNCTION TO GENERATE LONGITUDINAL DATA BY AGE
def convert_to_longitudinal_age(df,
                                key_col='brcid',
                                dob_col='dob',
                                start_date_col='actual_start_date',  # for interval only
                                end_date_col='actual_end_date',  # for interval only
                                measure_date_col='rating_date',  # for rating only
                                measure_col='length_stay_days',  # for rating only
                                dob_file=patients_data_file,
                                mode='interval'):  # interval: need to measure length (e.g. length of hospital stay), not interval: need to grab measure (e.g. HoNOS) by age
    df = clean_df(df)
    dob_data = pd.read_csv(dob_file, header=0)
    #######################################
    ### TODO: JOIN WITH DOB DATA
    #######################################
    if mode != 'interval':  # we just want to aggregate 1 measure by age, taking the average
        df['rating_age'] = round((df[measure_date_col] - df[dob_col]).dt.days / 365, 0)
        res = pd.pivot_table(df, values=measure_col, index=[key_col], columns=['rating_age'], aggfunc=np.mean)
        # res.unstack().reset_index()
        return res
    df['start_age'] = (df[start_date_col] - df[dob_col]).dt.days / 365
    df['end_age'] = (df[end_date_col] - df[dob_col]).dt.days / 365
    min_age = np.floor(df.start_age.min())
    max_age = np.ceil(df.end_age.max())
    for age in range(int(min_age), int(max_age)):
        up_bound = np.minimum(age + 1, df.end_age)
        low_bound = np.maximum(age, df.start_age)
        df[str(age)] = np.maximum(0, up_bound - low_bound) * 365

    res_cols = [key_col] + list(map(str, range(int(min_age), int(max_age))))
    res = df[res_cols].groupby([key_col], as_index=False).sum().sort_values(by=[key_col])
    return res


## FUNCTION TO GENERATE LONGITUDINAL DATA BY YEAR
def convert_to_longitudinal_dates(df,
                                  key_col='brcid',
                                  start_date_col='actual_start_date',
                                  end_date_col='actual_end_date',
                                  measure_date_col='rating_date',
                                  measure_col='length_stay_days',
                                  mode='interval'):
    df = clean_df(df)
    if mode != 'interval':  # we just want to aggregate 1 measure by age, taking the average
        df['rating_year'] = df[measure_date_col].dt.year
        res = pd.pivot_table(df, values=measure_col, index=[key_col], columns=['rating_year'], aggfunc=np.mean)
        return res

    min_date = df[start_date_col].dt.year.min()
    max_date = df[end_date_col].dt.year.max()
    for year in range(int(min_date), int(max_date + 1)):
        interval_start_date = datetime.datetime(year, month=1, day=1)
        interval_end_date = datetime.datetime(year + 1, month=1, day=1)
        up_bound = np.minimum(pd.Series(len(df) * [interval_end_date]), df.actual_end_date)
        low_bound = np.maximum(pd.Series(len(df) * [interval_start_date]), df.actual_start_date)
        df[str(year)] = np.maximum(0, (up_bound - low_bound).dt.days)

    res_cols = [key_col] + list(
        map(str, range(int(min_date), int(max_date + 1))))  # [x for x in df.columns if 'stay' in x or key_col in x]
    res = df[res_cols].groupby([key_col], as_index=False).sum().sort_values(by=[key_col])
    return res
