from code_utils.global_variables import *
import os
import pandas as pd
import numpy as np
import datetime
from collections import OrderedDict
from itertools import combinations


##############################################################################
# GENERAL UTILS FUNCTIONS
##############################################################################
def list_combos(lst, r=None):
    """
    Return successive r-length combinations of elements in the iterable.
    :param lst: iterable (can be list or tuple of elements)
    :param r: (default None) length of combinations desired. by default will generate combinations for all lengths
    :return: list of combinations
    """
    lst = list(lst)
    if r is not None: # list combinations for specific r
        res = list(combinations(lst, r))
    else: # list all combinations
        res = []
        for r in range(1, len(lst) + 1):
            res += list(combinations(lst, r))
    return res


def list_to_excel(lst_to_print, filepath='out.xlsx', sheet_name='Sheet1', startrow=0, startcol=0):
    """
    prints a list of elements in an excel workbook
    :param lst_to_print: list of elements to be printed
    :param filepath: path of excel workbook
    :param sheet_name: sheet where to print the data
    :param startrow: row at which to start printing the data
    :param startcol: column at which to start printing the data
    :return: last row at which data has been printed
    """
    mode = 'a' if os.path.isfile(filepath) else 'w'
    print('adding sheet', sheet_name, 'using mode:', ('append' if mode == 'a' else 'new workbook'))
    writer = pd.ExcelWriter(filepath, engine='openpyxl', mode=mode)
    cpt_row = startrow
    lst_to_print = to_list(lst_to_print)
    for i in lst_to_print:
        to_print = i if isinstance(i, pd.core.frame.DataFrame) else pd.DataFrame([i])
        print('printing to excel:\n', to_print)
        to_print.to_excel(writer, sheet_name=sheet_name, startrow=cpt_row, startcol=startcol)
        cpt_row += len(to_print) + 2
    writer.save()
    return cpt_row


def get_wa(sentence, keywords, context=10, fixed_weights=False, debug=False):
    """
    generate vector of weighted averages given specific keywords in a tokenized sentence.
    the words closest to the keyword will get maximum weight etc...
    :param sentence: tokenized sentence
    :param keywords: keywords to look for in the sentence (can be either a list or string)
    :param context: number of words before/after keyword to use
    :param fixed_weights: set to true to use weight=1 for all words in context
    :param debug: printouts for debugging
    :return: array of weights
    """
    if debug:
        print('assigning word weights using:\n', ('fixed weights' if fixed_weights else 'decreasing weights'),
              '\nkeywords:', keywords, '\ncontext:', context)
    sentence = to_list(sentence)
    keywords = to_list(keywords)
    # kw_idx = [sentence.index(x) for x in keywords if x in sentence]
    kw_idx = [sentence.index(s) for s in sentence if any(xs in s for xs in keywords)]
    weights = [0]*len(sentence)
    context_weights = [0]+[1]*context if fixed_weights else list(np.arange(0, 1 + 1 / context, 1 / context))
    for i in kw_idx:
        left = context_weights[-i:] if i > 0 else []
        right = context_weights[::-1][:len(sentence) - i - 1]
        lst = [0] * (i - context - 1) + left + [1] + right + [0] * (len(sentence) - i - context - 2)
        weights = np.maximum(weights, lst)
    if debug: print(sentence, weights)
    return weights


def round_nearest(x, intv=0.5, direction='down'):
    """
    rounds number or series of numbers to nearest value given interval
    :param x: number or pd>Series of numbers to round
    :param intv: interval to round to
    :param direction: up (ceil) or down (floor)
    :return: rounded number or series of numbers
    """
    if direction == 'down':
        res = np.floor(x / intv) * intv
    else:
        res = np.ceil(x / intv) * intv
    return res


def date2year(date):
    """
    converts date to year+portion of year (e.g. 1900.5 for 30/6/1900)
    :param date: date or series of dates
    :return: date of series of dates converted to year fraction
    """
    date = pd.to_datetime(date)
    if isinstance(date, pd.Series):
        res = ((date.dt.strftime("%j")).astype(float) - 1) / 366 + (date.dt.strftime("%Y").astype(float))
    else:
        res = (float(date.strftime("%j")) - 1) / 366 + float(date.strftime("%Y"))
    return res


def to_list(x):
    """
    converts variable to a list
    :param x: variable to convert
    :return: variable converted to list
    """
    if x is None:
        res = []
    elif isinstance(x, str):
        res = [x]
    elif isinstance(x, list):
        res = x
    else:
        res = list(x)
    return res


def print_pv_to_excel(pv, writer, sheet_name, startrow=0, startcol=0):
    pv.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol)
    return [startrow + len(pv) + 2, startcol + len(pv.columns) + 2]


def concat_clean(df1, df2):
    """
    concatenate 2 dataframes and removes duplicate columns found
    :param df1:
    :param df2:
    :return:
    """
    df = pd.concat([df1, df2], axis=1, sort=True)
    df.sort_index(axis=1, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.reindex([x for x in df.index if x != 'not known'] + ['not known'])


def clean_df(df, to_numeric=True, filter_col=None, filter_value=None, threshold_col=None, threshold_value=None):
    """
    cleans/standardizes dataframe data
    :param df:
    :param to_numeric: to convert numerical-like data to numeric type
    :param filter_col: to slice on a specific column based on a value
    :param filter_value: value to slice filter_col on
    :param threshold_col: to slice on a specific column based on a minimum threshold
    :param threshold_value: threshold (minimum) to slice threshold_col on
    :return: cleaned dataframe
    """
    df.columns = df.columns.str.lower()
    df.rename(columns={'brc_id': 'brcid', 'ï»¿brcid': 'brcid'}, inplace=True)
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
    """
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    """
    duplicate_column_names = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            other_col = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(other_col):
                duplicate_column_names.add(df.columns.values[y])

    return list(duplicate_column_names)


def drop_duplicate_columns(df):
    new_df = df.drop(columns=get_duplicate_columns(df))
    return new_df


# COUNT NUMBER WORDS IN TEXT FILE
def count_words(file_name):
    num_words = 0
    with open(file_name, 'r') as f:
        for line in f:
            words = line.split()
            num_words += len(words)
    return num_words


# read csv file
# file_path=r'T:\aurelie_mascio\multimorbidity\corpus\multimorbidity_attachments_discharge_docs.csv'
# TextFileReader=pd.read_csv(file_path,usecols=usecols,chunksize=10000,low_memory=False,header=0,encoding='utf8',engine='c',error_bad_lines=False)
# test=TextFileReader.get_chunk(3)
# row_count = sum(1 for row in TextFileReader)
def super_read_csv(file_path, usecols=None, clean_results=True, filter_col=None, filter_value=None, threshold_col=None,
                   threshold_value=None, read_from_SQL_export=False):
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
    res.columns = res.columns.str.lower()
    return res


# LOAD SYMPTOMS APP RESULTS (headers specific to CRIS NLP apps)
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
# RANDOM STUFF
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
# DATE FUNCTIONS FOR LONGITUDINAL MODELLING
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
# GENERATE LONGITUDINAL DATA FROM RAW DATA (HOSPITAL STAYS, MEASURES...)
##############################################################################
# test_file=r'T:\aurelie_mascio\CRIS data\F20_ward_stays_data_fromDB.csv'
# test_df=pd.read_csv(test_file,header=0)
# df=test_df.head().copy()
# df['dob']=datetime.datetime(1986, month=8, day=13)

# FUNCTION TO GENERATE LONGITUDINAL DATA BY AGE
def convert_to_longitudinal_age(df,
                                key_col='brcid',
                                dob_col='dob',
                                start_date_col='actual_start_date',  # for interval only
                                end_date_col='actual_end_date',  # for interval only
                                measure_date_col='rating_date',  # for rating only
                                measure_col='length_stay_days',  # for rating only
                                dob_file=r'T:\aurelie_mascio\CRIS data\F20_patients_documents_details_from_DB.csv',
                                mode='interval'):  # interval: need to measure length (e.g. length of hospital stay), not interval: need to grab measure (e.g. HoNOS) by age
    df = clean_df(df)
    dob_data = pd.read_csv(dob_file, header=0)
    #######################################
    # TODO: JOIN WITH DOB DATA
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


# FUNCTION TO GENERATE LONGITUDINAL DATA BY YEAR
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
