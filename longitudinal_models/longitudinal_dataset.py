import os
import pandas as pd
import numpy as np
import longitudinal_models.general_utils as gutils
import longitudinal_models.plot_utils as pltu
import datetime
import time

try:
    os.chdir(r'T:\aurelie_mascio\python_scripts')  # for use on CRIS computers
except:
    pass


class Dataset:
    def __init__(self, file_path, key='brcid',  # data path and key
                 baseline_cols=None, health_numeric_cols=None, na_values=None,  # identify columns
                 to_predict='score_combined', regressors=['age_at_score'],  # for regression model
                 to_bucket='age_at_score', bucket_min=50, bucket_max=90, interval=0.5, min_obs=3,  # to create groups
                 cols_to_pivot=None, index_to_pivot=None, index_to_pivot_baseline=None, agg_funcs=None  # for reporting
                 ):
        self.file_path = file_path
        self.baseline_cols = baseline_cols
        self.key = key
        self.health_numeric_cols = health_numeric_cols
        self.to_predict = to_predict
        self.regressors = list(regressors)
        self.na_values = na_values
        self.to_bucket = to_bucket
        self.bucket_min = bucket_min
        self.bucket_max = bucket_max
        self.interval = interval
        self.min_obs = min_obs
        self.cols_to_pivot = cols_to_pivot
        self.index_to_pivot = index_to_pivot
        self.index_to_pivot_baseline = index_to_pivot_baseline
        self.agg_funcs = agg_funcs

    def load_data(self, load_type='all'):
        df, df_baseline = read_and_clean_data(self.file_path, self.baseline_cols)
        if load_type == 'all':
            df_grouped = prep_data_for_model(df, regressors=self.regressors, to_predict=self.to_predict,
                                             col_to_bucket=self.to_bucket, bucket_min=self.bucket_min,
                                             bucket_max=self.bucket_max, interval=self.interval, min_obs=self.min_obs,
                                             na_values=self.na_values)
        else:
            df_grouped = pd.DataFrame()
        res = {'data': df,
               'data_baseline': df_baseline,
               'data_grouped': df_grouped}
        return res

    def create_report(self, output_file_path, data=None):
        if data is None:
            data = self.load_data(load_type='not all')
        write_report(output_file_path=output_file_path, df_all=data['data'], df_baseline=data['data_baseline'],
                     cols_to_pivot=self.cols_to_pivot, index_to_pivot=self.index_to_pivot,
                     index_to_pivot_baseline=self.index_to_pivot_baseline, score_funcs=self.agg_funcs,
                     health_numeric_cols=self.health_numeric_cols, key=self.key, score_metric=self.to_predict)

    def check(self):
        print("Dataset object created from", self.file_path)


default_dataset = Dataset(
    file_path='https://raw.githubusercontent.com/KCLaurelie/toy-models/master/longitudinal_models/mmse_trajectory_synthetic.csv?token=ALKII2U7IKICWCAEWC22H7S5EZQBW',
    baseline_cols=['brcid', 'age_at_score', 'score_combined', 'bmi_score', 'plasma_glucose_value', 'diastolic_value',
                   'systolic_value', 'smoking_status', 'bmi_bucket', 'diabetes_bucket', 'bp_bucket'],
    key='brcid',
    health_numeric_cols=['bmi_score', 'plasma_glucose_value', 'diastolic_value', 'systolic_value'],
    to_predict='score_combined',
    regressors=['patient_diagnosis_class', 'patient_diagnosis_super_class', 'score_date', 'age_at_score',
                'gender', 'ethnicity_group', 'first_language', 'occupation', 'living_status', 'marital_status',
                'education_bucket_raw', 'is_active', 'has_depression_anxiety_diagnosis', 'has_agitation_diagnosis',
                'smoking_status', 'aggression_status', 'plasma_glucose_value', 'diabetes_bucket', 'diastolic_value',
                'systolic_value', 'bp_bucket', 'bmi_score', 'bmi_bucket'],
    na_values=None,
    to_bucket='age_at_score',
    bucket_min=50,
    bucket_max=90,
    interval=0.5,
    min_obs=3,
    cols_to_pivot=['patient_diagnosis_super_class', 'patient_diagnosis_class'],
    index_to_pivot=['age_bucket_report', 'gender', 'ethnicity_group', 'first_language', 'occupation', 'living_status',
                    'marital_status', 'education_bucket_raw'],
    index_to_pivot_baseline='age_bucket_report',
    agg_funcs=['count', np.mean, np.std]
)
# default_dataset.create_report(r'C:\Users\K1774755\Downloads\testmmsereport.xlsx')


##############################################################################################
# UTIL FUNCTIONS
##############################################################################################
def my_pivot(df, cols_to_pivot, values, index, aggfunc=pd.Series.nunique):
    pv0 = df.pivot_table(values=values, index=index, columns=cols_to_pivot[0], aggfunc=aggfunc, margins=True).fillna(0)
    try:  # to avoid duplicates between super class and class: for multi-index
        pv0.drop(['organic only', 'All'], level=0, axis=1, inplace=True)  # if only 1 agg function
        pv0.drop(['organic only', 'All'], level=1, axis=1, inplace=True)  # if more than 1 agg function
    except:  # to avoid duplicates between super class and class: for single index
        pv0.drop(columns=['All'], inplace=True)
    pv1 = df.pivot_table(values=values, index=index, columns=cols_to_pivot[1], aggfunc=aggfunc, margins=True).fillna(0)
    pv = pd.concat([pv1, pv0], axis=1, sort=True)
    if isinstance(aggfunc, list):
        pv = pv.swaplevel(axis=1)
        pv.columns = ['_'.join(x) for x in pv.columns]
    # sort by column name
    pv.sort_index(axis=1, inplace=True)
    pv.rename(index={'All': 'All_' + index}, inplace=True)
    pv = pv.loc[:, ~pv.columns.duplicated()]
    return pv.reindex([x for x in pv.index if x != 'not known'] + ['not known'])


def print_pv_to_excel(pv, writer, sheet_name, startrow=0, startcol=0):
    pv.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol)
    return [startrow + len(pv) + 2, startcol + len(pv.columns) + 2]


def concat_clean(df1, df2):
    df = pd.concat([df1, df2], axis=1, sort=True)
    df.sort_index(axis=1, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.reindex([x for x in df.index if x != 'not known'] + ['not known'])


##############################################################################################
# READING/CLEANING/ENRICHING THE DATA
##############################################################################################
def prep_data_for_model(df, regressors, to_predict, col_to_bucket='age_at_score',
                        bucket_min=50, bucket_max=90, interval=0.5, min_obs=3, na_values=None):
    cols_to_keep = ['brcid'] + regressors + [to_predict]
    # only use data within bucket boundaries
    df = df[(df[col_to_bucket] >= bucket_min) & (df[col_to_bucket] <= bucket_max)][cols_to_keep]
    if na_values is not None:
        df.fillna(na_values, inplace=True)
    # transform bool cols to "yes"/"no" so they are not averaged out in the groupby
    bool_cols = [col for col in df if df[col].dropna().value_counts().index.isin([0, 1]).all()]
    if len(bool_cols) > 0: df[bool_cols] = df[bool_cols].replace({0: 'no', 1: 'yes'})
    # detect numerical and categorical columns
    static_data_col = [col for col in df.select_dtypes(include=['object', 'category']).columns if ('brcid' not in col)]
    numeric_col = [col for col in df._get_numeric_data().columns if ('brcid' not in col)]
    # group by buckets
    bucket_col = col_to_bucket + '_upper_bound'
    df[bucket_col] = np.ceil(df[col_to_bucket] / interval) * interval
    # we aggregate by average for numeric variables and baseline value for categorical variables
    keys = static_data_col + numeric_col
    values = ['first'] * len(static_data_col) + ['mean'] * len(numeric_col)
    grouping_dict = dict(zip(keys, values))

    df_grouped = df.groupby(['brcid'] + [bucket_col], as_index=False).agg(grouping_dict)
    df_baseline = df_grouped.sort_values(['brcid', 'age_at_score']).groupby('brcid').first()
    df_grouped = df_grouped.merge(df_baseline, on='brcid', suffixes=('', '_baseline'))
    df_grouped = df_grouped.sort_values(['brcid', 'age_at_score'])

    df_grouped['occur'] = df_grouped.groupby('brcid')['brcid'].transform('size')
    df_grouped = df_grouped[(df_grouped['occur'] >= min_obs)]
    # df_grouped['counter'] = df.groupby('brcid').cumcount() + 1
    all_buckets = pd.DataFrame(data=np.arange(start=bucket_min, stop=bucket_max, step=interval), columns=[bucket_col])
    all_buckets['counter'] = np.arange(start=1, stop=len(all_buckets) + 1, step=1)
    df_grouped = df_grouped.merge(all_buckets, on=bucket_col).sort_values(['brcid', bucket_col])

    return df_grouped.reset_index(drop=True)


def read_and_clean_data(file_path, baseline_cols=None):
    df = pd.read_csv(file_path, header=0, low_memory=False)
    df.columns = df.columns.str.lower()
    static_data_col = [col for col in df.select_dtypes(include=['object']).columns if
                       ('date' not in col) and ('age' not in col) and ('score_bucket' not in col)]
    df[static_data_col] = df[static_data_col].apply(lambda x: x.astype(str).str.lower())
    df['age_bucket_report'] = '[' + (np.ceil(df['age_at_score'] / 5) * 5 - 5).astype(str) + '-' + \
                              (np.ceil(df['age_at_score'] / 5) * 5).astype(str) + ']'
    df.loc[df['first_language'].str.contains('other', case=False, na=False), 'first_language'] = 'other language'
    df.loc[df['ethnicity'].str.contains('other', case=False, na=False), 'ethnicity'] = 'other ethnicity'
    df[static_data_col] = df[static_data_col].replace(
        ['null', 'unknown', np.nan, 'nan', 'other', 'not specified', 'not disclosed', 'not stated (z)'], 'not known')
    df.replace({'patient_diagnosis_class': {'smi+organic': 'schizo+bipolar+organic'}}, inplace=True)
    df['ethnicity_group'] = 'other ethnicity'
    df.loc[df['ethnicity'].str.contains('not known', case=False, na=False), 'ethnicity_group'] = 'not known'
    df.loc[df['ethnicity'].str.contains('|'.join(['and', 'mixed']), case=False, na=False), ['ethnicity',
                                                                                            'ethnicity_group']] = 'mixed'
    df.loc[df['ethnicity'].str.contains('african', case=False, na=False), 'ethnicity_group'] = 'black african'
    df.loc[df['ethnicity'].str.contains('caribbean', case=False, na=False), 'ethnicity_group'] = 'black caribbean'
    df.loc[df['ethnicity'].str.contains('|'.join(['irish', 'british', 'other white']), case=False,
                                        na=False), 'ethnicity_group'] = 'white'
    df.loc[df['ethnicity'].str.contains('|'.join(['bangladesh', 'pakistan', 'india']), case=False,
                                        na=False), 'ethnicity_group'] = 'indian'
    df.loc[df['marital_status'].str.contains('|'.join(['married', 'cohabit']), case=False,
                                             na=False), 'marital_status'] = 'married or cohabiting'
    df.loc[df['marital_status'].str.contains('|'.join(['divorce', 'widow', 'separat', 'single']), case=False,
                                             na=False), 'marital_status'] = 'single or separated'
    df.loc[~df['first_language'].str.contains('|'.join(['english', 'known']), case=False,
                                              na=False), 'first_language'] = 'not english'
    df.loc[df['occupation'].str.contains('employ', case=False, na=False) &
           ~df['occupation'].str.contains('unemploy', case=False, na=False), 'occupation'] = 'employed'
    df.loc[df['occupation'].str.contains('student', case=False, na=False), 'occupation'] = 'student'
    df.loc[df['living_status'].str.contains('|'.join(['nurs', 'trust']), case=False,
                                            na=False), 'living_status'] = 'nursing/residential/trust'
    df['bmi_bucket'] = gutils.bmi_category(df.bmi_score)
    df['bp_bucket'] = gutils.blood_pressure(df.systolic_value, df.diastolic_value)
    df['diabetes_bucket'] = gutils.diabetes(df.plasma_glucose_value)

    df['score_time_period'] = pd.PeriodIndex(pd.to_datetime(df.score_date), freq='Q').astype(str)
    df.score_time_period.replace({'Q1': 'H1', 'Q2': 'H1', 'Q3': 'H2', 'Q4': 'H2'}, regex=True, inplace=True)
    if baseline_cols is not None:
        df_baseline = df.sort_values(['brcid', 'age_at_score']).groupby('brcid').first().reset_index()
        df = df.merge(df_baseline[baseline_cols], on='brcid', suffixes=('', '_baseline'))
        df = df.sort_values(['brcid', 'age_at_score'])
    else:
        df_baseline = pd.DataFrame()
    return [df, df_baseline]


##############################################################################################
# POPULATION STATISTICS
##############################################################################################

def write_report(output_file_path, df_all, df_baseline, cols_to_pivot, index_to_pivot, score_funcs,
                 health_numeric_cols, key, score_metric, index_to_pivot_baseline):
    index_baseline = [col for col in df_all.columns if 'bucket_baseline' in col or 'smoking_status' in col]
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    writer = pd.ExcelWriter(output_file_path.replace('.xlsx', '_' + st + '.xlsx'), engine='xlsxwriter')
    cpt_row = 0
    pv_master = pd.DataFrame()
    for var in index_to_pivot + index_baseline:
        pv_scores = my_pivot(df_all, values=score_metric, index=var, cols_to_pivot=cols_to_pivot, aggfunc=score_funcs)
        pv_pop = my_pivot(df_all, values=key, index=var, cols_to_pivot=cols_to_pivot, aggfunc=pd.Series.nunique)
        pv_pop.columns = [x + '_' + str(key) for x in pv_pop.columns]
        pv = concat_clean(pv_scores, pv_pop)
        pv_master = pd.concat([pv_master, pv], axis=0, sort=False)
        pv.to_excel(writer, sheet_name='summary_separate', startrow=cpt_row)
        cpt_row += len(pv_scores) + 4

    # format header
    header = pd.DataFrame([[i.split('_', 1)[1] for i in pv_master.columns]],
                          columns=[i.split('_', 1)[0] for i in pv_master.columns])
    header.to_excel(writer, sheet_name='summary', startrow=0)
    pv_master.to_excel(writer, sheet_name='summary', startrow=len(header) + 1, header=False)

    pv_baseline = df_baseline.pivot_table(values=key, index=index_to_pivot_baseline, columns=cols_to_pivot,
                                          aggfunc=pd.Series.nunique, margins=True).fillna(0)
    pv_baseline.to_excel(writer, sheet_name='first_measure', startrow=0)

    health_stats0 = df_baseline.groupby(cols_to_pivot[0])[health_numeric_cols].agg(score_funcs)
    health_stats1 = df_baseline.groupby(cols_to_pivot[1])[health_numeric_cols].agg(score_funcs)
    health_stats = pd.concat([health_stats0, health_stats1], axis=0, sort=True)
    health_stats.to_excel(writer, sheet_name='health_stats', startrow=0)
    writer.save()

    return 0
