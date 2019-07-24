import pandas as pd
import numpy as np
import code_utils.general_utils as gutils
import datetime
import time
from code_utils.global_variables import *


class Dataset:
    def __init__(self, file_path, key='brcid', timestamp='age_at_score',  # data path and keys
                 baseline_cols=None, na_values=None,  # identify columns
                 to_predict='score_combined', regressors=('age_at_score',),  # for regression model
                 to_bucket='age_at_score', bucket_min=50, bucket_max=90, interval=0.5, min_obs=3,  # to create groups
                 ):
        self.file_path = file_path
        self.key = key
        self.timestamp = timestamp
        self.baseline_cols = gutils.to_list(baseline_cols)
        self.na_values = na_values
        self.to_predict = gutils.to_list(to_predict)
        self.regressors = gutils.to_list(regressors)
        self.to_bucket = str(to_bucket)
        self.bucket_min = bucket_min
        self.bucket_max = bucket_max
        self.interval = interval
        self.min_obs = min_obs
        self.data = None

    def read_and_clean_data(self):
        print("read data (method from parent class)")
        df = pd.read_csv(self.file_path, header=0, low_memory=False)
        df.columns = df.columns.str.lower()
        df_baseline = pd.DataFrame()
        self.data = {'data': df, 'data_baseline': df_baseline}
        return 0

    def prep_data_for_model(self):
        print("prep data (method from parent class)")
        df = self.data['data']
        cols_to_keep = gutils.to_list(self.key) + gutils.to_list(self.regressors) + gutils.to_list(self.to_bucket) + gutils.to_list(self.to_predict)
        cols_to_keep = list(dict.fromkeys(cols_to_keep)) # removing duplicates
        # only use data within bucket boundaries
        df = df.loc[(df[self.to_bucket] >= self.bucket_min) & (df[self.to_bucket] <= self.bucket_max), cols_to_keep]
        if self.na_values is not None:
            df.fillna(self.na_values, inplace=True)
        # transform bool cols to "yes"/"no" so they are not averaged out in the groupby
        bool_cols = [col for col in df.columns if df[col].value_counts().index.isin([0, 1]).all()]
        if len(bool_cols) > 0: df[bool_cols] = df[bool_cols].replace({0: 'no', 1: 'yes'})
        # detect numerical and categorical columns
        static_data_col = [col for col in df.select_dtypes(include=['object', 'category']).columns if
                           (self.key not in col)]
        numeric_col = [col for col in df._get_numeric_data().columns if (self.key not in col)]
        # group by buckets
        bucket_col = self.to_bucket + '_upper_bound'
        df[bucket_col] = np.ceil(df[self.to_bucket] / self.interval) * self.interval
        # we aggregate by average for numeric variables and baseline value for categorical variables
        keys = static_data_col + numeric_col
        values = ['first'] * len(static_data_col) + ['mean'] * len(numeric_col)
        grouping_dict = dict(zip(keys, values))

        df_grouped = df.groupby([self.key] + [bucket_col], as_index=False).agg(grouping_dict)
        df_grouped = df_grouped.sort_values([self.key, self.to_bucket])

        df_grouped['occur'] = df_grouped.groupby(self.key)[self.key].transform('size')
        df_grouped = df_grouped[(df_grouped['occur'] >= self.min_obs)]
        # df_grouped['counter'] = df.groupby(key).cumcount() + 1
        all_buckets = pd.DataFrame(data=np.arange(start=self.bucket_min, stop=self.bucket_max + self.interval, step=self.interval),
                                   columns=[bucket_col])
        all_buckets['counter'] = np.arange(start=1, stop=len(all_buckets) + 1, step=1)
        df_grouped = df_grouped.merge(all_buckets, on=bucket_col).sort_values([self.key, bucket_col])
        df_grouped = df_grouped.reset_index(drop=True)
        self.data['data_grouped'] = df_grouped
        # df.loc[df.brcid==326, ['score_combined', 'age_at_score', bucket_col]]
        return 0

    def prep_data(self, load_type='all'):
        self.read_and_clean_data()
        if load_type == 'all':
            self.prep_data_for_model()
        else:
            self.data['data_grouped'] = pd.DataFrame()
        return 0

    def write_report(self, output_file_path):
        raise NotImplementedError("Please Implement this method")

    def check(self):
        print("Dataset object created from", self.file_path)


class DatasetMMSE(Dataset):
    def __init__(self, health_numeric_cols=None, cols_to_pivot=None, index_to_pivot=None, index_to_pivot_baseline=None,
                 agg_funcs=None, **kwargs):
        super(DatasetMMSE, self).__init__(**kwargs)
        self.health_numeric_cols = health_numeric_cols
        self.cols_to_pivot = cols_to_pivot
        self.index_to_pivot = index_to_pivot
        self.index_to_pivot_baseline = index_to_pivot_baseline
        self.agg_funcs = agg_funcs

    def read_and_clean_data(self):
        Dataset.read_and_clean_data(self)
        print("read data (method from mmse class)")
        df = self.data['data']
        df_baseline = self.data['data_baseline']
        static_data_col = [col for col in df.select_dtypes(include=['object']).columns if
                           ('date' not in col) and ('age' not in col) and ('score_bucket' not in col)]
        df[static_data_col] = df[static_data_col].apply(lambda x: x.astype(str).str.lower())
        df['age_bucket_report'] = '[' + (np.ceil(df['age_at_score'] / 5) * 5 - 5).astype(str) + '-' + \
                                  (np.ceil(df['age_at_score'] / 5) * 5).astype(str) + ']'
        df.loc[df['first_language'].str.contains('other', case=False, na=False), 'first_language'] = 'other language'
        df.loc[df['ethnicity'].str.contains('other', case=False, na=False), 'ethnicity'] = 'other ethnicity'
        df[static_data_col] = df[static_data_col].replace(
            ['null', 'unknown', np.nan, 'nan', 'other', 'not specified', 'not disclosed', 'not stated (z)'],
            'not known')
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
        if self.baseline_cols is not None:
            df_baseline = df.sort_values([self.key, self.timestamp]).groupby(self.key).first().reset_index()
            df = df.merge(df_baseline[self.baseline_cols], on='brcid', suffixes=('', '_baseline'))
            df = df.sort_values([self.key, self.timestamp])

        self.data = {'data': df, 'data_baseline': df_baseline}
        return 0

    def write_report(self, output_file_path):
        try:
            df = self.data['data']
            df_baseline = self.data['data_baseline']
        except:
            self.prep_data(load_type='not all')
            df = self.data['data']
            df_baseline = self.data['data_baseline']
        index_baseline = [col for col in df.columns if 'bucket_baseline' in col or 'smoking_status' in col]
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
        writer = pd.ExcelWriter(output_file_path.replace('.xlsx', '_' + st + '.xlsx'), engine='xlsxwriter')
        cpt_row = 0
        pv_master = pd.DataFrame()
        for var in self.index_to_pivot + index_baseline:
            pv_scores = my_pivot(df, values=self.to_predict, index=var, cols_to_pivot=self.cols_to_pivot,
                                 aggfunc=self.agg_funcs)
            pv_pop = my_pivot(df, values=self.key, index=var, cols_to_pivot=self.cols_to_pivot,
                              aggfunc=pd.Series.nunique)
            pv_pop.columns = [x + '_' + str(self.key) for x in pv_pop.columns]
            pv = gutils.concat_clean(pv_scores, pv_pop)
            pv_master = pd.concat([pv_master, pv], axis=0, sort=False)
            pv.to_excel(writer, sheet_name='summary_separate', startrow=cpt_row)
            cpt_row += len(pv_scores) + 4

        # format header
        header = pd.DataFrame([[i.split('_', 1)[1] for i in pv_master.columns]],
                              columns=[i.split('_', 1)[0] for i in pv_master.columns])
        header.to_excel(writer, sheet_name='summary', startrow=0)
        pv_master.to_excel(writer, sheet_name='summary', startrow=len(header) + 1, header=False)

        pv_baseline = df_baseline.pivot_table(values=self.key, index=self.index_to_pivot_baseline,
                                              columns=self.cols_to_pivot, aggfunc=pd.Series.nunique,
                                              margins=True).fillna(0)
        pv_baseline.to_excel(writer, sheet_name='first_measure', startrow=0)

        health_stats0 = df_baseline.groupby(self.cols_to_pivot[0])[self.health_numeric_cols].agg(self.agg_funcs)
        health_stats1 = df_baseline.groupby(self.cols_to_pivot[1])[self.health_numeric_cols].agg(self.agg_funcs)
        health_stats = pd.concat([health_stats0, health_stats1], axis=0, sort=True)
        health_stats.to_excel(writer, sheet_name='health_stats', startrow=0)
        writer.save()

        return 0


default_dataset = DatasetMMSE(
    file_path='https://raw.githubusercontent.com/KCLaurelie/toy-models/master/longitudinal_models/mmse_trajectory_synthetic.csv?token=ALKII2U7IKICWCAEWC22H7S5EZQBW',
    baseline_cols=['brcid', 'age_at_score', 'score_combined', 'bmi_score', 'plasma_glucose_value', 'diastolic_value',
                   'systolic_value', 'smoking_status', 'bmi_bucket', 'diabetes_bucket', 'bp_bucket'],
    key='brcid',
    timestamp='age_at_score',
    health_numeric_cols=['bmi_score', 'plasma_glucose_value', 'diastolic_value', 'systolic_value'],
    to_predict='score_combined',
    regressors=['patient_diagnosis_class', 'patient_diagnosis_super_class', 'score_date', 'age_at_score',
                'score_combined_baseline', 'gender', 'ethnicity_group', 'first_language', 'occupation',
                'living_status', 'marital_status', 'education_bucket_raw', 'smoking_status_baseline',
                'plasma_glucose_value_baseline', 'diastolic_value_baseline', 'systolic_value_baseline',
                'bmi_score_baseline',
                'diabetes_bucket_baseline', 'bp_bucket_baseline', 'bmi_bucket_baseline'],
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
    agg_funcs=['count', np.mean, np.std],
)


# default_dataset.write_report(r'C:\Users\K1774755\Downloads\testmmsereport.xlsx')


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
