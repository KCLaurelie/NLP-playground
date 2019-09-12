import pandas as pd
import numpy as np
import code_utils.general_utils as gutils
import datetime
import time
from sklearn.preprocessing import MinMaxScaler


class Dataset:
    def __init__(self, file_path, key='brcid', timestamp='age_at_score',  # data path and keys
                 baseline_cols=None, na_values=None,  # identify columns
                 to_predict='score_combined', regressors=('age_at_score',),  # for regression model
                 to_bucket='age_at_score', bucket_min=50, bucket_max=90, interval=0.5, min_obs=3,  # to create groups
                 ):
        """
        create dataset object for trajectories modelling
        :param file_path: path of file containing data
        :param key: group identification (generally individual identification, e.g. brcid)
        :param timestamp: key used as time measure (for baseline values, the oldest/smallest timestamp will be used)
        :param baseline_cols: columns for which we want to keep baseline values
        :param na_values: value to use to replace missing data
        :param to_predict: measure to predict
        :param regressors: list of regressors for prediction modelling
        :param to_bucket: on what variable to bucket the data if applicable (will groupby based on this variable)
        :param bucket_min: min cutoff value for bucketting
        :param bucket_max: max cutoff value for bucketting
        :param interval: interval to use for bucketting (needs to be between 0 and 1)
        :param min_obs: remove individuals having less than min_obs observations
        """
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
        print("reading data (method from parent class)")
        df = pd.read_csv(self.file_path, header=0, low_memory=False)
        df.columns = df.columns.str.lower()
        df_baseline = pd.DataFrame()
        self.data = {'data': df, 'data_baseline': df_baseline}
        return 0

    def bucket_data(self, additional_cols_to_keep=None, timestamp_cols=None):
        print("bucketting data (method from parent class)")
        cols_to_keep = list(dict.fromkeys(gutils.to_list(self.key) + gutils.to_list(self.regressors) + gutils.to_list(
            self.to_bucket) + gutils.to_list(self.to_predict) + gutils.to_list(additional_cols_to_keep)))
        # only use data within bucket boundaries
        mask_bucket = (self.data['data'][self.to_bucket] >= self.bucket_min) & (
                self.data['data'][self.to_bucket] <= self.bucket_max)
        df = self.data['data'].loc[mask_bucket, cols_to_keep]
        if self.na_values is not None:
            df.fillna(self.na_values, inplace=True)
        # transform bool cols to "yes"/"no" so they are not averaged out in the groupby
        bool_cols = [col for col in df.columns if df[col].value_counts().index.isin([0, 1]).all()]
        if len(bool_cols) > 0: df[bool_cols] = df[bool_cols].replace({0: 'no', 1: 'yes'})
        # detect numerical and categorical columns
        categoric_col = [col for col in df.select_dtypes(include=['object', 'category']).columns if (self.key not in col)]
        numeric_col = [col for col in df._get_numeric_data().columns if (col not in [self.key])]
        # group by buckets
        bucket_col = self.to_bucket + '_upbound'
        df[bucket_col] = gutils.round_nearest(df[self.to_bucket], self.interval, 'up')
        # we aggregate by average for numeric variables and baseline value for categorical variables
        keys = categoric_col + numeric_col
        values = ['first'] * len(categoric_col) + ['mean'] * len(numeric_col)
        grouping_dict = dict(zip(keys, values))

        df_grouped = df.groupby([self.key] + [bucket_col], as_index=False).agg(grouping_dict)
        df_grouped = df_grouped.sort_values([self.key, self.to_bucket])

        df_grouped['occur'] = df_grouped.groupby(self.key)[self.key].transform('size')
        df_grouped = df_grouped[(df_grouped['occur'] >= self.min_obs)]
        df_grouped['counter'] = df_grouped.groupby(self.key).cumcount() + 1
        for x in timestamp_cols:
            df_grouped[x+'_upbound'] = gutils.round_nearest(df_grouped[x], self.interval, 'up')
            df_grouped[x+'_centered'] = df_grouped[x+'_upbound'] - df_grouped[x+'_upbound'].min()
        self.data['data_grouped'] = df_grouped

        # now update df and df_baseline with patients who made the cut for modelling
        keys_to_keep = list(df_grouped[self.key].unique())
        self.data['data']['include'] = np.where(mask_bucket & (self.data['data'][self.key].isin(keys_to_keep)), 'yes',
                                                'no')
        self.data['data_baseline']['include'] = np.where(self.data['data_baseline'][self.key].isin(keys_to_keep), 'yes',
                                                         'no')
        return 0

    def prep_data(self, load_type='all'):
        self.read_and_clean_data()
        if load_type == 'all':
            self.bucket_data()
        else:
            self.data['data_grouped'] = pd.DataFrame()
        return 0

    def regression_cleaning(self, normalize=False, dummyfy=False, keep_only_baseline=False):
        if self.data is None or self.data['data_grouped'] is None:
            self.prep_data(load_type='all')
        df = self.data['data_grouped']
        if normalize:
            numeric_cols = [col for col in df[self.regressors]._get_numeric_data().columns]
            cols_to_normalize = [self.to_predict] + [col for col in numeric_cols]
            scaler = MinMaxScaler()
            x = df[cols_to_normalize].values
            scaled_values = scaler.fit_transform(x)
            df[cols_to_normalize] = scaled_values
        if dummyfy:
            cols_to_dummyfy = df[self.regressors].select_dtypes(include=['object', 'category']).columns
            dummyfied_df = pd.get_dummies(df[cols_to_dummyfy])
            df = pd.concat([df.drop(columns=cols_to_dummyfy), dummyfied_df], axis=1, sort=True)
        if keep_only_baseline:
            to_drop = [col for col in df.columns if ('_baseline' in col)
                       and col.replace('_baseline', '') not in gutils.to_list(self.to_predict)
                       and col.replace('_baseline', '') in df.columns]
            df.drop(columns=to_drop, inplace=True)
        return df

    def write_report(self, output_file_path):
        raise NotImplementedError("Please Implement this method")

    def check(self):
        print("Dataset object created from", self.file_path)

    def expand_all_timestamps(self, timestamp_col='counter', merge_with_df=False):
        pv = self.pivot(index=timestamp_col, columns=self.key, values=self.to_predict)
        res = pv.unstack().reset_index()
        res.rename(columns={0: self.to_predict}, inplace=True)
        if merge_with_df:
            df = self.dataset['df_grouped'][[self.key] + [x for x in self.dataset['df_grouped'].columns if 'baseline' in x]]
            res = df.merge(res, on=self.key)
        return res


class DatasetMMSE(Dataset):
    def __init__(self, health_numeric_cols=None, cols_to_pivot=None, index_to_pivot=None, index_to_pivot_baseline=None,
                 agg_funcs=None, timestamp_cols=None, **kwargs):
        """

        :param health_numeric_cols: columns containing health numeric data (for report generation)
        :param cols_to_pivot: groups to use (needs to be a list, at the moment maximum of 2 groups allowed)
        :param index_to_pivot: variables to generate stats for
        :param index_to_pivot_baseline: baseline variables to generate stats for
        :param agg_funcs: functions to use for reporting. supports list of functions
        :param timestamp_cols: columns containing timestamp data (e.g. date of measure)
        :param kwargs: parameters from parent class
        """
        super(DatasetMMSE, self).__init__(**kwargs)
        self.health_numeric_cols = health_numeric_cols
        self.cols_to_pivot = gutils.to_list(cols_to_pivot)
        self.index_to_pivot = gutils.to_list(index_to_pivot)
        self.index_to_pivot_baseline = gutils.to_list(index_to_pivot_baseline)
        self.agg_funcs = agg_funcs
        self.timestamp_cols = timestamp_cols

    def read_and_clean_data(self):
        Dataset.read_and_clean_data(self)
        print("reading data (method from mmse class)")
        df = self.data['data']
        df_baseline = self.data['data_baseline']
        static_data_col = [col for col in df.select_dtypes(include=['object']).columns if
                           ('date' not in col) and ('age' not in col) and ('score_bucket' not in col)]
        df[static_data_col] = df[static_data_col].apply(lambda x: x.astype(str).str.lower())
        df['age_bucket_report'] = '[' + (gutils.round_nearest(df.age_at_score, 5, 'up') - 5).astype(str) + '-' \
                                  + gutils.round_nearest(df.age_at_score, 5, 'up').astype(str) + ']'
        df['imd_bucket'] = '[' + (gutils.round_nearest(df.imd, 10, 'up') - 10).astype(str) + '-' \
                           + gutils.round_nearest(df.imd, 10, 'up').astype(str) + ']'
        df.loc[df['ethnicity'].str.contains('other', case=False, na=False), 'ethnicity'] = 'other ethnicity'
        df[static_data_col] = df[static_data_col].replace(
            ['null', 'unknown', np.nan, 'nan', 'other', 'not specified', 'not disclosed', 'not stated (z)', 'Not Known']
            , 'not known')
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
        for col in self.timestamp_cols:
            df[col] = gutils.date2year(df[col])
        if self.baseline_cols is not None:
            df_baseline = df.sort_values([self.key, self.timestamp]).groupby(self.key).first().reset_index()
            df = df.merge(df_baseline[self.baseline_cols], on=self.key, suffixes=('', '_baseline'))
            df = df.sort_values([self.key, self.timestamp])

        self.data = {'data': df, 'data_baseline': df_baseline}
        return 0

    def bucket_data(self):
        print("bucketting data (method from mmse class)")
        Dataset.bucket_data(self,
                            additional_cols_to_keep=gutils.to_list(self.index_to_pivot) + gutils.to_list(self.cols_to_pivot),
                            timestamp_cols=['score_date'])

    def write_report(self, output_file_path, use_grouped=False):
        if self.data is None or self.data['data_grouped'] is None:
            self.prep_data(load_type='all')
        df = self.data['data_grouped'] if use_grouped else self.data['data'][self.data['data'].include == 'yes']
        df_baseline = self.data['data_baseline'][self.data['data_baseline'].include == 'yes']

        index_baseline = [col for col in df.columns if 'bucket_baseline' in col or 'smoking_status' in col]
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%Hh%M')
        writer = pd.ExcelWriter(output_file_path.replace('.xlsx', '_' + st + '.xlsx'), engine='xlsxwriter')
        cpt_row = 0
        pv_master = pd.DataFrame()
        for var in self.index_to_pivot + index_baseline:
            pv_scores = my_pivot(df, values=self.to_predict, index=var, cols_to_pivot=self.cols_to_pivot,
                                 aggfunc=self.agg_funcs)
            pv_pop = my_pivot(df, values=self.key, index=var, cols_to_pivot=self.cols_to_pivot,
                              aggfunc=pd.Series.nunique)
            pv_pop.columns = [str(self.key) + '_' + x for x in pv_pop.columns]
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

        pv_measures = self.data['data_grouped'].pivot_table(values=self.key, index=self.cols_to_pivot[0],
                                                            columns='occur', aggfunc=pd.Series.nunique,
                                                            margins=True).fillna(0)
        pv_measures.to_excel(writer, sheet_name='nb_measures', startrow=0)

        health_stats = df_baseline.groupby(self.cols_to_pivot[0])[self.health_numeric_cols].agg(self.agg_funcs)
        health_stats_total = df_baseline[self.health_numeric_cols].agg(self.agg_funcs).unstack()
        health_stats = pd.concat([health_stats, pd.DataFrame([health_stats_total]).rename(index={0: 'All'})], axis=0,
                                 sort=True)
        if isinstance(self.cols_to_pivot, list) and len(self.cols_to_pivot) > 1:  # several groups to use for pivot
            health_stats1 = df_baseline.groupby(self.cols_to_pivot[1])[self.health_numeric_cols].agg(self.agg_funcs)
            health_stats = pd.concat([health_stats, health_stats1], axis=0, sort=True)
        health_stats.to_excel(writer, sheet_name='health_stats', startrow=0)
        writer.save()

        return 0


default_dataset = DatasetMMSE(
    file_path='https://raw.githubusercontent.com/KCLaurelie/toy-models/master/longitudinal_models/mmse_trajectory_synthetic.csv?token=ALKII2WYE6LIQJM6RFQCEQK5JSILS',
    baseline_cols=['brcid', 'age_at_score', 'score_date', 'score_combined', 'bmi_score', 'plasma_glucose_value', 'diastolic_value',
                   'systolic_value', 'smoking_status', 'bmi_bucket', 'diabetes_bucket', 'bp_bucket', 'imd_bucket'],
    key='brcid',
    timestamp='age_at_score',
    health_numeric_cols=['bmi_score', 'plasma_glucose_value', 'diastolic_value', 'systolic_value'],
    to_predict='score_combined',
    regressors=['patient_diagnosis_class', 'patient_diagnosis_super_class', 'score_date', 'score_date_baseline',
                'age_at_score', 'age_at_score_baseline', 'score_combined_baseline', 'gender', 'ethnicity_group',
                'first_language', 'occupation', 'living_status', 'marital_status', 'education_bucket_raw',
                'smoking_status_baseline', 'plasma_glucose_value_baseline', 'diastolic_value_baseline', 'systolic_value_baseline',
                'bmi_score_baseline', 'diabetes_bucket_baseline', 'bp_bucket_baseline', 'bmi_bucket_baseline', 'imd_bucket_baseline'],
    na_values=None,
    to_bucket='age_at_score',
    bucket_min=50,
    bucket_max=90,
    interval=0.5,
    min_obs=3,
    cols_to_pivot=['patient_diagnosis_super_class'],  # , 'patient_diagnosis_class'],
    index_to_pivot=['age_bucket_report', 'gender', 'ethnicity_group', 'first_language', 'occupation', 'living_status',
                    'marital_status', 'education_bucket_raw'],
    index_to_pivot_baseline='age_bucket_report',
    agg_funcs=['count', np.mean, np.std],
    timestamp_cols=['score_date']
)


##############################################################################################
# UTIL FUNCTIONS
##############################################################################################
def my_pivot(df, cols_to_pivot, values, index, aggfunc=pd.Series.nunique):
    pv = df.pivot_table(values=values, index=index, columns=cols_to_pivot[0], aggfunc=aggfunc, margins=True).fillna(0)
    # in case we have a second group to use for pivot
    if isinstance(cols_to_pivot, list) and len(cols_to_pivot) > 1:
        try:  # to avoid duplicates between super class and class: for multi-index
            pv.drop(['organic only', 'All'], level=0, axis=1, inplace=True)  # if only 1 agg function
            pv.drop(['organic only', 'All'], level=1, axis=1, inplace=True)  # if more than 1 agg function
        except:  # to avoid duplicates between super class and class: for single index
            pv.drop(columns=['All'], inplace=True)
        pv1 = df.pivot_table(values=values, index=index, columns=cols_to_pivot[1], aggfunc=aggfunc,
                             margins=True).fillna(0)
        pv = pd.concat([pv1, pv], axis=1, sort=True)

    if isinstance(aggfunc, list):
        pv = pv.swaplevel(axis=1)
        pv.columns = ['_'.join(x) for x in pv.columns]
    # sort by column name
    pv.sort_index(axis=1, inplace=True)
    pv.rename(index={'All': 'All_' + index}, inplace=True)
    pv = pv.loc[:, ~pv.columns.duplicated()]
    return pv.reindex([x for x in pv.index if x != 'not known'] + ['not known'])
