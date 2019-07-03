import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm
import longitudinal_models.general_utils as gutils
import longitudinal_models.plot_utils as pltu
import datetime
import time
try: os.chdir(r'T:\aurelie_mascio\python_scripts')  # directory with python library
except: pass


# to do some testing
def init_params():
    file_path = 'https://raw.githubusercontent.com/KCLaurelie/toy-models/master/longitudinal_models/mmse_trajectory_synthetic.csv?token=ALKII2U7IKICWCAEWC22H7S5EZQBW'
    # file_path = os.path.join(root_path, 'honos_trajectory_data3.csv')
    score_funcs = ['count', np.mean, np.std]  # pd.Series.nunique]
    cols_to_pivot = ['patient_diagnosis_super_class', 'patient_diagnosis_class']
    index_to_pivot = ['age_bucket_at_score2', 'gender', 'ethnicity_group', 'first_language', 'occupation', 'living_status', 'marital_status', 'education_bucket_raw']
    health_numeric_cols = ['bmi_score', 'plasma_glucose_value', 'diastolic_value', 'systolic_value']
    baseline_cols = ['brcid', 'age_at_score', 'score_combined', 'bmi_score', 'plasma_glucose_value', 'diastolic_value',
                     'systolic_value', 'smoking_status', 'bmi_bucket', 'diabetes_bucket', 'bp_bucket']
    to_predict = 'score_combined'
    regressors = ['patient_diagnosis_class', 'patient_diagnosis_super_class', 'score_date', 'age_at_score',
                  'gender', 'ethnicity_group', 'first_language', 'occupation', 'living_status', 'marital_status',
                  'education_bucket_raw', 'is_active', 'has_depression_anxiety_diagnosis', 'has_agitation_diagnosis',
                  'smoking_status',  'aggression_status', 'plasma_glucose_value', 'diabetes_bucket', 'diastolic_value',
                  'systolic_value',  'bp_bucket', 'bmi_score', 'bmi_bucket']
    intercept, na_values, bucket_min, bucket_max, interval = ['score_combined_baseline', None, 50, 90, 0.5]

    return [file_path, score_funcs, cols_to_pivot, index_to_pivot, health_numeric_cols, baseline_cols, to_predict,
            regressors, intercept, na_values, bucket_min, bucket_max, interval]


##############################################################################################
# MAIN
##############################################################################################
def create_report(file_path, root_path, cols_to_pivot, index_to_pivot, score_funcs, health_numeric_cols, baseline_cols):
    metric = 'honos' if 'honos' in file_path else 'mmse'
    output_file_path = os.path.join(root_path, metric+'_trajectory_report.xlsx')
    df, df_baseline = read_and_clean_data(file_path, baseline_cols)
    write_report(output_file_path, df, df_baseline, cols_to_pivot, index_to_pivot, score_funcs, health_numeric_cols)
    df_baseline.info()
    df_baseline.describe(include='all')


def run_model(file_path, baseline_cols, to_predict, regressors):
    df = read_and_clean_data(file_path, baseline_cols)[0]
    df_grouped = prep_data_for_model(df, to_predict=to_predict, regressors=regressors)
    df_test = df[df.brcid == 9][['age_at_score', 'score_combined']]
    df_grouped_test = df_grouped[df_grouped.brcid == 9][['age_at_score', 'score_combined']]


##############################################################################################
# UTIL FUNCTIONS
##############################################################################################
def my_pivot(df,
             cols_to_pivot,
             aggfunc=pd.Series.nunique,
             values='score_combined',
             index='gender'):
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
    pv.rename(index={'All': 'All_'+index}, inplace=True)
    pv = pv.loc[:, ~pv.columns.duplicated()]
    return pv.reindex([x for x in pv.index if x != 'not known']+['not known'])


def print_pv_to_excel(pv, writer, sheet_name, startrow=0, startcol=0):
    pv.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol)
    return [startrow + len(pv) + 2, startcol + len(pv.columns) + 2]


def concat_clean(df1, df2):
    df = pd.concat([df1, df2], axis=1, sort=True)
    df.sort_index(axis=1, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.reindex([x for x in df.index if x != 'not known']+['not known'])


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
    df[bucket_col] = np.ceil(df[col_to_bucket]/interval)*interval
    # we aggregate by average for numeric variables and baseline value for categorical variables
    keys = static_data_col + numeric_col
    values = ['first']*len(static_data_col) + ['mean']*len(numeric_col)
    grouping_dict = dict(zip(keys, values))
    
    df_grouped = df.groupby(['brcid']+[bucket_col], as_index=False).agg(grouping_dict)
    df_baseline = df_grouped.sort_values(['brcid', 'age_at_score']).groupby('brcid').first()
    df_grouped = df_grouped.merge(df_baseline, on='brcid', suffixes=('', '_baseline'))
    df_grouped = df_grouped.sort_values(['brcid', 'age_at_score'])
    
    df_grouped['occur'] = df_grouped.groupby('brcid')['brcid'].transform('size')
    df_grouped = df_grouped[(df_grouped['occur'] >= min_obs)]
    # df_grouped['counter'] = df.groupby('brcid').cumcount() + 1
    all_buckets = pd.DataFrame(data=np.arange(start=bucket_min, stop=bucket_max, step=interval), columns=[bucket_col])
    all_buckets['counter'] = np.arange(start=1, stop=len(all_buckets)+1, step=1)
    df_grouped = df_grouped.merge(all_buckets, on=bucket_col).sort_values(['brcid', bucket_col])
    
    return df_grouped.reset_index(drop=True)


def read_and_clean_data(file_path, baseline_cols):
    df = pd.read_csv(file_path, header=0, low_memory=False)
    df.columns = df.columns.str.lower()
    # df['age_at_score_upper_bound']=np.ceil(df['age_at_score']/interval)*interval
    static_data_col= [col for col in df.select_dtypes(include=['object']).columns if ('date' not in col) and ('age' not in col) and ('score_bucket') not in col]
    df[static_data_col] = df[static_data_col].apply(lambda x: x.astype(str).str.lower())
    df.loc[df['first_language'].str.contains('other', case=False, na=False), 'first_language'] = 'other language'
    df.loc[df['ethnicity'].str.contains('other', case=False, na=False), 'ethnicity'] = 'other ethnicity'
    df[static_data_col] = df[static_data_col].replace(['null', 'unknown', np.nan, 'nan', 'other', 'not specified', 'not disclosed', 'not stated (z)'], 'not known')
    df.replace({'patient_diagnosis_class': {'smi+organic': 'schizo+bipolar+organic'}}, inplace=True)
    df['ethnicity_group'] = 'other ethnicity'
    df.loc[df['ethnicity'].str.contains('not known', case=False, na=False), 'ethnicity_group'] = 'not known'
    df.loc[df['ethnicity'].str.contains('|'.join(['and', 'mixed']), case=False, na=False), ['ethnicity', 'ethnicity_group']] = 'mixed'
    df.loc[df['ethnicity'].str.contains('african', case=False, na=False), 'ethnicity_group'] = 'black african'
    df.loc[df['ethnicity'].str.contains('caribbean', case=False, na=False), 'ethnicity_group'] = 'black caribbean'
    df.loc[df['ethnicity'].str.contains('|'.join(['irish', 'british', 'other white']), case=False, na=False), 'ethnicity_group'] = 'white'
    df.loc[df['ethnicity'].str.contains('|'.join(['bangladesh', 'pakistan', 'india']), case=False, na=False), 'ethnicity_group'] = 'indian'
    df.loc[df['marital_status'].str.contains('|'.join(['married', 'cohabit']), case=False, na=False), 'marital_status'] = 'married or cohabiting'
    df.loc[df['marital_status'].str.contains('|'.join(['divorce', 'widow', 'separat', 'single']), case=False, na=False), 'marital_status'] = 'single or separated'
    df.loc[~df['first_language'].str.contains('|'.join(['english', 'known']), case=False, na=False), 'first_language'] = 'not english'
    df.loc[df['occupation'].str.contains('employ', case=False, na=False) &
           ~df['occupation'].str.contains('unemploy', case=False, na=False), 'occupation'] = 'employed'
    df.loc[df['occupation'].str.contains('student', case=False, na=False), 'occupation'] = 'student'
    df.loc[df['living_status'].str.contains('|'.join(['nurs', 'trust']), case=False, na=False), 'living_status'] = 'nursing/residential/trust'
    df['bmi_bucket'] = gutils.bmi_category(df.bmi_score)
    df['bp_bucket'] = gutils.blood_pressure(df.systolic_value, df.diastolic_value)
    df['diabetes_bucket'] = gutils.diabetes(df.plasma_glucose_value)
    
    df['score_time_period'] = pd.PeriodIndex(pd.to_datetime(df.score_date), freq='Q').astype(str)
    df.score_time_period.replace({'Q1': 'H1', 'Q2': 'H1', 'Q3': 'H2', 'Q4': 'H2'}, regex=True, inplace=True)

    df_baseline = df.sort_values(['brcid', 'age_at_score']).groupby('brcid').first().reset_index()
    df_all = df.merge(df_baseline[baseline_cols], on='brcid', suffixes=('', '_baseline'))
    df_all = df_all.sort_values(['brcid', 'age_at_score'])
    return [df_all, df_baseline]


##############################################################################################
# POPULATION STATISTICS
##############################################################################################

def write_report(output_file_path, df_all, df_baseline, cols_to_pivot, index_to_pivot, score_funcs, health_numeric_cols):
    index_to_pivot_baseline = [col for col in df_all.columns if 'bucket_baseline' in col or 'smoking_status' in col]
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    writer = pd.ExcelWriter(output_file_path.replace('.xlsx', '_'+st+'.xlsx'), engine='xlsxwriter')
    cpt_row = 0
    pv_master = pd.DataFrame()
    for var in index_to_pivot+index_to_pivot_baseline:
        pv_scores = my_pivot(df_all, values='score_combined', index=var, cols_to_pivot=cols_to_pivot, aggfunc=score_funcs)
        pv_pop = my_pivot(df_all, values='brcid', index=var, cols_to_pivot=cols_to_pivot, aggfunc=pd.Series.nunique)
        pv_pop.columns = [x+'_brcid' for x in pv_pop.columns]
        pv = concat_clean(pv_scores, pv_pop)
        pv_master = pd.concat([pv_master, pv], axis=0, sort=False)
        pv.to_excel(writer, sheet_name='summary_separate', startrow=cpt_row)
        cpt_row += len(pv_scores)+4
    
    # format header
    header = pd.DataFrame([[i.split('_', 1)[1] for i in pv_master.columns]],
                          columns=[i.split('_', 1)[0] for i in pv_master.columns])
    header.to_excel(writer, sheet_name='summary', startrow=0)
    pv_master.to_excel(writer, sheet_name='summary', startrow=len(header)+1, header=False)
    
    pv_baseline = df_baseline.pivot_table(values='brcid', index='age_bucket_at_score2', columns=cols_to_pivot, aggfunc=pd.Series.nunique, margins=True).fillna(0)
    pv_baseline.to_excel(writer, sheet_name='first_measure', startrow=0)
    
    health_stats0 = df_baseline.groupby(cols_to_pivot[0])[health_numeric_cols].agg(score_funcs)
    health_stats1 = df_baseline.groupby(cols_to_pivot[1])[health_numeric_cols].agg(score_funcs)
    health_stats = pd.concat([health_stats0, health_stats1], axis=0, sort=True)
    health_stats.to_excel(writer, sheet_name='health_stats', startrow=0)
    writer.save()
    
    return 0


##############################################################################################
# LONGITUDINAL MODELLING
##############################################################################################


def model(file_path,
          regressors,
          to_predict='score_combined',
          intercept_col='score_combined_baseline',
          col_to_bucket='age_at_score', bucket_min=50, bucket_max=90, interval=0.5, min_obs=3, na_values=None,
          ):
    df = read_and_clean_data(file_path)[0]
    df = prep_data_for_model(df, regressors=regressors, to_predict=to_predict, na_values=na_values, col_to_bucket=col_to_bucket,
                             bucket_min=bucket_min, bucket_max=bucket_max, interval=interval, min_obs=min_obs)

    df['age_bucket_complete'] = 0  # TODO? fill with NAN if no data for specific bucket?
    df['intercept'] = df[intercept_col]  # take score at baseline
    
    temporal_data_col = [col for col in df._get_numeric_data().columns if 'num_obs' not in col]
    static_data_col = [col for col in df.select_dtypes(include=['object', 'category']).columns if 'date' not in col]
    cols_for_stats = static_data_col + temporal_data_col
    df_agg_patient_class = df[cols_for_stats].groupby('patient_diagnosis_class').agg([len, np.size, np.mean, np.std, np.min, np.max, stats.mode])
    df_agg_patient_super_class = df[cols_for_stats].groupby('patient_diagnosis_super_class').agg([len, np.size, np.mean, np.std, np.min, np.max, stats.mode])
    df_agg_patient_age = df[cols_for_stats].groupby(['patient_diagnosis_super_class', 'age_at_mmse_upper_bound']).agg([len, np.size, np.mean, np.std, np.min, np.max, stats.mode])
    df[cols_for_stats].describe()
    
    df_stats = df[cols_for_stats].describe(include='all')

    # TRANSFORM CATEGORICAL VARIABLES TO DUMMY
    # https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512
    # from sklearn.preprocessing import OneHotEncoder
    # enc = OneHotEncoder(handle_unknown='ignore')
    # enc.fit(X_train_ordinal)
    
    test = pd.get_dummies(df, prefix=['col1', 'col2'])
    
    model = smf.mixedlm("score_combined ~ age_at_score_upper_bound + gender",
                        df,
                        groups=df["patient_diagnosis_super_class"])
    results = model.fit()
    print(results.summary())
    
    df['intercept'] = 1
    
    model = mlm.MixedLM(endog=df['score_combined'],  # dependent variable (1D))
                        exog=df[['age_at_score_upper_bound', 'intercept']],  # fixed effect covariates (2D)
                        exog_re=df['intercept'],  # random effect covariates
                        groups=df['patient_diagnosis_super_class'])  # data from different groups are independent
    result = model.fit()
    print(result.summary())
