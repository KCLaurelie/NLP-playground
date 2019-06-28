# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:24:18 2019

@author: AMascio
"""

import os
try: os.chdir(r'T:\aurelie_mascio\python_scripts') #directory with python library
except: pass
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm
import longitudinal_models.general_utils as gutils
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time


root_path = r'C:\Users\K1774755\Downloads'
file_path = os.path.join(root_path, 'mmse_trajectory_synthetic.csv')
#file_path = os.path.join(root_path, 'honos_trajectory_data3.csv')
#aggfuncs=['count', pd.Series.nunique]
score_funcs = ['count', np.mean, np.std]

##############################################################################################
## MAIN
##############################################################################################
cols_to_pivot = ['patient_diagnosis_super_class', 'patient_diagnosis_class']
index_to_pivot = ['age_bucket_at_score2', 'gender', 'ethnicity_group', 'first_language', 'occupation', 'living_status', 'marital_status', 'education_bucket_raw' ]
health_numeric_data = ['bmi_score', 'plasma_glucose_value', 'diastolic_value', 'systolic_value']

def create_report(file_path):
    metric = 'honos' if 'honos' in file_path else 'mmse'
    output_file_path = os.path.join(root_path, metric+'_trajectory_report.xlsx')
    df_all, df_baseline = read_and_clean_data(file_path)
    write_report(output_file_path, df_all, df_baseline)

def run_model(file_path):
    df = read_and_clean_data(file_path)[0]
    df_grouped = group_data(df)
    df_test = df[df.brcid == 10000028][['age_at_score','score_combined']]
    df_grouped_test = df_grouped[df_grouped.brcid == 10000028][['age_at_score','score_combined']]
    return 0

##############################################################################################
## UTIL FUNCTIONS
##############################################################################################
def my_pivot(df,
             values='score_combined',
             index = 'gender',
             columns = ['patient_diagnosis_super_class', 'patient_diagnosis_class'],
             aggfunc = score_funcs):
    pv0 = df.pivot_table(values=values, index=index, columns=cols_to_pivot[0], aggfunc=aggfunc, margins=True).fillna(0)
    try: # to avoid duplicates between super class and class: for multi-index
        pv0.drop(['organic only', 'All'], level=0, axis=1,inplace=True) # if only 1 agg function
        pv0.drop(['organic only', 'All'], level=1, axis=1,inplace=True) # if more than 1 agg function
    except: # to avoid duplicates between super class and class: for single index
        pv0.drop(columns=['All'],inplace=True)
    pv1 = df.pivot_table(values=values, index=index, columns=cols_to_pivot[1], aggfunc=aggfunc, margins=True).fillna(0)
    pv = pd.concat([pv1, pv0], axis = 1, sort = True)
    if isinstance(aggfunc, list):
        pv = pv.swaplevel(axis=1)
        pv.columns = [ '_'.join(x) for x in pv.columns ]
    # sort by column name
    pv.sort_index(axis=1, inplace = True)
    pv.rename(index={'All':'All_'+index},inplace=True)
    pv = pv.loc[:,~pv.columns.duplicated()]
    return pv.reindex([x for x in pv.index if x !='not known']+['not known'])

def print_pv_to_excel(pv, writer, sheet_name, startrow=0, startcol=0):
    pv.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol)
    return [startrow + len(pv) + 2, startcol + len(pv.columns) +2]

def concat_clean(df1, df2):
        df = pd.concat([df1, df2], axis = 1, sort = True)
        df.sort_index(axis=1, inplace = True)
        df = df.loc[:,~df.columns.duplicated()]
        return df.reindex([x for x in df.index if x !='not known']+['not known'])

##############################################################################################
## READING/CLEANING/ENRICHING THE DATA
##############################################################################################
def group_data(df,
               cols_to_keep=['brcid', 'patient_diagnosis_class', 'patient_diagnosis_super_class','score_date', 'age_at_score', 'score_combined',
                             'gender', 'ethnicity_group', 'first_language', 'occupation', 'living_status', 'marital_status',
                             'is_active', 'education_bucket_raw','has_depression_anxiety_diagnosis', 'has_agitation_diagnosis', 'smoking_status',  'aggression_status',
                             'plasma_glucose_value','diabetes_bucket', 'diastolic_value', 'systolic_value',  'bp_bucket', 'bmi_score', 'bmi_bucket'],
                cat_cols = ['has_agitation_diagnosis','has_depression_anxiety_diagnosis','is_active'],
                col_to_bucket='age_at_score',
                bucket_min=50,
                bucket_max=80,
                interval=0.5,
                min_obs=3):
    # only use data within bucket boundaries
    bucket_col = col_to_bucket + '_upper_bound'
    df = df[(df[col_to_bucket]>=age_min)&(df[col_to_bucket]<=age_max)][cols_to_keep]
    if cat_cols is not None: df[cat_cols] = df[cat_cols].replace({0:'no',1:'yes'})
    static_data_col= [col for col in df.select_dtypes(include=['object']).columns if ('brcid' not in col)]
    numeric_col = [col for col in df._get_numeric_data().columns if ('brcid' not in col)]
    df[bucket_col]=np.ceil(df[col_to_bucket]/interval)*interval
    keys = static_data_col + numeric_col
    values = ['first']*len(static_data_col) + ['mean']*len(numeric_col)
    grouping_dict = dict(zip(keys,values))
    
    # TODO: fill missing values otherwise get excluded from groupby
    
    df_grouped = df.groupby(['brcid']+[bucket_col],as_index=False).agg(grouping_dict)
    df_baseline = df_grouped.sort_values(['brcid','age_at_score']).groupby('brcid').first().reset_index()   
    df_grouped = df_grouped.merge(df_baseline, on='brcid', suffixes=('', '_baseline'))
    df_grouped = df_grouped.sort_values(['brcid','age_at_score'])
    
    df_grouped['occur'] = df_grouped.groupby('brcid')['brcid'].transform('size')
    df_grouped = df_grouped[(df_grouped['occur']>=min_obs)]
    #df_grouped['counter'] = df.groupby('brcid').cumcount() + 1
    ages = pd.DataFrame(data=np.arange(start=age_min, stop=age_max, step=interval), columns=[bucket_col])
    ages['counter']=np.arange(start=1, stop=len(ages)+1, step=1)
    df_grouped = df_grouped.merge(ages, on=bucket_col).sort_values(['brcid', bucket_col])
    
    return df_grouped.reset_index()

def read_and_clean_data(file_path,
                        baseline_cols = ['brcid', 'age_at_score', 'score_combined',
                                         'bmi_score', 'plasma_glucose_value', 'diastolic_value', 'systolic_value', 'smoking_status',
                                         'bmi_bucket', 'diabetes_bucket', 'bp_bucket']
                        ):
    df = pd.read_csv(file_path, header=0,low_memory=False)
    df.columns = df.columns.str.lower()
    #df['age_at_score_upper_bound']=np.ceil(df['age_at_score']/interval)*interval
    static_data_col= [col for col in df.select_dtypes(include=['object']).columns if ('date' not in col) and ('age' not in col) and ('score_bucket') not in col]
    df[static_data_col] = df[static_data_col].apply(lambda x: x.astype(str).str.lower())
    df.loc[df['first_language'].str.contains('other', case=False, na=False), 'first_language'] = 'other language'
    df.loc[df['ethnicity'].str.contains('other', case=False, na=False), 'ethnicity'] = 'other ethnicity'
    df[static_data_col] = df[static_data_col].replace(['null', 'unknown', np.nan, 'nan', 'other', 'not specified', 'not disclosed', 'not stated (z)'], 'not known')
    df.replace({'patient_diagnosis_class' : { 'smi+organic' : 'schizo+bipolar+organic'}}, inplace = True)
    df['ethnicity_group'] = 'other ethnicity'
    df.loc[df['ethnicity'].str.contains('not known', case=False, na=False), 'ethnicity_group'] = 'not known'
    df.loc[df['ethnicity'].str.contains('|'.join(['and','mixed']), case=False, na=False), ['ethnicity', 'ethnicity_group']] = 'mixed'
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
    df.score_time_period.replace({'Q1':'H1', 'Q2':'H1', 'Q3':'H2', 'Q4':'H2'}, regex=True, inplace=True)

    df_baseline = df.sort_values(['brcid','age_at_score']).groupby('brcid').first().reset_index()   
    df_all = df.merge(df_baseline[baseline_cols], on='brcid', suffixes=('', '_baseline'))
    df_all = df_all.sort_values(['brcid','age_at_score'])
    return [df_all, df_baseline]


##############################################################################################
## POPULATION STATISTICS
##############################################################################################

def write_report(output_file_path, df_all, df_baseline):
    index_to_pivot_baseline = [col for col in df_all.columns if 'bucket_baseline' in col or 'smoking_status' in col]
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    writer = pd.ExcelWriter(output_file_path.replace('.xlsx', '_'+st+'.xlsx'), engine='xlsxwriter')
    cpt_row = 0
    pv_master=pd.DataFrame()
    for var in index_to_pivot+index_to_pivot_baseline:
        pv_scores =  my_pivot(df_all, values='score_combined', index=var, columns=cols_to_pivot, aggfunc=score_funcs)
        pv_pop =  my_pivot(df_all, values='brcid', index=var, columns=cols_to_pivot, aggfunc=pd.Series.nunique)
        pv_pop.columns = [x+'_brcid' for x in pv_pop.columns]
        pv = concat_clean(pv_scores, pv_pop)
        pv_master = pd.concat([pv_master, pv], axis=0, sort=False)
        pv.to_excel(writer, sheet_name='summary_separate', startrow=cpt_row)
        cpt_row+=len(pv_scores)+4
    
    # format header
    header = pd.DataFrame([[i.split('_', 1)[1] for i in pv_master.columns]],
                          columns = [i.split('_', 1)[0] for i in pv_master.columns])
    header.to_excel(writer, sheet_name='summary', startrow=0)
    pv_master.to_excel(writer, sheet_name='summary', startrow=len(header)+1, header=False)
    
    pv_baseline = df_baseline.pivot_table(values='brcid', index='age_bucket_at_score2', columns=cols_to_pivot, aggfunc=pd.Series.nunique, margins=True).fillna(0)
    pv_baseline.to_excel(writer, sheet_name='first_measure', startrow=0)
    
    health_stats0 = df_baseline.groupby(cols_to_pivot[0])[health_numeric_data].agg(score_funcs)
    health_stats1 = df_baseline.groupby(cols_to_pivot[1])[health_numeric_data].agg(score_funcs)
    health_stats = pd.concat([health_stats0, health_stats1], axis = 0, sort = True)
    health_stats.to_excel(writer, sheet_name= 'health_stats', startrow=0)
    writer.save()
    
    return 0



##############################################################################################
## MISSING DATA ANALYSIS
##############################################################################################

def plot_stuff(df_all, df_baseline):
    df_all = df_all.replace(['null', 'unknown', np.nan, 'other', 'not disclosed'], None)
    df_baseline = df_baseline.replace(['null', 'unknown', np.nan, 'other', 'not disclosed'], None)
    
    #http://stronginference.com/missing-data-imputation.html
    df_baseline.info()
    df_stats = df_baseline.describe(include='all')
    
    # correlations
    colormap = plt.cm.RdBu
    plt.figure(figsize=(32,10))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df_baseline.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
    
    # heatmap of missing values
    index_to_pivot_baseline = [col for col in df_all.columns if 'bucket_baseline' in col or 'smoking_status_baseline' in col]
    to_plot = [x for x in index_to_pivot+index_to_pivot_baseline if x in df_baseline.columns]
    sns.heatmap(df_baseline[to_plot].isnull(), cbar=False)
    masked_values = np.ma.masked_array(df_baseline['education_bucket_raw'], mask=df_baseline['education_bucket_raw'] is None)

##############################################################################################
## LONGITUDINAL MODELLING
##############################################################################################
predicted_variable, intercept, na_values, age_min, age_max, interval = ['score_combined', 'score_combined_baseline',None,50,90,0.5]
def model(file_path,
          predicted_variable='score_combined',
          intercept_col='score_combined_baseline',
          na_values=None,
          age_min=50,
          age_max=90,
          interval=0.5,
          static_data_col=['brcid', 'gender', 'ethnicity_group', 'first_language', 'occupation', 'living_status', 'marital_status', 'education_bucket_raw']
          ):
    df_all = read_and_clean_data(file_path)[0]
    static_data_col = static_data_col + [col for col in df_all.columns if 'baseline' in col]
    group = ['brcid','age_at_score_upper_bound']
    mask = (df_all.age_at_score_upper_bound>=age_min)&(df_all.age_at_score_upper_bound<=age_max)&(df_all.patient_diagnosis_super_class == 'organic only')
    df = df_all[mask].groupby(group,as_index=False).agg({predicted_variable:'mean'})
    df = df.sort_values(['brcid','age_at_score_upper_bound'])
    if na_values is not None: df.fillna(na_values, inplace=True) # if we want to keep datapoints with missing data
    df['occur'] = df.groupby('brcid')['brcid'].transform('size')
    df = df[(df['occur']>=3)]
    #df['counter'] = df.groupby('brcid').cumcount() + 1
    
    ## TODO
    ages = pd.DataFrame(data=np.arange(start=age_min, stop=age_max, step=interval), columns=['age_at_score_upper_bound'])
    ages['counter']=np.arange(start=1, stop=len(ages)+1, step=1)
    df = df.merge(ages, on='age_at_score_upper_bound')
    df['age_bucket_complete']= 0 # fill with NAN if no data for specific bucket
    df['intercept'] = df.intercept_col # take score at baseline
    
    temporal_data_col = [col for col in df_all._get_numeric_data().columns if 'num_obs' not in col]
    static_data_col= [col for col in df_all.select_dtypes(include=['object']).columns if 'date' not in col]
    cols_for_stats= static_data_col + temporal_data_col
    df_agg_patient_class=df_all[cols_for_stats].groupby('patient_diagnosis_class').agg([len, np.size, np.mean, np.std, np.min, np.max, stats.mode])
    df_agg_patient_super_class=df_all[cols_for_stats].groupby('patient_diagnosis_super_class').agg([len, np.size, np.mean, np.std, np.min, np.max, stats.mode])
    df_agg_patient_age=df_all[cols_for_stats].groupby(['patient_diagnosis_super_class','age_at_mmse_upper_bound']).agg([len, np.size, np.mean, np.std, np.min, np.max, stats.mode])
    df_all[cols_for_stats].describe()
    
    df_stats=df_all[cols_for_stats].describe(include='all')
    
    
    # TRANSFORM CATEGORICAL VARIABLES TO DUMMY
    # https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512
    #from sklearn.preprocessing import OneHotEncoder
    #enc = OneHotEncoder(handle_unknown='ignore')
    #enc.fit(X_train_ordinal)
    
    test= pd.get_dummies(df_all, prefix=['col1', 'col2'])
    
    model = smf.mixedlm("score_combined ~ age_at_score_upper_bound + gender",
                        df_all,
                        groups=df_all["patient_diagnosis_super_class"])
    results = model.fit()
    print(results.summary())
    
    df_all['intercept']=1
    
    model = mlm.MixedLM(endog=df_all['score_combined'], # dependent variable (1D))
                     exog=df_all[['age_at_score_upper_bound','intercept']], # fixed effect covariates (2D)
                     exog_re=df_all['intercept'], #random effect covariates
                     groups=df_all['patient_diagnosis_super_class']) # data from different groups are independent
    result = model.fit()
    print(result.summary())
