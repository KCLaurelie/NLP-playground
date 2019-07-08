import os

os.environ['R_HOME'] = r'C:\Program Files\R\R-3.6.0'  # where R is (needs to be run before importing pymer and rpy)
try:  # for use on CRIS computers
    os.chdir(r'T:\aurelie_mascio\python_scripts')
except:
    pass

import pandas as pd
import numpy as np
import longitudinal_models.general_utils as gutils
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm
import longitudinal_models.longitudinal_dataset as ds
from sklearn.preprocessing import MinMaxScaler
from pymer4.models import Lm, Lmer

"""
OPTION A)
- MODEL 1: Unconditional with MMSE= intercept (fixed)
- MODEL 2: Unconditional with MMSE= intercept (random)
- MODEL 3: Unconditional with MMSE= intercept + time (fixed)
- MODEL 4: Unconditional with MMSE= intercept + time (random)
- MODEL 5: Conditional with MMSE= intercept + time + grouping variable
Here grouping variable is Organic only/SMI only/SMI + O
OPTION B)
Run model 1- 4 in each group of the grouping variable. 3 outputs
"""


def check_r_loc():
    from rpy2.robjects.packages import importr
    base = importr('base')
    return base.R_home()


##############################################################################################
# LONGITUDINAL MODELLING
##############################################################################################
def pre_cleaning(dataset=ds.default_dataset, normalize=False, dummyfy=False, keep_only_baseline=False):
    if dataset.data is None:
        dataset.prep_data()
    df = dataset.data['data_grouped']
    # df_test = df[df.brcid == 9][['age_at_score', 'score_combined']]
    if normalize:
        numeric_cols = [col for col in df[dataset.regressors]._get_numeric_data().columns]
        cols_to_normalize = [dataset.to_predict] + [col for col in numeric_cols]
        scaler = MinMaxScaler()
        x = df[cols_to_normalize].values
        scaled_values = scaler.fit_transform(x)
        df[cols_to_normalize] = scaled_values
    if dummyfy:
        cols_to_dummyfy = df[dataset.regressors].select_dtypes(include=['object', 'category']).columns
        dummyfied_df = pd.get_dummies(df[cols_to_dummyfy])
        df = pd.concat([df.drop(columns=cols_to_dummyfy), dummyfied_df], axis=1)
    if keep_only_baseline:
        to_drop = [col for col in df.columns if ('_baseline' in col)
                   and col.replace('_baseline', '') not in gutils.to_list(dataset.to_predict)
                   and col.replace('_baseline', '') in df.columns]
        df.drop(columns=to_drop, inplace=True)
    return df


def multi_level_r(df, regressors, to_predict):
    from pymer4.utils import get_resource_path
    df = pd.read_csv(os.path.join(get_resource_path(), 'sample_data.csv'))
    model = Lm('DV ~ IV1 + IV3', data=df)
    model = Lmer('DV ~ IV2 + (IV2|Group)', data=df)
    model = Lmer('score_combined ~ age_at_score_upper_bound  + (score_combined_baseline|brcid)', data=df)
    result = model.fit()

    model2 = smf.mixedlm("score_combined ~ age_at_score_upper_bound + gender", df, groups=df['brcid'])
    result = model2.fit()
    print(result.summary())

    df['intercept'] = df['score_combined_baseline']
    model3 = mlm.MixedLM(endog=df['score_combined'],  # dependent variable (1D))
                         exog=df[['age_at_score_upper_bound', 'intercept']],  # fixed effect covariates (2D)
                         exog_re=df['intercept'],  # random effect covariates
                         groups=df['brcid'])  # data from different groups are independent
    result = model3.fit()
    print(result.summary())


def model(file_path,
          regressors,
          to_predict='score_combined',
          intercept_col='score_combined_baseline',
          col_to_bucket='age_at_score', bucket_min=50, bucket_max=90, interval=0.5, min_obs=3, na_values=None,
          ):
    df = ds.read_and_clean_data(file_path)[0]
    df = ds.prep_data_for_model(df, regressors=regressors, to_predict=to_predict, na_values=na_values,
                                col_to_bucket=col_to_bucket,
                                bucket_min=bucket_min, bucket_max=bucket_max, interval=interval, min_obs=min_obs)

    df['age_bucket_complete'] = 0  # TODO? fill with NAN if no data for specific bucket?
    df['intercept'] = df[intercept_col]  # take score at baseline

    temporal_data_col = [col for col in df._get_numeric_data().columns if 'num_obs' not in col]
    static_data_col = [col for col in df.select_dtypes(include=['object', 'category']).columns if 'date' not in col]
    cols_for_stats = static_data_col + temporal_data_col
    df_agg_patient_class = df[cols_for_stats].groupby('patient_diagnosis_class').agg(
        [len, np.size, np.mean, np.std, np.min, np.max, stats.mode])
    df_agg_patient_super_class = df[cols_for_stats].groupby('patient_diagnosis_super_class').agg(
        [len, np.size, np.mean, np.std, np.min, np.max, stats.mode])
    df_agg_patient_age = df[cols_for_stats].groupby(['patient_diagnosis_super_class', 'age_at_mmse_upper_bound']).agg(
        [len, np.size, np.mean, np.std, np.min, np.max, stats.mode])
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
