import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm
import longitudinal_models.longitudinal_data_prep as dataprep

try:
    os.chdir(r'T:\aurelie_mascio\python_scripts')  # for use on CRIS computers
except:
    pass


##############################################################################################
# LONGITUDINAL MODELLING
##############################################################################################
def run_model(file_path, baseline_cols, to_predict, regressors):
    df = dataprep.read_and_clean_data(file_path, baseline_cols)[0]
    df_grouped = dataprep.prep_data_for_model(df, to_predict=to_predict, regressors=regressors)
    df_test = df[df.brcid == 9][['age_at_score', 'score_combined']]
    df_grouped_test = df_grouped[df_grouped.brcid == 9][['age_at_score', 'score_combined']]


def model(file_path,
          regressors,
          to_predict='score_combined',
          intercept_col='score_combined_baseline',
          col_to_bucket='age_at_score', bucket_min=50, bucket_max=90, interval=0.5, min_obs=3, na_values=None,
          ):
    df = dataprep.read_and_clean_data(file_path)[0]
    df = dataprep.prep_data_for_model(df, regressors=regressors, to_predict=to_predict, na_values=na_values,
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
