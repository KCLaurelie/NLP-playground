from code_utils.global_variables import *
import pandas as pd
import numpy as np
import longitudinal_models.longitudinal_dataset as ds
from pymer4.models import Lm, Lmer

# for python models
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm



# TODO: analysis for each super diagnosis class subgroup
# TODO plots for 200 sample in each subgroup
# TODO compare age at baseline vs age a diagnosis
# TODO redo pivot tables with population used
# https://rpsychologist.com/r-guide-longitudinal-lme-lmer
# TODO add missing values
# TODO count patients per occurence (nb patients with 3, 4... scores)
# TODO: run model by age or by year? do by year? -> better to use by year and use age at baseline as covariate
# TODO: plot data https://stats.idre.ucla.edu/r/faq/how-can-i-visualize-longitudinal-data-in-ggplot2/

def check_r_loc():
    from rpy2.robjects.packages import importr
    base = importr('base')
    return base.R_home()


##############################################################################################
# LONGITUDINAL MODELLING
##############################################################################################
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


def run_report(dataset=ds.default_dataset):
    dataset.cols_to_pivot = ['patient_diagnosis_super_class']
    dataset.write_report(r'C:\Users\K1774755\Downloads\mmse_report_1class.xlsx')
    dataset.cols_to_pivot = ['patient_diagnosis_super_class', 'patient_diagnosis_class']
    dataset.write_report(r'C:\Users\K1774755\Downloads\mmse_report_2classes.xlsx')


def multi_level_r(dataset=ds.default_dataset, intercept='score_combined_baseline', timestamp='score_date_upbound'):
    df = dataset.regression_cleaning(normalize=False, dummyfy=False, keep_only_baseline=False)
    df['intercept'] = df[intercept]
    df['timestamp'] = df[timestamp]
    df_smi = df[df.patient_diagnosis_super_class == 'smi only']
    df_orga = df[df.patient_diagnosis_super_class == 'organic only']
    df_smi_orga = df[df.patient_diagnosis_super_class == 'smi+organic']

    # MODEL 1: basic model (random intercept and fixed slope)
    model = Lmer('score_combined ~ score_date_upbound + (1|brcid)', data=df)  # MMSE score by year
    model = Lmer('score_combined ~ score_date_upbound + (1|brcid)', data=df_smi)  # for subgroup
    model = Lmer('score_combined ~ score_date_upbound + age_at_score_baseline + (1|brcid)', data=df)  # adding age at baseline as covariate (is this correct??)

    # MODEL 2: random intercept and random slope
    model = Lmer('score_combined ~  (score_date_upbound | brcid)', data=df)  # fails to converge
    model = Lmer('score_combined ~  1 + score_date_upbound + (1|brcid) + (0 + score_date_upbound | brcid)', data=df)  # this converges but is it correct?
    model = Lmer('score_combined ~  (score_date_upbound + age_at_score_baseline| brcid)', data=df)

    # MODEL 3: basic model but quadratic
    model = Lmer('score_combined ~ score_date_upbound + I(score_date_upbound^2) + (1|brcid)', data=df)

    print(model.fit())

    # MODEL 1: python equivalent
    model_py = smf.mixedlm("score_combined ~ score_date_upbound", df, groups=df['brcid'])
    result = model_py.fit()
    print(result.summary())

    # random slope and intercept
    model = sm.MixedLM.from_formula("score_combined ~ score_date_upbound"
                                    , df
                                    , re_formula="score_date_upbound"
                                    , groups=df['brcid'])
    # random slope only
    model = sm.MixedLM.from_formula("score_combined ~ score_date_upbound"
                                    , df
                                    , re_formula="0 + score_date_upbound"
                                    , groups=df['brcid'])

    # MODEL 2: python equivalent (??)
    vcf = {"score_date_upbound": "0 + C(score_date_upbound)", "brcid": "0 + C(brcid)"}
    model_py = sm.MixedLM.from_formula("score_combined ~ score_date_upbound", groups=df['brcid'],
                                       vc_formula=vcf, re_formula="0", data=df)
    print(model_py.fit().summary())


    model3 = mlm.MixedLM(endog=df['score_combined'],  # dependent variable (1D))
                         exog=df[['score_date_upbound', 'intercept']],  # fixed effect covariates (2D)
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
