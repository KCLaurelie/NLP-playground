from code_utils.global_variables import *
from longitudinal_models.longitudinal_models import *
from longitudinal_models.imputation import *
from longitudinal_models.data_stats import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm

df = pd.read_excel(r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\mmse_trajectory_data_20200407_synthetic.xlsx',
                   index_col=None, sheet_name='combined')

df = df.loc[(df.patient_diagnosis_super_class != 'organic only')]
#df = df.loc[df.duration !=False ,['brcid','duration','patient_diagnosis_super_class']]
#res=ttest_ind(df, value='score_combined', group_col='patient_diagnosis_super_class')
#df = df.loc[(df.patient_diagnosis_super_class != 'SMI only') & (df.age_at_dementia_diag >= 50)]

def test(df, impute=False, bucket_data=False, calc_stats=False, complete=False):

    if bucket_data:
        df = gutils.bucket_data(df, to_bucket='age_at_score', key='brcid', bucket_min=50, bucket_max=90, interval=0.5
                                       , cols_to_exclude=None, na_values='unknown', min_obs=3, timestamp_cols=['age_at_score', 'score_date'])
    if impute:  # if imputed data needs to be generated (this step takes a while)
        df = impute_with_baseline(df, key='brcid', baseline_cols=['brcid', 'score_date'],
                                      output_column=['gender', 'ethnicity', 'first_language', 'marital_status', 'education', 'imd_bucket_baseline', 'smoking_status_baseline', 'cvd_problem_baseline'],
                                      input_columns_to_exclude=['brcid', 'age_at_dementia_diag', 'occur', 'counter', 'score_date_centered', 'score_date', 'age_at_score_centered', 'age_at_score_baseline'])
    if calc_stats:  # NEED TO DO ON RAW DATA
        stats0 = data_stats(df, group_col='patient_diagnosis_super_class')
        stats1 = ttest(df, group_col='patient_diagnosis_super_class', groups_to_study=['organic only', 'SMI+organic'])
        stats2 = chisq(df, group_col='patient_diagnosis_super_class', groups_to_study=['organic only', 'SMI+organic'])
    if complete:
        df = df.replace({'unknown': np.nan, 'not known': np.nan}).dropna()

    socio_dem_imp = ['gender', 'ethnicity_imputed', 'marital_status_imputed', 'education',
                     'first_language_imputed', 'imd_bucket_baseline_imputed'] #, 'age_at_dementia_diag'
    cvd_imp = ['smoking_status_baseline_imputed', 'cvd_problem_baseline_imputed']
    med = ['dementia_medication_baseline', 'antipsychotic_medication_baseline', 'antidepressant_medication_baseline']
    references = {'education': 'no', 'ethnicity_imputed': 'white', 'ethnicity': 'white', 'smoking_status_baseline_imputed': 'no',
                  'smoking_status_baseline': 'no', 'marital_status_imputed': 'married_cohabitating',
                  'marital_status': 'married_cohabitating', 'patient_diagnosis_class':'organic only'}
    socio_dem = [x.replace('_imputed', '') for x in socio_dem_imp]
    cvd = [x.replace('_imputed', '') for x in cvd_imp]

    for key, val in references.items():
        if key in df: df[key] = df[key].replace({val: 'aaa_' + val})

    covariates = ['patient_diagnosis_class', 'age_at_score_baseline']
    res0 = run_models(timestamps=('score_date_centered',), model_data=df, models=('linear_rdn_all',), covariates=covariates, covariates_slope=True, patients_split_col=None)
    covariates = ['patient_diagnosis_class', 'age_at_score_baseline'] + socio_dem
    res1 = run_models(timestamps=('score_date_centered',), model_data=df, models=('linear_rdn_all',), covariates=covariates, covariates_slope=True, patients_split_col=None)
    covariates = ['patient_diagnosis_class', 'age_at_score_baseline'] + socio_dem + ['smoking_status_baseline']
    res2 = run_models(timestamps=('score_date_centered',), model_data=df, models=('linear_rdn_all',), covariates=covariates, covariates_slope=True, patients_split_col=None)
    covariates = ['patient_diagnosis_class', 'age_at_score_baseline'] + socio_dem + cvd
    res3 = run_models(timestamps=('score_date_centered',), model_data=df, models=('linear_rdn_all',), covariates=covariates, covariates_slope=True, patients_split_col=None)
    covariates = ['patient_diagnosis_class', 'age_at_score_baseline'] + socio_dem + cvd + med
    res4 = run_models(timestamps=('score_date_centered',), model_data=df, models=('linear_rdn_all',), covariates=covariates, covariates_slope=True, patients_split_col=None)
    covariates = ['patient_diagnosis_class', 'age_at_score_baseline'] + socio_dem + cvd + ['dementia_medication_baseline']
    res5 = run_models(timestamps=('score_date_centered',), model_data=df, models=('linear_rdn_all',), covariates=covariates, covariates_slope=True, patients_split_col=None)
    covariates = ['patient_diagnosis_class', 'age_at_score_baseline'] + socio_dem + cvd + ['antipsychotic_medication_baseline']
    res6 = run_models(timestamps=('score_date_centered',), model_data=df, models=('linear_rdn_all',), covariates=covariates, covariates_slope=True, patients_split_col=None)
    covariates = ['patient_diagnosis_class', 'age_at_score_baseline'] + socio_dem + cvd + ['antidepressant_medication_baseline']
    res7 = run_models(timestamps=('score_date_centered',), model_data=df, models=('linear_rdn_all',), covariates=covariates, covariates_slope=True, patients_split_col=None)

    # models = ('linear_rdn_all', 'linear_rdn_int', 'quadratic_rdn_int')
    # covariates_slope = True
    # patients_split_col = 'patient_diagnosis_super_class'
    # if patients_split_col is None: socio_dem_imp.insert(1, 'patient_diagnosis_super_class')
    # res = run_models(timestamps=('score_date_centered',), model_data=df, models=('linear_rdn_all',),
    #                  covariates='patient_diagnosis_super_class', covariates_slope=covariates_slope,
    #                  patients_split_col=patients_split_col)


def run_report(dataset=ds.default_dataset):
    dataset.file_path = r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_trajectory_data_final6.csv'
    dataset.cols_to_pivot = ['patient_diagnosis_super_class']
    dataset.write_report(r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_report_1class.xlsx')
    dataset.cols_to_pivot = ['patient_diagnosis_super_class', 'patient_diagnosis_class']
    dataset.write_report(r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_report_2classes.xlsx')

def playground2():
    df = pd.read_excel(r'C:\Users\K1774755\PycharmProjects\prometheus\longitudinal_modelling\trajectories_synthetic.xlsx',sheet_name='data')
    # ['brcid', 'diagnosis', 'date', 'score', 'gender', 'med']
    r_formula = 'score_combined ~  score_date_centered + age_at_score_baseline + patient_diagnosis_super_class + score_date_centered * age_at_score_baseline + score_date_centered * patient_diagnosis_super_class'

    model = Lmer('score ~ date  + (1|brcid)', data=df)
    # random slope and intercept
    model = smf.mixedlm(r_formula, df, groups=df['brcid'], re_formula="~score")
    model = sm.MixedLM.from_formula(r_formula, df, re_formula="score ", groups=df['brcid'])
    result = model.fit()
    print(result.summary())

def model_playground():
    df = pd.read_excel(r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\mmse_synthetic_data_20190919.xlsx',
                       index_col=None)
    df_smi = df[df.patient_diagnosis_super_class == 'smi only']
    df_orga = df[df.patient_diagnosis_super_class == 'organic only']
    df_smi_orga = df[df.patient_diagnosis_super_class == 'smi+organic']
    df_to_use = df_orga

    # MODEL 1: basic model (random intercept and fixed slope)
    model = Lmer('score_combined ~ score_date_centered  + (1|brcid)', data=df_to_use)  # MMSE score by year
    model = Lmer('score_combined ~ score_date_centered  + gender + (1|brcid)',
                 data=df_to_use)  # adding age at baseline as covariate (is this correct??)
    to_print = print_r_model_output(model.fit())
    # MODEL 2: random intercept and random slope
    model = Lmer('score_combined ~  (score_date_centered  | brcid)', data=df_to_use)  # this removes the intercept?
    model = Lmer('score_combined ~  (score_date_centered  + gender| brcid)', data=df_to_use)

    model = Lmer("score_combined ~ score_date_centered + (1 + score_date_centered | brcid)",
                 data=df_to_use)  # correct one?
    model = Lmer('score_combined ~  score_date_centered + gender + (1 + score_date_centered | brcid)', data=df_to_use)

    model = Lmer('score_combined ~  1 + score_date_centered  + (1|brcid) + (0 + score_date_centered  | brcid)',
                 data=df)  # 2 random effects constrained to be uncorrelated

    # MODEL 3: basic model but quadratic
    model = Lmer('score_combined ~ score_date_centered  + I(score_date_centered ^2) + (1|brcid)', data=df_to_use)

    print(model.fit())

    #######################################################################################
    #  PYTHON STUFF
    # R formula:
    r_formula = 'score_combined ~  score_date_centered + age_at_score_baseline + patient_diagnosis_super_class + score_date_centered * age_at_score_baseline + score_date_centered * patient_diagnosis_super_class'

    # MODEL 1: python equivalent
    model_py = smf.mixedlm("score_combined ~ score_date_centered ", df_to_use, groups=df_to_use['brcid'])
    result = model_py.fit()
    print(model_py.fit().summary())

    # random slope and intercept
    model_py = smf.mixedlm(r_formula, df_to_use, groups=df_to_use['brcid'], re_formula="~score_date_centered")
    model_py = sm.MixedLM.from_formula(r_formula
                                       , df_to_use
                                       , re_formula="score_date_centered "
                                       , groups=df_to_use['brcid'])
    # random slope only
    model_py = sm.MixedLM.from_formula("score_combined ~ score_date_centered "
                                       , df_to_use
                                       , re_formula="0 + score_date_centered "
                                       , groups=df_to_use['brcid'])

    # MODEL 2: python equivalent (??)
    vcf = {"score_date_centered ": "0 + C(score_date_centered )", "brcid": "0 + C(brcid)"}
    model_py = sm.MixedLM.from_formula("score_combined ~ score_date_centered ", groups=df_to_use['brcid'],
                                       vc_formula=vcf, re_formula="0", data=df_to_use)
    print(model_py.fit().summary())

    model3 = mlm.MixedLM(endog=df_to_use['score_combined'],  # dependent variable (1D))
                         exog=df_to_use[['score_date_centered ', 'intercept']],  # fixed effect covariates (2D)
                         exog_re=df_to_use['intercept'],  # random effect covariates
                         groups=df_to_use['brcid'])  # data from different groups are independent
    result = model3.fit()
    print(result.summary())
