from code_utils.global_variables import *
from longitudinal_models.longitudinal_models import *
from longitudinal_models.imputation import *
from longitudinal_models.data_stats import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm

df = pd.read_excel(r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\mmse_trajectory_data_20200407_synthetic.xlsx', index_col=None, sheet_name='combined')
df = df.loc[df.patient_diagnosis_super_class != 'SMI only']
#df = df.loc[df.include == 'yes']

def test(df, impute=False, bucket_data=False, calc_stats=False):

    if bucket_data:
        df_bucket = gutils.bucket_data(df, to_bucket='age_at_score', key='brcid', bucket_min=50, bucket_max=90, interval=0.5
                                       , cols_to_exclude=None, na_values='unknown', min_obs=3, timestamp_cols=['age_at_score', 'score_date'])
    if impute:  # if imputed data needs to be generated (this step takes a while)
        df_imp = impute_with_baseline(df_bucket, key='brcid', baseline_cols=['brcid', 'score_date'],
                                      output_column=['gender', 'ethnicity', 'first_language', 'marital_status', 'education', 'imd_bucket_baseline', 'smoking_status_baseline', 'cvd_problem_baseline'],
                                      input_columns_to_exclude=['brcid', 'age_at_dementia_diag', 'occur', 'counter', 'score_date_centered', 'score_date', 'age_at_score_centered', 'age_at_score_baseline'])
    if calc_stats:  # NEED TO DO ON RAW DATA
        stats0 = data_stats(df)
        stats1 = ttest(df, groups_to_study=['organic only', 'SMI+organic'])
        stats2 = chisq(df, groups_to_study=['organic only', 'SMI+organic'])

    socio_dem_imp = ['gender', 'ethnicity_imputed', 'marital_status_imputed',
                     'education', 'first_language_imputed', 'imd_bucket_baseline_imputed']
    cvd_imp = ['smoking_status_baseline_imputed', 'cvd_problem_baseline_imputed']
    med = ['dementia_medication_baseline', 'antipsychotic_medication_baseline','antidepressant_medication_baseline']
    references = {'education': 'no', 'ethnicity_imputed': 'white', 'ethnicity': 'white', 'smoking_status_baseline_imputed': 'no',
                  'smoking_status_baseline': 'no', 'marital_status_imputed': 'single_separated',
                  'marital_status': 'single_separated'}
    socio_dem = [x.replace('_imputed', '') for x in socio_dem_imp]
    cvd = [x.replace('_imputed', '') for x in cvd_imp]

    for key, val in references.items():
        if key in df: df[key] = df[key].replace({val: 'aaa_' + val})

    covariates = ['age_at_score_baseline', 'age_at_dementia_diag', 'patient_diagnosis_super_class'] + socio_dem_imp + cvd_imp + med
    res = run_models(timestamps=('score_date_centered',), model_data=df, models=('linear_rdn_all',), covariates=covariates, covariates_slope=True, patients_split_col=None)

    # models = ('linear_rdn_all', 'linear_rdn_int')  # , 'linear_rdn_all', 'quadratic_rdn_int')
    # ts = ('score_date_centered',)
    # covariates_slope = True
    # patients_split_col = None
    # if patients_split_col is None: socio_dem_imp.insert(1, 'patient_diagnosis_super_class')
    # res = run_models(timestamps=ts, model_data=df, models=('linear_rdn_all',),
    #                  covariates='patient_diagnosis_super_class', covariates_slope=covariates_slope,
    #                  patients_split_col=patients_split_col)
    # res = run_models(model_data=df, models=models, covariates=cvd_imp, covariates_slope=covariates_slope,
    #                  patients_split_col=patients_split_col,
    #                  output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_health_all_groups.xlsx')
    # res = run_models(model_data=df, models=models, covariates=socio_dem_imp, covariates_slope=covariates_slope,
    #                  patients_split_col=patients_split_col,
    #                  output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_sociodem_all_groups.xlsx')
    # if 'patient_diagnosis_super_class' in cvd_imp: cvd_imp.remove('patient_diagnosis_super_class')
    # res = run_models(model_data=df, models=models, covariates=socio_dem_imp + cvd_imp,
    #                  covariates_slope=covariates_slope, patients_split_col=patients_split_col,
    #                  output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_all_all_groups.xlsx')
    # res = run_models(model_data=df, models=models, covariates=cvd, covariates_slope=covariates_slope,
    #                  patients_split_col=patients_split_col,
    #                  output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_health_not_imputed.xlsx')
    # res = run_models(model_data=df, models=models, covariates=socio_dem, covariates_slope=covariates_slope,
    #                  patients_split_col=patients_split_col,
    #                  output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_sociodem_not_imputed.xlsx')
    # res = run_models(model_data=df, models=models, covariates=socio_dem + cvd, covariates_slope=covariates_slope,
    #                  patients_split_col=patients_split_col,
    #                  output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_all_not_imputed.xlsx')


def run_report(dataset=ds.default_dataset):
    dataset.file_path = r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_trajectory_data_final6.csv'
    dataset.cols_to_pivot = ['patient_diagnosis_super_class']
    dataset.write_report(r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_report_1class.xlsx')
    dataset.cols_to_pivot = ['patient_diagnosis_super_class', 'patient_diagnosis_class']
    dataset.write_report(r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_report_2classes.xlsx')


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

    # MODEL 1: python equivalent
    model_py = smf.mixedlm("score_combined ~ score_date_centered ", df_to_use, groups=df_to_use['brcid'])
    result = model_py.fit()
    print(model_py.fit().summary())

    # random slope and intercept
    model_py = smf.mixedlm("score_combined ~ score_date_centered", df_to_use, groups=df_to_use['brcid'],
                           re_formula="~score_date_centered")
    model_py = sm.MixedLM.from_formula("score_combined ~ score_date_centered "
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
