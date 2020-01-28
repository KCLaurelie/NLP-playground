from code_utils.global_variables import *
from longitudinal_models.longitudinal_models import *
from longitudinal_models.imputation import *
from longitudinal_models.data_stats import load_data
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm


df = load_data(r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\mmse_synthetic_data_20200119.xlsx')

def test(df, impute=False):
    models = ('linear_rdn_all', 'linear_rdn_int')  # , 'linear_rdn_all', 'quadratic_rdn_int')
    ts = ('score_date_centered',)
    covariates_slope = True
    patients_split_col = None
    socio_dem_imp = ['age_at_score_baseline', 'gender', 'ethnicity_imputed', 'marital_status_imputed',
                     'education', 'first_language_imputed', 'imd_bucket_imputed', 'age_at_first_diag']
    cvd_imp = ['smoking_status_imputed', 'cvd_problem_imputed']
    references = {'education': 'no', 'ethnicity_imputed': 'white', 'ethnicity': 'white', 'smoking_status_imputed': 'no',
                  'smoking_status': 'no', 'marital_status_imputed': 'single or separated', 'marital_status': 'single or separated'}

    if patients_split_col is None: socio_dem_imp.insert(1, 'patient_diagnosis_super_class')
    socio_dem = [x.replace('_imputed', '') for x in socio_dem_imp]
    cvd = [x.replace('_imputed', '') for x in cvd_imp]

    #df[['smoking_status','smoking_status_imputed']] = df['smoking_status_imputed'].replace({'past': 'no'})
    for key, val in references.items():
        if key in df:
            df[key] = df[key].replace({val: 'aaa_'+val})
            print('replaced', val)
    if impute:  # if imputed data needs to be generated (this step takes a while)
        df = impute_with_baseline(df, key='brcid', baseline_cols=['brcid', 'score_date_upbound'],
                                  input_columns_to_exclude=['brcid', 'score_combined', 'occur', 'counter', 'score_combined_baseline'])

    res = run_models(timestamps=ts, model_data=df, models=('linear_rdn_all',), covariates='patient_diagnosis_super_class', covariates_slope=covariates_slope, patients_split_col=patients_split_col)
    res = run_models(timestamps=ts, model_data=df, models=('linear_rdn_all',), covariates=socio_dem_imp + cvd_imp, covariates_slope=covariates_slope, patients_split_col=patients_split_col)
    res = run_models(model_data=df, models=models, covariates=cvd_imp, covariates_slope=covariates_slope, patients_split_col=patients_split_col,
                     output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_health_all_groups.xlsx')
    res = run_models(model_data=df, models=models, covariates=socio_dem_imp, covariates_slope=covariates_slope, patients_split_col=patients_split_col,
                     output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_sociodem_all_groups.xlsx')
    if 'patient_diagnosis_super_class' in cvd_imp: cvd_imp.remove('patient_diagnosis_super_class')
    res = run_models(model_data=df, models=models, covariates=socio_dem_imp + cvd_imp, covariates_slope=covariates_slope, patients_split_col=patients_split_col,
                     output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_all_all_groups.xlsx')
    res = run_models(model_data=df, models=models, covariates=cvd, covariates_slope=covariates_slope, patients_split_col=patients_split_col,
                     output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_health_not_imputed.xlsx')
    res = run_models(model_data=df, models=models, covariates=socio_dem, covariates_slope=covariates_slope, patients_split_col=patients_split_col,
                     output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_sociodem_not_imputed.xlsx')
    res = run_models(model_data=df, models=models, covariates=socio_dem + cvd, covariates_slope=covariates_slope, patients_split_col=patients_split_col,
                     output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_all_not_imputed.xlsx')


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
    model = Lmer('score_combined ~ score_date_centered  + gender + (1|brcid)', data=df_to_use)  # adding age at baseline as covariate (is this correct??)
    to_print = print_r_model_output(model.fit())
    # MODEL 2: random intercept and random slope
    model = Lmer('score_combined ~  (score_date_centered  | brcid)', data=df_to_use)  # this removes the intercept?
    model = Lmer('score_combined ~  (score_date_centered  + gender| brcid)', data=df_to_use)

    model = Lmer("score_combined ~ score_date_centered + (1 + score_date_centered | brcid)", data=df_to_use)  # correct one?
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