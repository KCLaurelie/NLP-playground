from longitudinal_models.longitudinal_models import *
from longitudinal_models.imputation import *


def test(impute=False):
    df = pd.read_excel(r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\mmse_synthetic_data_20190919.xlsx', sheet_name='combined', index_col=None)
    models = ('linear_rdn_all',)  # ('linear_rdn_int', 'linear_rdn_all_no_intercept', 'linear_rdn_all', 'quadratic_rdn_int')
    socio_dem_imp = ['age_at_score_baseline', 'gender', 'ethnicity_group_imputed', 'marital_status_imputed',
                     'education_bucket_raw_imputed', 'first_language_imputed', 'imd_bucket_baseline_imputed', 'age_at_first_diag']
    socio_dem = [x.replace('_imputed', '') for x in socio_dem_imp]
    cvd_imp = ['smoking_status_baseline_imputed', 'cvd_problem_imputed']
    cvd = [x.replace('_imputed', '') for x in cvd_imp]

    if impute:  # if imputed data needs to be generated (this step takes a while)
        df_baseline = df.sort_values(['brcid', 'score_date_upbound']).groupby('brcid').first().reset_index()
        df_imputed = impute_all_data(df_baseline, output_column=None, clean_df=True,
                                     input_columns_to_exclude=['brcid', 'score_combined', 'occur', 'counter', 'score_combined_baseline'])
        df = df.merge(df_imputed[['brcid'] + [x for x in df_imputed.columns if '_final' in x]], on='brcid')
        df = df.reindex(sorted(df_imputed.columns), axis=1)

    res = run_models(model_data=df, complete_case=False, models=models, covariates=cvd_imp,
                     output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_health_imputed.xlsx')
    res = run_models(model_data=df, complete_case=False, models=models, covariates=cvd,
                     output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_health_not_imputed.xlsx')
    res = run_models(model_data=df, complete_case=False, models=models, covariates=socio_dem_imp,
                     output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_sociodem_imputed.xlsx')
    res = run_models(model_data=df, complete_case=False, models=models, covariates=socio_dem,
                     output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_sociodem_not_imputed.xlsx')
    res = run_models(model_data=df, complete_case=False, models=models, covariates=socio_dem_imp + cvd_imp,
                     output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression_all_imputed.xlsx')
    res = run_models(model_data=df, complete_case=False, models=models, covariates=socio_dem + cvd,
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