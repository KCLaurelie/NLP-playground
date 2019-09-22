from code_utils.global_variables import *
from code_utils.general_utils import list_combos
import longitudinal_models.longitudinal_dataset as ds
from longitudinal_models.lmer_utils import *
import datetime
import time
from pymer4.models import Lmer  # , Lm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm


##############################################################################################
# LONGITUDINAL MODELLING
##############################################################################################
# TODO compare age at baseline vs age a diagnosis
# https://rpsychologist.com/r-guide-longitudinal-lme-lmer
# TODO add missing values (impute or complete case??)
# TODO plots for 200 sample in each subgroup: https://stats.idre.ucla.edu/r/faq/how-can-i-visualize-longitudinal-data-in-ggplot2/


def run_report(dataset=ds.default_dataset):
    dataset.file_path = r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_trajectory_data_final6.csv'
    dataset.cols_to_pivot = ['patient_diagnosis_super_class']
    dataset.write_report(r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_report_1class.xlsx')
    dataset.cols_to_pivot = ['patient_diagnosis_super_class', 'patient_diagnosis_class']
    dataset.write_report(r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_report_2classes.xlsx')


def prep_regression_data(dataset=ds.default_dataset,
                         raw_data_path=None):
    if raw_data_path is not None:  # need to prepare data for regression
        dataset.file_path = raw_data_path
    regression_data = dataset.regression_cleaning(normalize=False, dummyfy=False, keep_only_baseline=False)
    to_predict = dataset.to_predict[0]
    key = dataset.key
    covariates = dataset.regressors
    timestamps = dataset.timestamp_cols
    return [regression_data, to_predict, key, covariates, timestamps]


def run_models(model_data=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\mmse_synthetic_data_20190919.xlsx',
               to_predict='score_combined',
               key='brcid',
               covariates=None,
               timestamps=('score_date_centered',),
               complete_case=False,
               models=('linear_rdn_int', 'linear_rdn_all_no_intercept', 'linear_rdn_all', 'quadratic_rdn_int'),
               output_file_path=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\regression.xlsx'):
    if isinstance(model_data, str) and 'xlsx' in model_data:  # load regression data
        model_data = pd.read_excel(model_data, index_col=None)
    if complete_case:
        model_data = model_data.replace(
            {'not known': np.nan, 'Not Known': np.nan, 'unknown': np.nan, 'Unknown': np.nan, '[nan-nan]': np.nan})
        model_data = model_data.dropna(subset=list(covariates), how='any')
    if output_file_path is not None:
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%Hh%M')
        tag = '_' + st + ('_cov_' + str(covariates) if covariates is not None else '') \
              + ('_complete_case' if complete_case else '')
        writer = pd.ExcelWriter(output_file_path.replace('.xlsx', tag + '.xlsx'), engine='xlsxwriter')

    res = []
    col_num = 0
    for patient_group in list(model_data.patient_diagnosis_super_class.unique()):
        df_tmp = model_data[model_data.patient_diagnosis_super_class == patient_group] \
            if patient_group != 'all' else model_data
        row_num = 0
        for ts in timestamps:
            for m in models:
                print('running model:', m, '(patient group:', patient_group, ', timestamp:', ts, ')')
                formula = lmer_formula(model_type=m, regressor=to_predict, timestamp=ts,
                                       covariates=covariates, group=key)
                print('using formula', formula)
                model = Lmer(formula, data=df_tmp)
                try:
                    model.fit(REML=True)
                    if model.warnings is not None:  # try unrestricted MLE if convergence failed
                        model.fit(REML=False)
                    to_print = print_r_model_output(model)
                except:
                    print('something went wrong with model fitting')
                    to_print = pd.DataFrame({'output': 'failure'}, index=[0])
                to_print = pd.concat([to_print], keys=[patient_group], names=[m])

                if output_file_path is not None:
                    to_print.to_excel(writer, startrow=row_num, startcol=col_num)
                    row_num += 2 + len(to_print)
                else:
                    res = res.append(to_print)

        if output_file_path is not None: col_num += to_print.shape[1] + 3
    if output_file_path is not None: writer.save()
    return res


def all_covariates(model_data=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\mmse_synthetic_data_20190919.xlsx',
                   models='linear_rdn_all',
                   covariates=('gender', 'ethnicity_group', 'first_language', 'marital_status', 'education_bucket_raw',
                               'smoking_status_baseline', 'imd_bucket_baseline', 'cvd_problem'),
                   combos_length=None,
                   **kwargs):
    df = pd.read_excel(model_data, index_col=None)
    df = df.replace({'not known': np.nan, 'Not Known': np.nan, 'unknown': np.nan, 'Unknown': np.nan, '[nan-nan]': np.nan})
    cov_comb = list_combos(covariates, r=combos_length)  # create all combinations of covariates

    res = []
    for cov in cov_comb:
        print('running models for', cov)
        res = res.append(
            run_models(model_data=df, covariates=cov, models=(models,), output_file_path=None, **kwargs))

    return res


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
