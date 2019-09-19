from code_utils.global_variables import *
import datetime
import time
import longitudinal_models.longitudinal_dataset as ds
from longitudinal_models.lmer_utils import *
from pymer4.models import Lmer  # , Lm
# for python models
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.regression.mixed_linear_model as mlm

##############################################################################################
# LONGITUDINAL MODELLING
##############################################################################################
# TODO compare age at baseline vs age a diagnosis
# https://rpsychologist.com/r-guide-longitudinal-lme-lmer
# TODO add missing values (??)
# TODO plots for 200 sample in each subgroup: https://stats.idre.ucla.edu/r/faq/how-can-i-visualize-longitudinal-data-in-ggplot2/
"""
OPTION A) Here grouping variable is Organic only/SMI only/SMI + O
- MODEL 1: Unconditional with MMSE= intercept (fixed)
- MODEL 2: Unconditional with MMSE= intercept (random)
- MODEL 3: Unconditional with MMSE= intercept + time (fixed)
- MODEL 4: Unconditional with MMSE= intercept + time (random)
- MODEL 5: Conditional with MMSE= intercept + time + grouping variable

OPTION B) Run model 1- 4 in each group of the grouping variable. 3 outputs

Model 2: adjusted for socio-demographic covariates (age at first measurement, education, ethnicity, occupation, living status, marital status). 
Model 3: adjusted for health factors at baseline (BMI, systolic BP, smoking, glucose)
"""


def run_report(dataset=ds.default_dataset):
    dataset.file_path = r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_trajectory_data_final6.csv'
    dataset.cols_to_pivot = ['patient_diagnosis_super_class']
    dataset.write_report(r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_report_1class.xlsx')
    dataset.cols_to_pivot = ['patient_diagnosis_super_class', 'patient_diagnosis_class']
    dataset.write_report(r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_report_2classes.xlsx')


def all_models(dataset=ds.default_dataset,
               timestamps=['score_date_centered'],  # 'age_at_score_upbound'],
               covariates=None,
               models=['linear_rdn_int', 'linear_rdn_int_slope', 'quadratic_rdn_int'],
               input_file_path=r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_trajectory_data_final6.csv',
               output_file_path=r'T:\aurelie_mascio\multimorbidity\mmse_work\regression_results.xlsx'):
    if input_file_path is not None: dataset.file_path = input_file_path
    df = dataset.regression_cleaning(normalize=False, dummyfy=False, keep_only_baseline=False)
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%Hh%M')
    writer = pd.ExcelWriter(output_file_path.replace('.xlsx', '_' + st + '.xlsx'), engine='xlsxwriter')
    col_num = 0
    for patient_group in ['all'] + list(df.patient_diagnosis_super_class.unique()):
        df_tmp = df[df.patient_diagnosis_super_class == patient_group] if patient_group != 'all' else df
        row_num = 0
        for ts in timestamps:
            for m in models:
                formula = lmer_formula(model_type=m, regressor=dataset.to_predict[0], timestamp=ts,
                                       covariates=covariates, group=dataset.key)
                model = Lmer(formula, data=df_tmp)
                model.fit()
                title = pd.DataFrame([str(patient_group)], columns=['MODEL ' + m + ': ' + str(model.formula)],
                                     index=[ts])
                title.to_excel(writer, startrow=row_num, startcol=col_num)
                to_print = print_r_model_output(model)
                to_print.to_excel(writer, startrow=row_num + 2, startcol=col_num)
                row_num += 5 + len(to_print)
        col_num += to_print.shape[1] + 3
    writer.save()


def model_playground(dataset=ds.default_dataset, intercept='score_combined_baseline', timestamp='score_date_upbound'):
    # df = dataset.regression_cleaning(normalize=False, dummyfy=False, keep_only_baseline=False)
    df = pd.read_excel(r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\mmse_synthetic_data_20190919.xlsx',
                       index_col=None)
    df_smi = df[df.patient_diagnosis_super_class == 'smi only']
    df_orga = df[df.patient_diagnosis_super_class == 'organic only']
    df_smi_orga = df[df.patient_diagnosis_super_class == 'smi+organic']

    df_to_use = df_smi
    # MODEL 1: basic model (random intercept and fixed slope)
    model = Lmer('score_combined ~ score_date_centered  + (1|brcid)', data=df_to_use)  # MMSE score by year
    model = Lmer('score_combined ~ score_date_centered  + (1|brcid)', data=df_to_use)  # for subgroup
    model = Lmer('score_combined ~ score_date_centered  + age_at_score_baseline + (1|brcid)',
                 data=df)  # adding age at baseline as covariate (is this correct??)

    # MODEL 2: random intercept and random slope
    model = Lmer('score_combined ~  (score_date_centered  | brcid)', data=df_to_use)  # fails to converge
    model = Lmer("score_combined ~ score_date_centered + (1 + score_date_centered | brcid)",
                 data=df_to_use)  # correct one?
    model = Lmer('score_combined ~  1 + score_date_centered  + (1|brcid) + (0 + score_date_centered  | brcid)',
                 data=df)  # this converges but is it correct?
    model = Lmer('score_combined ~  (score_date_centered  + age_at_score_baseline| brcid)', data=df_to_use)

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
