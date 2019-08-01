from code_utils.global_variables import *
import pandas as pd
import numpy as np
import datetime
import time
import longitudinal_models.longitudinal_dataset as ds
from pymer4.models import Lmer  # , Lm

# for python models
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
    dataset.file_path = r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_trajectory_data_final6.csv'
    dataset.cols_to_pivot = ['patient_diagnosis_super_class']
    dataset.write_report(r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_report_1class.xlsx')
    dataset.cols_to_pivot = ['patient_diagnosis_super_class', 'patient_diagnosis_class']
    dataset.write_report(r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_report_2classes.xlsx')


def print_r_model_output(model):
    stat = pd.DataFrame({'type': 'stats', 'Estimate (SE)': [np.round(model.logLike, 3), np.round(model.AIC, 3)]},
                        index=['-2LL', 'AIC'])
    rnd_eff = model.ranef_var.Var.round(3).astype(str) + ' (' + model.ranef_var.Std.round(3).astype(str) + ')'
    rnd_eff = pd.DataFrame({'type': 'variances', 'Estimate (SE)': rnd_eff}).set_index(
        (model.ranef_var.index + ' ' + model.ranef_var.Name).values)
    various = pd.DataFrame({'type': 'misc', 'Estimate (SE)': [model.grps, len(model.data), model.warnings]},
                           index=['groups', 'obs', 'warnings'])
    coefs = pd.DataFrame(model.coefs)
    coefs['type'] = 'coefs'
    coefs['CI'] = '[' + coefs['2.5_ci'].round(3).astype(str) + ',' + coefs['97.5_ci'].round(3).astype(str) + ']'
    coefs['Estimate (SE)'] = coefs.Estimate.round(3).astype(str) + ' (' + coefs.SE.round(3).astype(str) + ')'
    coefs = coefs[['type', 'Estimate (SE)', 'CI', 'P-val']]

    # res = pd.concat([coefs, rnd_eff, stat, various], sort=True)[['type', 'Estimate (SE)','CI', 'Var (Std)', 'P-val']]
    res = pd.concat([coefs, rnd_eff, stat, various], sort=True)[['type', 'Estimate (SE)', 'CI', 'P-val']]
    return res.set_index(['type', res.index])


def all_models(dataset=ds.default_dataset,
               input_file_path=r'T:\aurelie_mascio\multimorbidity\mmse_work\mmse_trajectory_data_final6.csv',
               output_file_path=r'T:\aurelie_mascio\multimorbidity\mmse_work\regression_results.xlsx'):
    dataset.file_path = input_file_path
    df = dataset.regression_cleaning(normalize=False, dummyfy=False, keep_only_baseline=False)
    df['healfyear_centered'] = df.score_date_upbound - df.score_date_upbound.min()
    timestamps = ['score_date_upbound', 'healfyear_centered', 'counter', 'age_at_score_upbound']
    df_smi = df[df.patient_diagnosis_super_class == 'smi only']
    df_orga = df[df.patient_diagnosis_super_class == 'organic only']
    df_smi_orga = df[df.patient_diagnosis_super_class == 'smi+organic']
    dfs = [df, df_smi, df_orga, df_smi_orga]
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%Hh%M')
    writer = pd.ExcelWriter(output_file_path.replace('.xlsx', '_' + st + '.xlsx'), engine='xlsxwriter')
    col_num = 0
    for df_tmp in dfs:  # for ts in timestamps:
        row_num = 0
        for ts in timestamps:  # for df_tmp in dfs:
            df_tmp['timestamp'] = df_tmp[ts]
            model = Lmer('score_combined ~ timestamp + (1|brcid)', data=df_tmp)
            model.fit()
            pd.DataFrame([str(df_tmp.patient_diagnosis_super_class.unique())],
                         columns=['MODEL 1: ' + str(model.formula)], index=[ts]).to_excel(writer, startrow=row_num,
                                                                                          startcol=col_num)
            to_print = print_r_model_output(model)
            to_print.to_excel(writer, startrow=row_num + 2, startcol=col_num)
            row_num += 5 + len(to_print)

            model = Lmer('score_combined ~ (timestamp | brcid)', data=df_tmp)
            model.fit()
            pd.DataFrame([str(df_tmp.patient_diagnosis_super_class.unique())],
                         columns=['MODEL 2: ' + str(model.formula)], index=[ts]).to_excel(writer, startrow=row_num,
                                                                                          startcol=col_num)
            to_print = print_r_model_output(model)
            to_print.to_excel(writer, startrow=row_num + 2, startcol=col_num)
            row_num += 5 + len(to_print)

            model = Lmer('score_combined ~ timestamp + I(timestamp^2) + (1|brcid)', data=df_tmp)
            model.fit()
            pd.DataFrame([str(df_tmp.patient_diagnosis_super_class.unique())],
                         columns=['MODEL 3: ' + str(model.formula)], index=[ts]).to_excel(writer, startrow=row_num,
                                                                                          startcol=col_num)
            to_print = print_r_model_output(model)
            to_print.to_excel(writer, startrow=row_num + 2, startcol=col_num)
            row_num += 5 + len(to_print)
        col_num += to_print.shape[1] + 3
    writer.save()


def model_playground(dataset=ds.default_dataset, intercept='score_combined_baseline', timestamp='score_date_upbound'):
    df = dataset.regression_cleaning(normalize=False, dummyfy=False, keep_only_baseline=False)
    df['intercept'] = df[intercept]
    df['timestamp'] = df[timestamp]
    df_smi = df[df.patient_diagnosis_super_class == 'smi only']
    df_orga = df[df.patient_diagnosis_super_class == 'organic only']
    df_smi_orga = df[df.patient_diagnosis_super_class == 'smi+organic']

    # MODEL 1: basic model (random intercept and fixed slope)
    model = Lmer('score_combined ~ score_date_upbound + (1|brcid)', data=df)  # MMSE score by year
    model = Lmer('score_combined ~ score_date_upbound + (1|brcid)', data=df_smi)  # for subgroup
    model = Lmer('score_combined ~ score_date_upbound + age_at_score_baseline + (1|brcid)',
                 data=df)  # adding age at baseline as covariate (is this correct??)

    # MODEL 2: random intercept and random slope
    model = Lmer('score_combined ~  (score_date_upbound | brcid)', data=df)  # fails to converge
    model = Lmer('score_combined ~  1 + score_date_upbound + (1|brcid) + (0 + score_date_upbound | brcid)',
                 data=df)  # this converges but is it correct?
    model = Lmer('score_combined ~  (score_date_upbound + age_at_score_baseline| brcid)', data=df)

    # MODEL 3: basic model but quadratic
    model = Lmer('score_combined ~ score_date_upbound + I(score_date_upbound^2) + (1|brcid)', data=df)

    print(model.fit())

    #######################################################################################
    #  PYTHON STUFF

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
