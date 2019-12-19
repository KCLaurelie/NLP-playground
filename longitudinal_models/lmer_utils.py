from code_utils.global_variables import *
import pandas as pd
import numpy as np
from rpy2.robjects.packages import importr
import code_utils.general_utils as gutils


def check_r_loc():
    base = importr('base')
    return base.R_home()


def lmer_formula(model_type='linear_rdn_int',
                 regressor='score_combined',
                 timestamp='score_date_centered',
                 covariates=None,
                 covariates_slope=False,
                 group='brcid'):
    # decent explanation of different R models:
    # https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html

    # first build covariates string
    if covariates is None:
        str_cov = ''
    else:
        covariates = gutils.to_list(covariates)
        str_cov = ' + ' + ' + '.join(covariates)
        if covariates_slope:
            add_slope = ' + ' + timestamp + ' * '
            str_cov += add_slope + add_slope.join(covariates)

    # now build formula
    if model_type == 'linear_rdn_int':  # random intercept only, linear model
        model_str = regressor + ' ~ ' + timestamp + str_cov + ' + (1|' + group + ')'
    elif model_type == 'linear_rdn_all_no_intercept':  # random slope only, no intercept (??)
        model_str = regressor + ' ~  (' + timestamp + str_cov + ' | ' + group + ')'
    elif model_type == 'linear_rdn_all':  # random slope, random intercept
        model_str = regressor + ' ~  ' + timestamp + str_cov + ' + (1 + ' + timestamp + ' | ' + group + ')'
    elif model_type == 'linear_rdn_all_uncorrel':  # random effects are constrained to be uncorrelated
        model_str = regressor + ' ~  1 + ' + timestamp + str_cov \
                    + ' + (0 + ' + timestamp + ' | ' + group + ')' \
                    + ' + (1|' + group + ')'
    elif model_type == 'quadratic_rdn_int':  # random intercept only, quadratic model
        model_str = regressor + ' ~ ' + timestamp + ' + I(' + timestamp + '^2)' + str_cov + ' + (1|' + group + ')'
    else:
        return 'model unknown'
    return model_str


def print_r_model_output(model):
    stat = pd.DataFrame({'type': 'stats', 'Estimate (SE)': [np.round(model.logLike, 3), np.round(model.AIC, 3),
                                                            BIC(model), ICC(model)]},
                        index=['-2LL', 'AIC', 'BIC', 'ICC'])
    rnd_eff = model.ranef_var.Var.round(3).astype(str) + ' (' + model.ranef_var.Std.round(3).astype(str) + ')'
    rnd_eff = pd.DataFrame({'type': 'variances', 'Estimate (SE)': rnd_eff}).set_index(
        (model.ranef_var.index + ' ' + model.ranef_var.Name).values)
    various = pd.DataFrame({'type': 'misc', 'Estimate (SE)': [model.grps, len(model.data),
                                                              model.warnings, model.formula]},
                           index=['groups', 'obs', 'warnings', 'formula'])
    coefs = pd.DataFrame(model.coefs)
    coefs['type'] = 'coefs'
    # get multiplier for slope * covariate
    coefs.loc[coefs.index.str.contains(':'), 'tmp'] = coefs.loc[coefs.index.str.contains(':')].index.str.split(pat=":").str[0]
    coefs['mult'] = coefs['tmp'].apply(lambda x: np.sign(coefs.loc[coefs.index == x, 'Estimate'][0]) if x in coefs.index else 1)
    coefs.Estimate = coefs.Estimate * coefs.mult
    # compute stats
    coefs['CI'] = '[' + coefs['2.5_ci'].round(3).astype(str) + ',' + coefs['97.5_ci'].round(3).astype(str) + ']'
    coefs['Estimate (SE)'] = coefs.Estimate.round(3).astype(str) + ' (' + coefs.SE.round(3).astype(str) + ')'
    coefs = coefs[['type', 'Estimate (SE)', 'CI', 'P-val', 'Sig']]
    # coefs['significance'] = gutils.p_value_sig(coefs['P-val'])
    coefs['Estimate (SE)'] = coefs['Estimate (SE)'] + coefs.Sig  # + gutils.p_value_sig(coefs['P-val']).astype(str)
    res = pd.concat([coefs, rnd_eff, stat, various], sort=True)[['type', 'Estimate (SE)', 'CI', 'P-val']]
    return res.set_index(['type', res.index])


def BIC(model):
    # formula taken from: http://eshinjolly.com/pymer4/_modules/pymer4/models.html#Lmer
    resid = model.data['residuals']
    x = model.design_matrix
    res = np.log((len(resid))) * x.shape[1] - 2 * model.logLike
    return res


def ICC(model):
    # formula taken from: https://stats.stackexchange.com/questions/113577/interpreting-the-random-effect-in-a-mixed-effect-model/113825#113825
    resid_var = model.ranef_var['Var']['Residual']
    sum_vars = model.ranef_var['Var'].sum()
    res = 1 - resid_var/sum_vars
    return res


def ICC2(df, groups_col='brcid', values_col='score_combined'):
    # formula taken from: https://stackoverflow.com/questions/40965579/intraclass-correlation-in-python-module
    r_icc = importr("ICC")
    icc_res = r_icc.ICCbare(groups_col, values_col, data=df)
    icc_val = icc_res[0]  # icc_val now holds the icc value
    return icc_val
