from code_utils.global_variables import *
import pandas as pd
import numpy as np
from rpy2.robjects.packages import importr


def check_r_loc():
    base = importr('base')
    return base.R_home()


def lmer_formula(model_type='linear_rdn_int',
                 regressor='score_combined',
                 timestamp='age_score_upbound',
                 covariates=None,
                 group='brcid'):
    if covariates is None:
        str_cov = ''
    elif isinstance(covariates, str):
        str_cov = ' + ' + covariates
    else:
        str_cov = ' + ' + ' + '.join(covariates)
    if model_type == 'linear_rdn_int':
        model_str = regressor + ' ~ ' + timestamp + str_cov + ' + (1|' + group + ')'
    elif model_type == 'linear_rdn_int_slope':
        model_str = regressor + ' ~  (' + timestamp + str_cov + ' | ' + group + ')'
        # TODO check which one is correct
        # model_str = regressor + ' ~  ' + timestamp + str_cov + ' (1 + ' + timestamp + ' | ' + group + ')'
    elif model_type == 'quadratic_rdn_int':
        model_str = regressor + ' ~ ' + timestamp + ' + I(' + timestamp + '^2)' + str_cov + ' + (1|' + group + ')'
    else:
        return 'model unknown'
    return model_str


def print_r_model_output(model):
    stat = pd.DataFrame({'type': 'stats', 'Estimate (SE)': [np.round(model.logLike, 3), np.round(model.AIC, 3), BIC(model)]},
                        index=['-2LL', 'AIC', 'BIC'])
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

    res = pd.concat([coefs, rnd_eff, stat, various], sort=True)[['type', 'Estimate (SE)', 'CI', 'P-val']]
    return res.set_index(['type', res.index])


def BIC(model):
    resid = model.data['residuals']
    x = model.design_matrix
    res = np.log((len(resid))) * x.shape[1] - 2 * model.logLike
    return res


def ICC(df, groups_col='brcid', values_col='score_combined'):
    # from here: https://stackoverflow.com/questions/40965579/intraclass-correlation-in-python-module
    r_icc = importr("ICC")
    icc_res = r_icc.ICCbare(groups_col, values_col, data=df)
    icc_val = icc_res[0]  # icc_val now holds the icc value
    return icc_val
