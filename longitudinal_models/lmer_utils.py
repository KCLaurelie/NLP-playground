from code_utils.global_variables import *
import pandas as pd
import numpy as np


def check_r_loc():
    from rpy2.robjects.packages import importr
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
    elif model_type == 'quadratic_rdn_int':
        model_str = regressor + ' ~ ' + timestamp + ' + I(' + timestamp + '^2)' + str_cov + ' + (1|' + group + ')'
    else:
        return 'model unknown'
    return model_str


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

    res = pd.concat([coefs, rnd_eff, stat, various], sort=True)[['type', 'Estimate (SE)', 'CI', 'P-val']]
    return res.set_index(['type', res.index])