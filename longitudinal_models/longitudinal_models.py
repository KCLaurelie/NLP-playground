from code_utils.general_utils import list_combos, to_list
import longitudinal_models.longitudinal_dataset as ds
from longitudinal_models.lmer_utils import *
import datetime
import time
from pymer4.models import Lmer  # , Lm

##############################################################################################
# LONGITUDINAL MODELLING
##############################################################################################
# https://rpsychologist.com/r-guide-longitudinal-lme-lmer
# https://www.statsmodels.org/stable/examples/notebooks/generated/mixed_lm_example.html
# TODO plots for 200 sample in each subgroup: https://stats.idre.ucla.edu/r/faq/how-can-i-visualize-longitudinal-data-in-ggplot2/


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
               covariates_slope=False,
               patients_split_col='patient_diagnosis_super_class',
               timestamps=('score_date_centered',),
               complete_case=False,
               models=('linear_rdn_int', 'linear_rdn_all_no_intercept', 'linear_rdn_all', 'quadratic_rdn_int'),
               output_file_path=None,
               conf_int='Wald',
               REML=True):
    """

    :param model_data:
    :param to_predict:
    :param key:
    :param covariates:
    :param covariates_slope:
    :param patients_split_col:
    :param timestamps:
    :param complete_case:
    :param models:
    :param output_file_path:
    :param conf_int: which method to compute confidence intervals; 'profile', 'Wald' (default), or 'boot' (parametric bootstrap)
    :param REML: (bool) whether to fit using restricted maximum likelihood estimation instead of maximum likelihood estimation; default True
    :return:
    """
    if isinstance(model_data, str) and 'xlsx' in model_data:  # load regression data
        model_data = pd.read_excel(model_data, index_col=None)
    if covariates is not None:  # check covariates actually exist in the model data
        covariates = to_list(covariates)
        if not all(elem in model_data.columns for elem in list(covariates)):
            print('covariates entered do not exist in input data')
            return pd.DataFrame({'output': 'failure - covariates not in input data'}, index=[0])
    if complete_case:
        print('all cases:', len(model_data), 'observations, ', len(model_data[key].unique()), 'patients')
        model_data = model_data.replace(
            {'not known': np.nan, 'Not Known': np.nan, 'unknown': np.nan, 'Unknown': np.nan, '[nan-nan]': np.nan})
        model_data = model_data.dropna(subset=list(covariates), how='any')
        print('only complete cases:', len(model_data), 'observations, ', len(model_data[key].unique()), 'patients')
    if output_file_path is not None:
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%Hh%M')
        writer = pd.ExcelWriter(output_file_path.replace('.xlsx', st + '.xlsx'), engine='xlsxwriter')

    res = pd.DataFrame()
    col_num = 0
    patient_groups = list(model_data[patients_split_col].unique()) if patients_split_col is not None else ['all']
    for patient_group in patient_groups:
        df_tmp = model_data[model_data.patient_diagnosis_super_class == patient_group] \
            if patient_group != 'all' else model_data
        row_num = 0
        for ts in timestamps:
            for m in models:
                print('running model:', m, '(patient group:', patient_group, ', timestamp:', ts, ')')
                formula = lmer_formula(model_type=m, regressor=to_predict, timestamp=ts,
                                       covariates=covariates, covariates_slope=covariates_slope, group=key)
                print('using formula', formula)
                model = Lmer(formula, data=df_tmp)
                try:
                    model.fit(REML=REML, conf_int=conf_int)
                    if model.warnings is not None:  # try other method if convergence failed
                        model.fit(REML=(not REML), conf_int=conf_int)
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
                               'smoking_status_baseline', 'imd_bucket_baseline', 'cvd_problem', 'age_at_score_baseline'),
                   combos_length=1,
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



