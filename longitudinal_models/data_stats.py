import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu

subgroup_cols = ['age_bucket_baseline', 'gender', 'ethnicity', 'marital_status', 'education', 'first_language',
                 'imd_bucket', 'smoking_status', 'cvd_problem']


def load_data(data_file=r'C:\Users\K1774755\Downloads\phd\mmse_rebecca\mmse_synthetic_data_20200119.xlsx',
              sheet_name='combined'):
    df = pd.read_excel(data_file, sheet_name=sheet_name, index_col=None)
    if 'keep' in df.columns: df = df.loc[(df.keep == 'yes') or (df.keep == 1) or (df.keep is True)]
    df = df.loc[df.patient_diagnosis_super_class != 'smi only']
    # if 'counter' in df.columns: df = df.loc[df.counter > 2]
    # df = df.loc[(df.age_at_first_diag > 49.5) & (df.age_at_score >= 50) & (df.age_at_score <= 90)]
    # df = df.loc[df.age_at_score_baseline >= df.age_at_first_diag]
    # df['has_smi'] = np.where(df['patient_diagnosis_super_class'].str.lower().str.contains('smi'), 'yes', 'no')
    return df


def ttest(df, value='score_combined',
          group_col='patient_diagnosis_super_class',
          groups_to_study=['organic only', 'smi+organic'],
          subgroup_col='age_bucket_baseline'):
    usecols = [subgroup_col, group_col, value] if subgroup_col is not None else [group_col, value]
    #df = pd.read_excel(data_file, sheet_name='combined', index_col=None, usecols=usecols)
    subgroups = df[subgroup_col].unique() if subgroup_col is not None else ['whatever']
    subgroups.sort()

    res = pd.DataFrame(columns=['t (ttest)', 'p (ttest)', 'stat (utest)', 'p (utest)'])
    for sub in subgroups:
        df_tmp = df.loc[df[subgroup_col] == sub] if subgroup_col is not None else df
        g1 = df_tmp.loc[df_tmp[group_col] == groups_to_study[0], value]
        g2 = df_tmp.loc[df_tmp[group_col] == groups_to_study[1], value]
        t, p = stats.ttest_ind(g1, g2, equal_var=False)
        stat, pu = mannwhitneyu(g1, g2)
        res.loc[sub] = [t, p, stat, pu]
        print('variable=', sub, 't (ttest)=', t, 'p (ttest)=', p)
    return res


def chisq(df, value='brcid',
          group_col='patient_diagnosis_super_class',
          groups_to_study=['organic only', 'smi+organic']):
    usecols = subgroup_cols + [group_col, value]
    #df = pd.read_excel(data_file, sheet_name='combined', index_col=None, usecols=usecols)
    df = df[df[group_col].isin(groups_to_study)]

    res = pd.DataFrame()  # (columns=['g', 'p'])
    for sub in subgroup_cols:
        df_tmp = df.pivot_table(values=value, index=sub, columns=group_col, aggfunc=pd.Series.nunique, margins=False,
                                fill_value=0)
        chi2, p, dof, ex = stats.chi2_contingency(df_tmp)
        df_tmp['variable'] = sub
        df_tmp['chi2'] = chi2
        df_tmp['p'] = p
        res = pd.concat([res, df_tmp])
    return res


def data_stats(df, group_col='patient_diagnosis_super_class'):
    #df = pd.read_excel(data_file, sheet_name='combined', index_col=None)

    res = pd.DataFrame()
    for sub in subgroup_cols:
        df_countunique = df.pivot_table(values='brcid', index=sub, columns=group_col, aggfunc=pd.Series.nunique, margins=True, fill_value=0)
        df_countunique.columns = [x + '(1. N)' for x in df_countunique.columns]
        for col in df_countunique.columns:
            df_countunique[col.replace('N', '%')] = (df_countunique[col] / (df_countunique[col].sum()/2) * 100)
        df_count = df.pivot_table(values='brcid', index=sub, columns=group_col, aggfunc='count', margins=True, fill_value=0)
        df_count.columns = [x + '(2. obs)' for x in df_count.columns]
        df_mean = df.pivot_table(values='score_combined', index=sub, columns=group_col, aggfunc=np.mean, margins=True)
        df_mean.columns = [x + '(3. M)' for x in df_mean.columns]
        df_sd = df.pivot_table(values='score_combined', index=sub, columns=group_col, aggfunc=np.std, margins=True)
        df_sd.columns = [x + '(4. SD)' for x in df_sd.columns]
        res_tmp = pd.concat([df_mean, df_count, df_sd, df_countunique], axis=1)
        res_tmp = res_tmp.reindex(sorted(res_tmp.columns), axis=1)
        res_tmp['variable'] = sub
        res = pd.concat([res, res_tmp])

    return res