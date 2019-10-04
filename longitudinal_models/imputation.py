# http://nadbordrozd.github.io/blog/2017/03/05/missing-data-imputation-with-bayesian-networks/
# https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779
# https://www.worldscientific.com/doi/pdf/10.1142/9789813207813_0021 (paper for justification???)
import datawig
import pandas as pd
from pandas.api.types import is_string_dtype
import numpy as np


def test():
    df_orig = pd.read_csv("https://goo.gl/ioc2Td", usecols=['pop_1992', 'pop_1997', 'pop_2002', 'pop_2007', 'country', 'continent'])
    df = df_orig.mask(np.random.random(df_orig.shape) < 0.3)
    input_columns = ['pop_1992', 'pop_1997', 'pop_2002', 'country']
    output_column = 'pop_2007'

    res = impute_all_data(df)

    df_train, df_test = datawig.utils.random_split(df)

    imputer = datawig.SimpleImputer(
        input_columns=input_columns,  # column(s) containing information about the column we want to impute
        output_column=output_column,  # the column we'd like to impute values for
        )
    imputer.fit(train_df=df_train, num_epochs=50)
    imputed = imputer.predict(df_test)


def impute_data(df, output_column, input_columns, num_epochs=50):
    df_train = df.dropna(subset=[output_column])
    if is_string_dtype(df[output_column]) and\
            len(df[output_column].unique()) >= len(df[output_column].dropna()):
        print(output_column, 'is categorical and only has unique values, cannot do imputation')
        return df

    imputer = datawig.SimpleImputer(
        input_columns=input_columns,  # column(s) containing info about the column we want to impute
        output_column=output_column,  # the column we'd like to impute values for
    )
    # Fit an imputer model on the train data
    imputer.fit(train_df=df_train, num_epochs=num_epochs)

    # Impute missing values and return original dataframe with predictions
    imputed_df = imputer.predict(df)
    return imputed_df


def impute_all_data(df, output_column=None, input_columns_to_exclude=None, clean_df=False, **kwargs):
    if clean_df:
        df = df.apply(lambda x: x.str.lower() if isinstance(x, str) else x)
        df = df.replace({'not known': np.nan, 'unknown': np.nan, '[nan-nan]': np.nan})
    if output_column is None:  # by default use all columns containing missing values
        output_column = df.columns[df.isna().any()].tolist()

    for col in output_column:
        print('imputing data for', col)
        input_columns = [x for x in df.columns if x != col]
        if input_columns_to_exclude is not None:
            input_columns = [x for x in input_columns if x not in input_columns_to_exclude]
        imputed_df = impute_data(df, output_column=col, input_columns=input_columns, **kwargs)

        df[col + '_imputed'] = imputed_df[col + '_imputed'] if (col + '_imputed') in imputed_df else 'imputation failed'
        if (col + '_imputed_proba') in imputed_df: df[col + '_imputed_proba'] = imputed_df[col + '_imputed_proba']
        df[col + '_final'] = imputed_df[col].fillna(df[col + '_imputed'])

    return df
