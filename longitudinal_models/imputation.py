# http://nadbordrozd.github.io/blog/2017/03/05/missing-data-imputation-with-bayesian-networks/
import pymc
from dstk.pymc_utils import make_bernoulli, cartesian_bernoulli_child
from dstk.imputation import BayesNetImputer
import datawig

class DuckImputer(BayesNetImputer):
    def construct_net(self, df):
        quacks = make_bernoulli('quacks_like_a_duck', value=df.quacks_like_a_duck)
        swims = make_bernoulli('swims_like_a_duck', value=df.swims_like_a_duck)
        duck = cartesian_bernoulli_child('duck', parents=[quacks, swims], value=df.duck)
        return pymc.Model([quacks, swims, duck])

print(DuckImputer(method='MCMC').fit_transform(with_missing))




df_train, df_test = datawig.utils.random_split(train)

# Initialize a SimpleImputer model
imputer = datawig.SimpleImputer(
    input_columns=['1','2','3','4','5','6','7', 'target'], # column(s) containing information about the column we want to impute
    output_column='0', # the column we'd like to impute values for
    output_path='imputer_model' # stores model data and metrics
    )

# Fit an imputer model on the train data
imputer.fit(train_df=df_train, num_epochs=50)

# Impute missing values and return original dataframe with predictions
imputed = imputer.predict(df_test)