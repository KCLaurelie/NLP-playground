# http://nadbordrozd.github.io/blog/2017/03/05/missing-data-imputation-with-bayesian-networks/

from dstk.pymc_utils import make_bernoulli, cartesian_bernoulli_child
from dstk.imputation import BayesNetImputer

class DuckImputer(BayesNetImputer):
    def construct_net(self, df):
        quacks = make_bernoulli('quacks_like_a_duck', value=df.quacks_like_a_duck)
        swims = make_bernoulli('swims_like_a_duck', value=df.swims_like_a_duck)
        duck = cartesian_bernoulli_child('duck', parents=[quacks, swims], value=df.duck)
        return pymc.Model([quacks, swims, duck])

print(DuckImputer(method='MCMC').fit_transform(with_missing))