from __future__ import division
import numpy as np
import math
import pandas as pd
import scipy
import scipy.stats as sct

def copula(n_sim,corr_matrix,SEED=42,cop_type='gauss',nu=5):
    mean=[0]*len(corr_matrix)
    np.random.seed(SEED)
    if cop_type=='t':
        rdn=multivariate_t(mean,corr_matrix,nu,int(n_sim))
        rdn=sct.t.cdf(rdn,nu)
    else:
        rdn=np.random.multivariate_normal(mean,corr_matrix,int(n_sim))
        rdn=sct.norm.cdf(rdn)
    return rdn

def multivariate_t(mean,sigma,nu,n_sim):
    d=len(sigma)
    g=np.tile(np.random.gamma(nu/2.,2./nu,n_sim),(d,1)).T
    Z=np.random.multivariate_normal(np.zeros(d),sigma,n_sim)
    return mean+Z/np.sqrt(g)

def pareto_copula(cov,n_sim,xm,L20,L5,SEED):
    ETL=0.8*xm+0.15*L20+0.05*L5
    alpha=ETL/(ETL-xm)
    rdn=copula(n_sim,cov,SEED)
    return sct.pareto.ppf(rdn,alpha,scale=xm)