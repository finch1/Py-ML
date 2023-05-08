# https://github.com/rasmusab/bayesianprobabilitiesworkshop/blob/master/Exercise%201.ipynb
# https://www.youtube.com/watch?v=3OJEae7Qb_o

import pandas as pd 
import numpy as numpy

# number of random draws from the prior
n_draws = 10000

#  sample n_draws draws from the prior into pandas Series ( to have 
# convenient methods acailanle for histograms and descriptive statistics, e.g. median)
prior = pd.Series(...)

prior.hist() # It's always good to eyeball the prior to make sre it looks ok

# define the generative model
def generative_model(params):
    return()

# simulate data using the parameters from the prior and the generative model
sim_data = list()
for p in prior:
    sim_data.append(generative_model(p))

# filter off all draws that do not match the data
posterior = prior[list(map(lambda x: x == observed_data, sim_data))]

posterior.hist() # eyeball the posterior

# see that we got enough draws left after the filtering
# there are no rules here, but probably best to aim for > 1000 draws

# summerize the posterior, where a common cummary is to take the mean or median
# posterior and perhaps a 95% quantile interval

print('Number of draws left: %d, Posterior median: %.3f, Posterior quantile interval: %.3f-%.3f' % len(posterior), posterior.median(), posterior.quantile(0.025), posterior.quantile(0.975))