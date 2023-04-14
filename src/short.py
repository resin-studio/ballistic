import logging
import os

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro

import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import math

# smoke_test = ('CI' in os.environ)
# assert pyro.__version__.startswith('1.8.4')

# print(pyro.sample("sigma", dist.Uniform(0., 10.)))

# only the sample operation is affected by the plate
# seasonality * 3 ~ normal(0.0, 1.0)
with pyro.plate('S', 3):
    seasonality = pyro.sample('seasonality', dist.Normal(0.0, 1.0))
    seasonality = seasonality - seasonality.mean()

slope = 10
with pyro.plate('N', 10) as time:
    result = slope + seasonality.repeat(math.ceil(10/3))[:10]
    ###################
    # result = slope + seasonality[time % 3]
    # result = 55 
    print(result)
