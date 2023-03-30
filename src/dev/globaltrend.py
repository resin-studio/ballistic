import logging
import os

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import math

import pyro.distributions as dist
# from pyro.distributions import Normal, TorchDistributionMixin, LogNormal 
# HalfNormal = TorchDistributionMixin(torch.distributions.half_normal.HalfNormal)
# import pyro.distributions as dist
# Bernoulli = dist.TorchDistributionMixin(torch.distributions.Bernoulli)

# def model_global_trend(y):
#     N = y.size(0)
#     slope = pyro.sample('slope', Normal(0.0, 1.0))
#     obs_sd = pyro.sample('obs_sd', HalfNormal(1.0))
#     with pyro.plate('S', 52):
#         seasonality = pyro.sample('seasonality', Normal(0.0, 1.0))
#         seasonality = seasonality - seasonality.mean()
#     with pyro.plate('N', N) as time:
#         seasonality = seasonality.repeat(math.ceil(N/52))[:N]
#         log_y_hat = slope * time + seasonality
#         return pyro.sample('y', LogNormal(log_y_hat, obs_sd), obs=y)

def weather():
    cloudy = pyro.sample('cloudy', dist.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = pyro.sample('temp', dist.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()

def scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    measurement = pyro.sample("measurement", dist.Normal(weight, 0.75))
    return measurement

conditioned_scale = pyro.condition(scale, data={"measurement": torch.tensor(14.)})

from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import HMC
from pyro.infer import EmpiricalMarginal
import matplotlib.pyplot as plt
# %matplotlib inline
guess_prior = 10.
hmc_kernel = HMC(conditioned_scale, step_size=0.9, num_steps=4)
posterior = MCMC(hmc_kernel, 
                 num_samples=1000, 
                 warmup_steps=50).run(guess_prior)

marginal = EmpiricalMarginal(posterior, "weight")
# plt.hist([marginal().item() for _ in range(1000)],)
# plt.title("P(weight | measurement = 14)")
# plt.xlabel("Weight")
# plt.ylabel("#")

# plt.show()


# if __name__ == "__main__":