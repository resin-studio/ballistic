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
# with pyro.plate('S', 3):
#     seasonality = pyro.sample('seasonality', dist.Normal(0.0, 1.0))
#     seasonality = seasonality - seasonality.mean()

# slope = 10
# with pyro.plate('N', 10) as time:
#     result = slope + seasonality.repeat(math.ceil(10/3))[:10]
#     ###################
#     # result = slope + seasonality[time % 3]
#     # result = 55 
#     print(result)


# for x in torch.tensor([45, 78]):
#     print(x)


# for x in range(1,2): print(x)

# samples = torch.tensor([[1],[2],[3]])
# print(samples[:, 0].tolist())

# for row, n in (zip(samples, [9, 8, 7])):
#     print(f'{row}, {n}')

import math
# print(1. * torch.tensor([2,3,4]))

# for x in (range(1, 2)):
#     print(x)

print([1,2,3] * 2)

print(next((True for i in [1,2,3] if i == 2), False))