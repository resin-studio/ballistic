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

model_str = f'''
def model(x):
    
    m = pyro.sample("m", dist.Normal(0.0, 1.0))
    
    b = pyro.sample("b", dist.Normal(0.0, 1.0))
    
    z = m*x+b
    
    with pyro.plate("data", len(x)):
        return pyro.sample("obs", dist.Normal(z, 1.0))
'''

(exec(model_str))
print(model(torch.tensor([5.0])))