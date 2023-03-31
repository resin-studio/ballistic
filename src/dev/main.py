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

def mything():
    model_str = f'''
def model(x):
    
    m = pyro.sample("m", dist.Normal(0.0, 1.0))
    
    b = pyro.sample("b", dist.Normal(0.0, 1.0))
    
    z = m*x+b
    
    with pyro.plate("data", len(x)):
        return pyro.sample("obs", dist.Normal(z, 1.0))

auto_guide = pyro.infer.autoguide.AutoNormal(model)
adam = pyro.optim.Adam({{"lr": 0.02}})  # Consider decreasing learning rate.
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

predictive = pyro.infer.Predictive(model, guide=auto_guide, num_samples=100)
# construct a function from scalar to distribution
def foo(x):
    global predictive
    svi_samples = predictive(torch.tensor([x]))
    svi_gdp = svi_samples["obs"]
    return svi_gdp[:, 0]
    '''

    d = {}
    exec(model_str, globals(), d)
    foo = d['foo']
    return foo



foo = mything()
print(foo(5.0))
# print(foo(5.0).mean().item())

# def model(x):
    
#     m = pyro.sample("m", dist.Normal(0.0, 1.0))
    
#     b = pyro.sample("b", dist.Normal(0.0, 1.0))
    
#     z = m*x+b
    
#     with pyro.plate("data", len(x)):
#         return pyro.sample("obs", dist.Normal(z, 1.0))


# auto_guide = pyro.infer.autoguide.AutoNormal(model)
# adam = pyro.optim.Adam({"lr": 0.02})  # Consider decreasing learning rate.
# elbo = pyro.infer.Trace_ELBO()
# svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

# # losses = []
# # for step in range(1000 if not smoke_test else 2):  # Consider running for more steps.
# #     loss = svi.step(is_cont_africa, ruggedness, log_gdp)
# #     losses.append(loss)
# #     if step % 100 == 0:
# #         logging.info("Elbo loss: {}".format(loss))

# # plt.figure(figsize=(5, 2))
# # plt.plot(losses)
# # plt.xlabel("SVI step")
# # plt.ylabel("ELBO loss");

# predictive = pyro.infer.Predictive(model, guide=auto_guide, num_samples=100)
# # construct a function from scalar to distribution
# def foo(x):
#     svi_samples = predictive(torch.tensor([x]))
#     svi_gdp = svi_samples["obs"]
#     return svi_gdp[:, 0]

# print(foo(5.0))
# print(foo(5.0).mean().item())


# print(model(torch.tensor([5.0])))