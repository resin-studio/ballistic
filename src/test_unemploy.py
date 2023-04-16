from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from ballistic import prog
from ballistic import util 

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

from textx import metamodel_from_file
from datetime import datetime, date 

def d(s):
    return (datetime.strptime(s, '%m/%d/%y')).date()

if __name__ == "__main__":
    ##############################
    ## print('-----------------------------------------------------')
    ## program = prog.parse_from_file(util.resource('examples/unemploy.bll'))
    ## print(program)
    ## print('-----------------------------------------------------')
    ## print(prog.generate_model_from_ast(program))
    ## print('-----------------------------------------------------')
    ## print(d('1/1/05').toordinal())
    ## print(d('1/2/05').toordinal())
    ## df = df.iloc[0:4,:]
    ## print(df.head(4))
    ## print(df.values)
    ##############################

    DATA_URL = util.resource("data/unemployment_claims.csv")
    df = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
    df = df[df["County/Area"] == "WASHINGTON STATE"][df["Claim Type"] == "Initial claims"]
    df["Date"] = df["Date"].map(lambda s : d(s).toordinal())
    df = df[df["Date"] >= d('1/1/08').toordinal()][df["Date"] < d('11/1/19').toordinal()]
    # df["Date"] = df["Date"].map(lambda n : n - d('6/1/08').toordinal())
    df["Claims"] = df["Claims"].map(lambda s : float(s))
    df = df[np.isfinite(df["Claims"])]
    df["Claims"] = np.log(df["Claims"])
    # df = df[["Claims"]]
    df = (df.reset_index().reset_index()[["level_0", "Claims"]])


    data = torch.tensor(df.values, dtype=torch.float)
    #########################
    # result = prog.generate_function(util.resource('examples/unemploy.bll'), data)

    import math
    def model(month, obs=None):
        slope = pyro.param("slope", torch.tensor(0.))
        # slope = pyro.param("slope", lambda: torch.randn(()))
        intercept = pyro.param("intercept", torch.tensor(13.))
        # intercept = pyro.param("intercept", lambda: torch.randn(()))
        scale = 1.
        # sigma = pyro.param("sigma", lambda: torch.ones(()))
        
        with pyro.plate("data", len(month)):
            return pyro.sample("obs", dist.LogNormal(slope * month + intercept, scale), obs=obs)

    # def model(month, obs=None):
    #     slope = pyro.sample("slope", dist.Normal(0.0, 1.0))
    #     sd = pyro.sample("sd", dist.HalfNormal(1.0))
    #     with pyro.plate("ms_plate", 12):
    #         ms = pyro.sample("ms", dist.Normal(0.0, 1.0))

    #     sms = ms - ms.mean()
        
    #     with pyro.plate("data", len(month)):
    #         return pyro.sample("obs", dist.LogNormal(slope * month + sms.repeat(math.ceil(len(month)/len(sms)))[:len(month)], sd), obs=obs)
        
    auto_guide = pyro.infer.autoguide.AutoNormal(model)
    adam = pyro.optim.Adam({"lr": 0.01})  # Consider decreasing learning rate.
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, auto_guide, adam, elbo)


    smoke_test = ('CI' in os.environ)
    losses = []
    for step in range(2000 if not smoke_test else 2):  # Consider running for more steps.
        loss = svi.step(data[:, 0], data[:, 1])
        losses.append(loss)
            

    predictive = pyro.infer.Predictive(model, guide=auto_guide, num_samples=3000)

    def multi(month):
        global predictive 
        svi_samples = predictive(month)
        svi_obs = svi_samples["obs"]
        return svi_obs

    def single(month):
        global predictive 
        svi_samples = predictive(torch.tensor([month]))
        svi_obs = svi_samples["obs"]
        return svi_obs[:, 0]
    #########################

    prediction = multi(data[:,0])
    # prediction = result.multi(time_data)
    prediction_mean = prediction.mean(0).detach().cpu().numpy() 
    print(prediction_mean)
    print(data[:,1])
    # prediction_lower = prediction.kthvalue(int(len(prediction) * 0.05), dim=0)[0].detach().cpu.numpy()
    # prediction_upper = prediction.kthvalue(int(len(prediction) * 0.95), dim=0)[0].detach().cpu.numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharey=True)
    fig.suptitle("Unemployment claims over time", fontsize=16)

    ax.plot(data[:,0], prediction_mean)
    # ax.fill_between(time_data, prediction_lower, prediction_upper, alpha=0.5)
    ax.plot(data[:,0], data[:,1], "x")
    ax.set(xlabel="Time", ylabel="Unemployment claims", title="Washington State")

    plt.show()