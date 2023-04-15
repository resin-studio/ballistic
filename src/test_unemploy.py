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
    # print(d('1/1/05').toordinal())
    # print(d('1/2/05').toordinal())

    # DATA_URL = util.resource("data/unemployment_claims.csv")
    # df = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
    # df = df[df["County/Area"] == "WASHINGTON STATE"][df["Claim Type"] == "Initial claims"]
    # df["Date"] = df["Date"].map(lambda s : d(s).toordinal())
    # df = df[df["Date"] >= d('6/1/08').toordinal()][df["Date"] < d('6/1/19').toordinal()]
    # df["Date"] = df["Date"].map(lambda n : n - d('6/1/08').toordinal())
    # df["Claims"] = df["Claims"].map(lambda s : float(s))
    # df = df[["Date", "Claims"]]

    # # df = df.iloc[0:4,:]
    # # print(df.head(4))
    # # print(df.values)

    # data_train = torch.tensor(df.values, dtype=torch.float)

    #############################

    print('-----------------------------------------------------')
    program = prog.parse_from_file(util.resource('examples/unemploy.bll'))
    print(program)
    print('-----------------------------------------------------')
    print(prog.generate_model_from_ast(program))
    print('-----------------------------------------------------')
    # result = prog.generate_function(util.resource('examples/unemploy.bll'), data_train)
    #############################
    ## GENERATED CODE EXECUTION TEST

    # import pyro
    # import pyro.distributions as dist
    # import math
    # from math import ceil, floor
    
    # def mean(o): return o.mean()
    # def prob_sgt():
    #     N = 100 
    #     slope = pyro.sample('slope', dist.Normal(0.0, 1.0))
    #     obs_sd = pyro.sample('obs_sd', dist.HalfNormal(1.0))
    #     with pyro.plate('S', 52):
    #         seasonality = pyro.sample('seasonality', dist.Normal(0.0, 1.0))
    #         seasonality = seasonality - seasonality.mean()
    #     with pyro.plate('N', N) as time:
    #         log_y_hat = slope * time + seasonality.repeat(math.ceil(N / 52))[:N]
    #         return pyro.sample('y', dist.LogNormal(log_y_hat, obs_sd))
    # print(prob_sgt())

    # rewrite (seasonality[time]) into seasonality.repeat(math.ceil(len(time) / len(seasonality)))[:len(time)]

    #############################

    # # print(data_train)
    # time_data = data_train[:,0]
    # claim_data = data_train[:,1]

    # prediction = result.multi(time_data)
    # prediction_mean = prediction.mean(0).detach().cpu().numpy() 
    # prediction_lower = prediction.kthvalue(int(len(prediction) * 0.05), dim=0)[0].detach().cpu.numpy()
    # prediction_upper = prediction.kthvalue(int(len(prediction) * 0.95), dim=0)[0].detach().cpu.numpy()

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharey=True)
    # fig.suptitle("Unemployment claims over time", fontsize=16)

    # ax.plot(time_data, prediction_mean)
    # ax.fill_between(time_data, prediction_lower, prediction_upper, alpha=0.5)
    # ax.plot(time_data, claim_data, "x")
    # ax.set(xlabel="Time", ylabel="Unemployment claims", title="Washington State")

    # plt.show()