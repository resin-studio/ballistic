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

if __name__ == "__main__":
    # program = parse_from_file(util.resource('examples/hello.bll'))
    # print('-----------------------------------------------------')
    # print(generate_model_from_ast(program))
    # print('-----------------------------------------------------')

    DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
    raw_data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
    df = raw_data[["cont_africa", "rugged", "rgdppc_2000"]]

    df = df[np.isfinite(df.rgdppc_2000)]
    df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

    data_train = torch.tensor(df.values, dtype=torch.float)

    ########################
    result = prog.generate_function(util.resource('examples/gdp.bll'), data_train)

    ########################
    x1 = (data_train[45,0].item())
    x2 = (data_train[45,1].item())
    # print(result.single(x1, x2))

    ########################
    is_cont_africa = data_train[:,0]
    ruggedness = data_train[:,1]
    log_gdp  = data_train[:,2]

    svi_gdp = result.multi(is_cont_africa, ruggedness)

    predictions = pd.DataFrame({
        "cont_africa": is_cont_africa,
        "rugged": ruggedness,
        "y_mean": svi_gdp.mean(0).detach().cpu().numpy(),
        "y_perc_5": svi_gdp.kthvalue(int(len(svi_gdp) * 0.05), dim=0)[0].detach().cpu().numpy(),
        "y_perc_95": svi_gdp.kthvalue(int(len(svi_gdp) * 0.95), dim=0)[0].detach().cpu().numpy(),
        "true_gdp": log_gdp,
    })
    african_nations = predictions[predictions["cont_africa"] == 1].sort_values(by=["rugged"])
    non_african_nations = predictions[predictions["cont_africa"] == 0].sort_values(by=["rugged"])

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    fig.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)

    ax[0].plot(non_african_nations["rugged"], non_african_nations["y_mean"])
    ax[0].fill_between(non_african_nations["rugged"], non_african_nations["y_perc_5"], non_african_nations["y_perc_95"], alpha=0.5)
    ax[0].plot(non_african_nations["rugged"], non_african_nations["true_gdp"], "o")
    ax[0].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="Non African Nations")

    ax[1].plot(african_nations["rugged"], african_nations["y_mean"])
    ax[1].fill_between(african_nations["rugged"], african_nations["y_perc_5"], african_nations["y_perc_95"], alpha=0.5)
    ax[1].plot(african_nations["rugged"], african_nations["true_gdp"], "o")
    ax[1].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="African Nations");

    plt.show()












    # print(f'''
    # -------------------------
    # {util.resource("bll.tx")}
    # -------------------------
    # {util.project_path()}
    # -------------------------
    # ''')
    # pass

# is_cont_africa, ruggedness => 
# {log_gdp | is_cont_africa, ruggedness, log_gdp : data} ~
#     normal(0., 10.) @ a =>
#     normal(0., 1.) @ b_a =>
#     normal(0., 1.) @ b_r =>
#     normal(0., 1.) @ b_ar =>
#     uniform(0., 10.) @ sigma =>
#     direct(a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness) @ mean =>
#     normal(mean, sigma)