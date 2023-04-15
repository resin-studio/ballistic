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

    DATA_URL = util.resource("data/unemployment_claims.csv")
    df = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
    df = df[df["County/Area"] == "WASHINGTON STATE"][df["Claim Type"] == "Initial claims"]
    df = df[["Date", "Claims"]]
    # df = df.iloc[0:4,:]
    df["Date"] = df["Date"].map(lambda s : d(s).toordinal())
    df["Claims"] = df["Claims"].map(lambda s : float(s))
    df = df[df["Date"] >= d('6/1/08').toordinal()][df["Date"] < d('6/1/19').toordinal()]
    # df = df.reset_index()

    # print(df.head(100))

    print(df.values)

    data_train = torch.tensor(df.values, dtype=torch.float)
    time_data = data_train[:,0]
    claim_data = data_train[:,1]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), sharey=True)
    fig.suptitle("Unemployment claims over time", fontsize=16)

    ax.plot(claim_data, "x")
    ax.set(xlabel="Time", ylabel="Unemployment claims", title="Washington State")

    plt.show()