from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import util

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
parser = metamodel_from_file(util.resource('grammars/bll.tx'))

def parse_from_file(path):
    return parser.model_from_file(path)

def generate_model_from_ast(ast):
    input_name = next(param.name for param in ast.params)

    training = ''
    if ast.spec:
        input_cols = [
            f'data[:, {ast.spec.args.index(param.name)}]'
            for param in ast.params 
        ]

        output_id = ast.spec.result 
        output_col = f'data[:, {ast.spec.args.index(output_id)}'
        training = f'''
smoke_test = ('CI' in os.environ)
losses = []
for step in range(1000 if not smoke_test else 2):  # Consider running for more steps.
    loss = svi.step({", ".join(input_cols)}, {output_col}])
    losses.append(loss)
        '''

    
    return f'''
def model({", ".join([param.name for param in ast.params])}, obs=None):
    {generate_model_from_body(input_name, ast.body)}

auto_guide = pyro.infer.autoguide.AutoNormal(model)
adam = pyro.optim.Adam({{"lr": 0.02}})  # Consider decreasing learning rate.
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

{training}

predictive = pyro.infer.Predictive(model, guide=auto_guide, num_samples=100)

def multi({", ".join(param.name for param in ast.params)}):
    global predictive 
    svi_samples = predictive({", ".join(param.name for param in ast.params)})
    svi_gdp = svi_samples["obs"]
    return svi_gdp

def single({", ".join(param.name for param in ast.params)}):
    global predictive 
    svi_samples = predictive({", ".join("torch.tensor([" + param.name + "])" for param in ast.params)})
    svi_gdp = svi_samples["obs"]
    return svi_gdp[:, 0]
    '''

def generate_model_from_body(input_name, body):
    if body.__class__.__name__ == "Bind":
        return f'''
    {body.name} = {generate_model_from_dist(body.name, body.src)}
    {generate_model_from_body(input_name, body.dst)}
        '''
    else: 
        return f'''
    with pyro.plate("data", len({input_name})):
        return {generate_model_from_dist("obs", body)}
        '''

def generate_model_from_dist(name, dist):
    obs_str = ', obs=obs' if name == "obs" else '' 
    if dist.__class__.__name__ == "Normal":
        return f'pyro.sample("{name}", dist.Normal({generate_model_from_expr(dist.mean)}, {generate_model_from_expr(dist.sigma)}){obs_str})'
    elif dist.__class__.__name__ == "Uniform":
        return f'pyro.sample("{name}", dist.Uniform({generate_model_from_expr(dist.mean)}, {generate_model_from_expr(dist.sigma)}){obs_str})'
    else:
        assert dist.__class__.__name__ == "Direct"
        return f'{generate_model_from_expr(dist.content)}'

def generate_model_from_expr(expr):
    return f'{expr}'

@dataclass(frozen=True, eq=True)
class Stoch:
    multi : Callable 
    single : Callable 


def generate_function(file, data=None):
    program_ast = parse_from_file(file)
    python_str = generate_model_from_ast(program_ast)
    print('------------------------')
    print(python_str)
    print('------------------------')
    d = {'data' : data}
    exec(python_str, globals(), d)
    return Stoch(multi = d['multi'], single=d['single'])


if __name__ == "__main__":
    # program = parse_from_file(util.resource('examples/hello.bll'))
    # print('-----------------------------------------------------')
    # print(generate_model_from_ast(program))
    # print('-----------------------------------------------------')

    DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
    data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
    df = data[["cont_africa", "rugged", "rgdppc_2000"]]

    df = df[np.isfinite(df.rgdppc_2000)]
    df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

    train = torch.tensor(df.values, dtype=torch.float)

    ########################
    result = generate_function(util.resource('examples/hello.bll'), train)

    ########################
    x1 = (train[45,0].item())
    x2 = (train[45,1].item())
    # print(result.single(x1, x2))

    ########################
    is_cont_africa = train[:,0]
    ruggedness = train[:,1]
    log_gdp  = train[:,2]

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