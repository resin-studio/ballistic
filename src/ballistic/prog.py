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

def generate_training_from_spec(spec):
    if not spec:
        return f'''
        '''
    else:
        output_id = spec.result 
        output_col = spec.args.index(output_id)

        return f'''
auto_guide = pyro.infer.autoguide.AutoNormal(model)
adam = pyro.optim.Adam({{"lr": 0.02}})  # Consider decreasing learning rate.
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

        '''

def generate_model_from_ast(ast):
    input_name = next(param.name for param in ast.params)

    input_cols = [
        f'data[:, {ast.spec.args.index(param.name)}]'
        for param in ast.params 
    ]

    output_id = ast.spec.result 
    output_col = f'data[:, {ast.spec.args.index(output_id)}'

    training = ''
    if ast.spec:
        training = f'''
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

def predict({", ".join(param.name for param in ast.params)}):
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

def generate_function(file, data=None):
    program_ast = parse_from_file(file)
    python_str = generate_model_from_ast(program_ast)
    d = {'data' : data}
    exec(python_str, globals(), d)
    return d['predict']


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

    # print(train)

    # hello = generate_function(util.resource('examples/hello.bll'), train)
    # print(hello(5.0))
    # print(f'''
    # -------------------------
    # {util.resource("bll.tx")}
    # -------------------------
    # {util.project_path()}
    # -------------------------
    # ''')
    # pass