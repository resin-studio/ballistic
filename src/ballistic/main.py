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
    return f'''
def model({", ".join([param.name for param in ast.params])}):
    {generate_model_from_body(input_name, ast.body)}

auto_guide = pyro.infer.autoguide.AutoNormal(model)
adam = pyro.optim.Adam({{"lr": 0.02}})  # Consider decreasing learning rate.
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

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
    if dist.__class__.__name__ == "Normal":
        return f'pyro.sample("{name}", dist.Normal({generate_model_from_expr(dist.mean)}, {generate_model_from_expr(dist.sigma)}))'
    else:
        assert dist.__class__.__name__ == "Direct"
        return f'{generate_model_from_expr(dist.content)}'

def generate_model_from_expr(expr):
    return f'{expr}'

def generate_function(file):
    program_ast = parse_from_file(file)
    python_str = generate_model_from_ast(program_ast)
    exec(python_str)


    d = {}
    exec(python_str, globals(), d)
    return d['predict']


if __name__ == "__main__":
    # program = parse_from_file(util.resource('examples/hello.bll'))
    # print('-----------------------------------------------------')
    # print(generate_model_from_ast(program))
    # print('-----------------------------------------------------')

    predict = generate_function(util.resource('examples/hello.bll'))
    print(predict(5.0))
    # print(f'''
    # -------------------------
    # {util.resource("bll.tx")}
    # -------------------------
    # {util.project_path()}
    # -------------------------
    # ''')
    # pass