from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

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

from math import ceil, floor
def mean(o): return o.mean()

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
    svi_obs = svi_samples["obs"]
    return svi_obs

def single({", ".join(param.name for param in ast.params)}):
    global predictive 
    svi_samples = predictive({", ".join("torch.tensor([" + param.name + "])" for param in ast.params)})
    svi_obs = svi_samples["obs"]
    return svi_obs[:, 0]
    '''

def generate_model_from_body(input_name, body):
    if body.__class__.__name__ == "Sample":
        return f'''
    {body.name} = {generate_model_from_dist(body.name, body.src)}
    {generate_model_from_body(input_name, body.contin)}
        '''

    elif body.__class__.__name__ == "Plate":
        return f'''
    with pyro.plate("{body.name}", {body.size}):
        {body.name} = {generate_model_from_dist(body.name, body.src)}
    {generate_model_from_body(input_name, body.contin)}
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
    elif dist.__class__.__name__ == "Lognorm":
        return f'pyro.sample("{name}", dist.LogNormal({generate_model_from_expr(dist.mean)}, {generate_model_from_expr(dist.sigma)}){obs_str})'
    elif dist.__class__.__name__ == "Uniform":
        return f'pyro.sample("{name}", dist.Uniform({generate_model_from_expr(dist.mean)}, {generate_model_from_expr(dist.sigma)}){obs_str})'
    elif dist.__class__.__name__ == "Halfnorm":
        return f'pyro.sample("{name}", dist.HalfNormal({generate_model_from_expr(dist.scale)}){obs_str})'
    else:
        assert dist.__class__.__name__ == "Direct"
        return f'{generate_model_from_expr(dist.content)}'

def generate_model_from_expr(expr):
    base_str = generate_model_from_prod(expr.base)
    exts_str = ''
    for ext in expr.exts:
        exts_str += (f' {ext.op} ' + generate_model_from_prod(ext.arg))
        
    return base_str + exts_str 

def generate_model_from_prod(prod):
    base_str = generate_model_from_atom(prod.base)
    factors_str = ''
    for factor in prod.factors:
        factors_str += (f' {factor.op} ' + generate_model_from_atom(factor.arg))
        
    return base_str + factors_str 

def generate_model_from_atom(atom):
    if atom.__class__.__name__ == "Paren":
        return '(' + generate_model_from_expr(atom.content) + ')'
    elif atom.__class__.__name__ == "Mean":
        return 'mean(' + generate_model_from_expr(atom.vector) + ')'
    elif atom.__class__.__name__ == "Project":
        v = generate_model_from_expr(atom.vector)
        i = generate_model_from_expr(atom.index)
        return f'{v}.repeat(math.ceil(len({i}) / len({v})))[:len({i})]'
    else:
        return f'{atom}'

@dataclass(frozen=True, eq=True)
class Stoch:
    multi : Callable 
    single : Callable 


def generate_function(file, data=None):
    program_ast = parse_from_file(file)
    python_str = generate_model_from_ast(program_ast)
    print('--- Generated Python: Start ---')
    print(python_str)
    print('--- Generated Python: End -----')
    d = {'data' : data}
    exec(python_str, globals(), d)
    return Stoch(multi = d['multi'], single=d['single'])

