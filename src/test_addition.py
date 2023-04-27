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


    data_train = torch.tensor([
        [1, 1, 2],
        [1, 2, 3],
        [2, 3, 5],
        [4, 7, 11],
        [5, 5, 10],
        [20, 24, 44],
        [4, 1, 5],
        [3, 6, 8],
        [12, 50, 62],
    ]) * 1.

    result = prog.generate_function(util.resource('examples/addition_spec.bll'), data_train)