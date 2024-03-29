from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence, Any


from ballistic import util

import logging
import os

import torch
from torch import Tensor, tensor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints



import math
from math import e

import z3
from z3 import *
set_option(precision = 5)

from textx import metamodel_from_file

parser = metamodel_from_file(util.resource('grammars/bll.tx'))

def parse_from_file(path):
    return parser.model_from_file(path)

def parse_from_str(code):
    return parser.model_from_str(code)


class Compiler:

    def from_ast(self, ast):
        assert ast.body

        input_name = next(param for param in ast.params)

        training = ''
        if ast.spec:
            input_cols = [
                f'data[:, {ast.spec.args.index(param)}]'
                for param in ast.params 
            ]

            output_id = ast.spec.result 
            outcol = f'data[:, {ast.spec.args.index(output_id)}'
            training = f'''
    smoke_test = ('CI' in os.environ)
    losses = []
    for step in range(1000 if not smoke_test else 2):  # Consider running for more steps.
        loss = svi.step({", ".join(input_cols)}, {outcol}])
        losses.append(loss)
            '''

    #     return f'''
    # def model({", ".join([param for param in ast.params])}, obs=None):
    #     {from_body(input_name, ast.body)}

    # auto_guide = pyro.infer.autoguide.AutoNormal(model)
    # adam = pyro.optim.Adam({{"lr": 0.02}})  # Consider decreasing learning rate.
    # elbo = pyro.infer.Trace_ELBO()
    # svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

    # {training}

    # predictive = pyro.infer.Predictive(model, guide=auto_guide, num_samples=600)

        
        return f'''
def model({", ".join([param for param in ast.params])}, obs=None):
    {self.from_body(input_name, ast.body)}

predictive = None
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
try:
    auto_guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)
    svi = pyro.infer.SVI(model, auto_guide, pyro.optim.Adam({{"lr": 0.01}}), loss = pyro.infer.Trace_ELBO())
    smoke_test = ('CI' in os.environ)
    pyro.clear_param_store()
    {training}
    with pyro.plate("samples", 800, dim=-2):
        posterior_samples = auto_guide(data[:,0])

    assert "obs" not in posterior_samples 
    predictive = pyro.infer.Predictive(model, posterior_samples=posterior_samples)

except:
    auto_guide = pyro.infer.autoguide.AutoNormal(model)
    auto_guide = pyro.infer.autoguide.AutoNormal(model)
    adam = pyro.optim.Adam({{"lr": 0.02}})  # Consider decreasing learning rate.
    elbo = pyro.infer.Trace_ELBO()
    svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

    {training}

    predictive = pyro.infer.Predictive(model, guide=auto_guide, num_samples=600)

finally:
    pass

def multi({", ".join(param for param in ast.params)}):
    global predictive 
    assert predictive != None
    svi_samples = predictive({", ".join(param for param in ast.params)})
    svi_obs = svi_samples["obs"]
    return svi_obs

def single({", ".join(param for param in ast.params)}):
    global predictive 
    assert predictive != None
    svi_samples = predictive({", ".join("torch.tensor([" + param + "])" for param in ast.params)})
    svi_obs = svi_samples["obs"]
    return svi_obs[:, 0]
        '''

    def from_body(self, input_name, body):
        if body.__class__.__name__ == "Sample":
            return f'''
    {body.name} = {self.from_dist(body.name, body.src, input_name)}
    {self.from_body(input_name, body.contin)}
            '''

        elif body.__class__.__name__ == "Plate":
            return f'''
    with pyro.plate("{body.name}_plate", {body.size}):
        {body.name} = {self.from_dist(body.name, body.src, input_name)}
    {self.from_body(input_name, body.contin)}
            '''
        elif body.__class__.__name__ == "Final":
            return f'''
    with pyro.plate("data", len({input_name})):
        return {self.from_dist("obs", body.content, input_name)}
            '''

    def from_dist(self, name, dist, input_name):
        obs_str = ', obs=obs' if name == "obs" else '' 
        if dist.__class__.__name__ == "Normal":
            return f'pyro.sample("{name}", dist.Normal({self.from_expr(dist.mean, input_name)}, {self.from_expr(dist.sigma, input_name)}){obs_str})'
        elif dist.__class__.__name__ == "Lognorm":
            return f'pyro.sample("{name}", dist.LogNormal({self.from_expr(dist.mean, input_name)}, {self.from_expr(dist.sigma, input_name)}){obs_str})'
        elif dist.__class__.__name__ == "Uniform":
            return f'pyro.sample("{name}", dist.Uniform({self.from_expr(dist.low, input_name)}, {self.from_expr(dist.high, input_name)}){obs_str})'
        elif dist.__class__.__name__ == "Halfnorm":
            return f'pyro.sample("{name}", dist.HalfNormal({self.from_expr(dist.scale, input_name)}){obs_str})'
        else:
            assert dist.__class__.__name__ == "Direct"
            v = f'{self.from_expr(dist.content, input_name)}'
            if name == "obs":
                return f'pyro.sample("{name}", dist.Normal(torch.nan_to_num(1. * ({v}), 0), 0.01){obs_str})'
            else:
                return v 

    def from_expr(self, expr, input_name):
        base_str = self.from_prod(expr.base, input_name)
        exts_str = ''
        for ext in expr.exts:
            if ext.__class__.__name__ == "Plus":
                exts_str += (' + ' + self.from_prod(ext.arg, input_name))
            elif ext.__class__.__name__ == "Minus":
                exts_str += (' - ' + self.from_prod(ext.arg, input_name))
            else:
                assert False
            
            
        return base_str + exts_str 

    def from_prod(self, prod, input_name):
        base_str = self.from_atom(prod.base, input_name)
        factors_str = ''
        for factor in prod.factors:
            if factor.__class__.__name__ == "Mul":
                factors_str += (' * ' + self.from_atom(factor.arg, input_name))
            elif factor.__class__.__name__ == "Div":
                factors_str += (' / ' + self.from_atom(factor.arg, input_name))
            else:
                assert False
            
        return base_str + factors_str 

    def from_atom(self, atom, input_name):
        if atom.__class__.__name__ == "Paren":
            return '(' + self.from_expr(atom.content, input_name) + ')'
        elif atom.__class__.__name__ == "Mean":
            return atom.vector + '.mean()'
        elif atom.__class__.__name__ == "Align":
            v = atom.vector
            return f'{v}.repeat(math.ceil(len({input_name})/len({v})))[:len({input_name})]'
        else:
            return f'{atom}'
### End Compiler ###

@dataclass(frozen=True, eq=True)
class Stoch:
    multi : Callable 
    single : Callable 


def learn_posteriors(program_ast, data):
    compiler = Compiler()
    python_str = compiler.from_ast(program_ast)
    print('--- compiled Python: Start ---')
    print(python_str)
    print('--- compiled Python: End -----')
    d = {'data' : data}
    exec(python_str, globals(), d)
    return Stoch(multi = d['multi'], single=d['single'])

from typing import TypeVar, Generic
T = TypeVar('T')

### Internal Choice Tree Structure

class Choices(Generic[T], dict[BoolRef, T]): pass


@dataclass(frozen=True, eq=True)
class SampleTree:
    name : str 
    src : Choices[DistroOption]
    contin : Choices[BodyOption] 

@dataclass(frozen=True, eq=True)
class PlateTree:
    name : str 
    size : int
    src : Choices[DistroOption]
    contin : Choices[BodyOption] 

@dataclass(frozen=True, eq=True)
class FinalTree:
    content : Choices[DistroOption] 

BodyOption = SampleTree | PlateTree | FinalTree

@dataclass(frozen=True, eq=True)
class NormalTree:
    mean : ExprTree 
    sigma : ExprTree 

@dataclass(frozen=True, eq=True)
class LognormTree:
    mean : ExprTree 
    sigma : ExprTree 

@dataclass(frozen=True, eq=True)
class UniformTree:
    low : ExprTree 
    high : ExprTree 

@dataclass(frozen=True, eq=True)
class HalfnormTree:
    scale : ExprTree 

@dataclass(frozen=True, eq=True)
class DirectTree:
    content : ExprTree 

DistroOption = NormalTree | LognormTree | UniformTree | HalfnormTree | DirectTree


@dataclass(frozen=True, eq=True)
class NilExtTree:
    pass

@dataclass(frozen=True, eq=True)
class ConsExtTree:
    head : Choices[ExtOption]
    tail : Choices[ExtsOption]

ExtsOption = NilExtTree | ConsExtTree 

@dataclass(frozen=True, eq=True)
class ExprTree:
    base : ProdTree
    exts : Choices[ExtsOption] 

@dataclass(frozen=True, eq=True)
class PlusTree:
    arg : ProdTree

@dataclass(frozen=True, eq=True)
class MinusTree:
    arg : ProdTree

ExtOption = PlusTree | MinusTree

@dataclass(frozen=True, eq=True)
class NilFactorTree:
    pass

@dataclass(frozen=True, eq=True)
class ConsFactorTree:
    head : Choices[FactorOption]
    tail : Choices[FactorsOption]

FactorsOption = NilFactorTree | ConsFactorTree 

@dataclass(frozen=True, eq=True)
class ProdTree:
    base : Choices[AtomOption] 
    factors : Choices[FactorsOption]

@dataclass(frozen=True, eq=True)
class MulTree:
    arg : Choices[AtomOption] 

@dataclass(frozen=True, eq=True)
class DivTree:
    arg : Choices[AtomOption]

FactorOption = MulTree | DivTree

@dataclass(frozen=True, eq=True)
class ParenTree:
    content : ExprTree 

@dataclass(frozen=True, eq=True)
class MeanTree:
    plate : str 

@dataclass(frozen=True, eq=True)
class AlignTree:
    plate : str 

@dataclass(frozen=True, eq=True)
class IdTree:
    content : str 

@dataclass(frozen=True, eq=True)
class FloatTree:
    content : Any 

AtomOption = ParenTree | MeanTree | AlignTree | IdTree | FloatTree

### End Internal Tree Structure


# @dataclass(frozen=True, eq=True)
# class ChoiceTree:
#     label: str
#     choices : dict[str, ProductionTree] 

# @dataclass(frozen=True, eq=True)
# class ProductionTree:
#     label: str
#     productions : dict[str, ChoiceTree | ProductionTree | list[ChoiceTree] | Any | str | int] 


def extract_fields(labels, fields, data : Tensor) -> Tensor:
    return data[:, [fields.index(label) for label in labels]]

def extract_field(label, fields, data : Tensor) -> Tensor:
    return data[:, fields.index(label)]

@dataclass(frozen=True, eq=True)
class Column(Generic[T]):
    mean : list[T]
    align : list[T]


@dataclass(frozen=True, eq=True)
class SearchSpace(Generic[T]):
    # 0: a tree with choices identified by control variables
    tree: Choices[T] 
    # 1: a description length 
    dlen: Any 
    # 2: an output column as an smt formula built form control variables and input data
    outcol: Column[Any] 
    # 3: constraints 
    constraints : list[Probe | BoolRef]

@dataclass(frozen=True, eq=True)
class SubSpace(Generic[T]):
    # 0: a tree with choices identified by control variables
    tree: T 
    # 1: a description length 
    dlen: Any 
    # 2: an output column as an smt formula built form control variables and input data
    outcol: Column[Any] 
    # 3
    constraints : list[Probe | BoolRef]

def not_pairs(bs) -> list[Probe | BoolRef]:
    return [(Not(And(b1, b2))) 
            for i, b1 in enumerate(bs)
            for j, b2 in enumerate(bs)
            if i < j]


class Spacer:

    def __init__(self, params : list[str], input_data):
        self.input_data = input_data
        self.params = params

        self.var_count = 0
        self.weight_count = 0

        self.plate_base_count = 0
        self.log_count = 0
        self.control_count = 0
        self.body_fuel = 1 
        self.expr_fuel = 2 
        self.plate_size_max = 0 
        self.exts_fuel = 2 
        self.factors_fuel = 0 

    def fresh_plate_base(self) -> str:
        plate_base = f'_vec_{self.plate_base_count}'
        self.plate_base_count = self.plate_base_count + 1
        return plate_base 

    def fresh_weight(self) -> Any:
        w = Real(f'_w_{self.weight_count}')
        self.weight_count = self.weight_count + 1
        return w 
        
    def fresh_var(self) -> str:
        var = f'_g_{self.var_count}'
        self.var_count = self.var_count + 1
        return var

    def fresh_control(self) -> BoolRef:
        c = Bool(f'_c_{self.control_count}')
        self.control_count = self.control_count + 1
        return c 

    def combine(self, spaces : list[SubSpace[T]]) -> SearchSpace[T]:

        control_spaces = [(self.fresh_control(), space) for space in spaces]

        choices = Choices() 
        constraint = BoolVal(False)
        dlr = RealVal(0) 
        for control, space in control_spaces:
            choices[control] = space.tree
            dlr = If(control, space.dlen, dlr)
            constraint = If(control, And(space.constraints), constraint)

        # out_constraints : list[Probe | BoolRef] = [
        #     Implies(control, And(space.out_constraints))
        #     for control, space in control_spaces
        #     if len(space.out_constraints) > 0
        # ]
        
        dlen = RealVal(math.log(len(spaces))) + dlr

        outcol_mean : list[Any] = []
        for i in range(len(self.input_data)):
            out_mean = RealVal(0) 
            for (control, space) in control_spaces:
                out_mean = If(control, space.outcol.mean[i], out_mean)
            outcol_mean.append(out_mean)

        assert len(outcol_mean) == len(self.input_data)

        outcol_align : list[Any] = []
        some_align_col = next((True for space in spaces if len(space.outcol.align) > 0), False)
        if some_align_col:
            for i in range(len(self.input_data)):
                out_align = RealVal(0) 
                for (control, space) in control_spaces:
                    if len(space.outcol.align) > 0:
                        out_align = If(control, space.outcol.align[i], out_align)
                outcol_align.append(out_align)

        outcol = Column(outcol_mean, outcol_align)

        controls = [control for (control, _) in control_spaces]
        constraints : list[Any] = not_pairs(controls) + [constraint]

        return SearchSpace(choices, dlen, outcol, constraints)


    def to_body(self, plates_in_scope : dict[str, Column], vars_in_scope : dict[str, Column]) -> SearchSpace[BodyOption]:
        spaces : list[SubSpace[BodyOption]] = []

        if self.body_fuel > 0: 
            self.body_fuel -= 1

            # def scope_sample():
            #     src = self.to_dist(plates_in_scope, vars_in_scope)   
            #     id = self.fresh_var() 

            #     vars_in_scope_ = vars_in_scope.copy() 
            #     vars_in_scope_[id] = src.outcol 
            #     contin : SearchSpace = self.to_body(plates_in_scope, vars_in_scope_)

            #     spaces.append(SubSpace(
            #         tree = SampleTree(
            #             name = id,
            #             src = src.tree,
            #             contin = contin.tree
            #         ),     
            #         dlen = src.dlen + contin.dlen,
            #         outcol = contin.outcol,
            #         constraints = src.constraints + contin.constraints
            #     ))
            # scope_sample()

            # def scope_plates():
            #     if self.plate_size_max <= 1:
            #         return

            #     plate_sizes = range(1, self.plate_size_max + 1)
            #     for plate_size in plate_sizes:

            #         src = self.to_dist(plates_in_scope, vars_in_scope, plate_size)   

            #         plate_base = self.fresh_plate_base()

            #         plates_in_scope_ = plates_in_scope.copy() 
            #         plates_in_scope_[plate_base] = src.outcol 

            #         contin : SearchSpace = self.to_body(plates_in_scope_, vars_in_scope)

            #         spaces.append(SubSpace(
            #             tree = PlateTree(
            #                 name = plate_base,
            #                 size = plate_size,
            #                 src = src.tree,
            #                 contin = contin.tree,
            #             ),
            #             dlen = src.dlen + contin.dlen,
            #             outcol = contin.outcol,
            #             constraints = src.constraints + contin.constraints
            #         ))
            # scope_plates()

        def scope_final():

            content = self.to_dist(plates_in_scope, vars_in_scope)   

            spaces.append(SubSpace( 
                tree = FinalTree(
                    content = content.tree,
                ),
                dlen = content.dlen,
                outcol = content.outcol,
                constraints = content.constraints
            ))
        scope_final()

        # Combine
        return self.combine(spaces)

    def fresh_column(self, plate_size : int):
        outcol_mean = [self.fresh_weight() for _ in  self.input_data]
        outcol_align = []


        if plate_size > 0:
            plate_weight_matrix = [[self.fresh_weight() for _ in range(plate_size)] for _ in self.input_data]
            outcol_mean = [] 
            for plate_row in plate_weight_matrix: 
                mean_weight = Sum(plate_row) / RealVal(len(plate_row))
                outcol_mean.append(mean_weight)

            outcol_align = []
            for col_idx in range(plate_size): 
                plate_col = []
                for plate_weights in plate_weight_matrix: 
                    plate_col.append(plate_weights[col_idx])
                plate_col_mean = Sum(plate_col) / RealVal(len(plate_col))
                outcol_align.append(plate_col_mean)
            outcol_align = (outcol_align * (math.ceil(len(self.input_data)/plate_size)))[:len(self.input_data)]

        return Column(outcol_mean, outcol_align)

    def to_dist(self, 
                plates_in_scope : dict[str, Column], 
                vars_in_scope : dict[str, Column],
                plate_size : int = 1, 
            ) -> SearchSpace[DistroOption]:

        spaces : list[SubSpace[DistroOption]] = []
        
        def scope_normal():
            outcol = self.fresh_column(plate_size)
            mean = self.to_expr(plates_in_scope, vars_in_scope)
            sigma = self.to_expr(plates_in_scope, vars_in_scope)

            constraints = []
            for out, mean_out, sigma_out in zip(outcol.mean, mean.outcol.mean, sigma.outcol.mean): 
                lower = mean_out - 3 * sigma_out 
                upper = mean_out + 3 * sigma_out
                constraints.append(And([sigma_out > 0, lower <= out, out <= upper]))

            spaces.append(SubSpace(
                tree = NormalTree(
                    mean = mean.tree,
                    sigma = sigma.tree
                ),   
                dlen = mean.dlen + sigma.dlen,
                outcol = outcol,
                constraints = mean.constraints + sigma.constraints + constraints
            ))
        scope_normal()

        def scope_lognorm():
            outcol = self.fresh_column(plate_size)
            mean = self.to_expr(plates_in_scope, vars_in_scope)
            sigma = self.to_expr(plates_in_scope, vars_in_scope)

            constraints = []
            for out, mean_out, sigma_out in zip(outcol.mean, mean.outcol.mean, sigma.outcol.mean): 
                sigma_sq = sigma_out ** 2 

                mode = 2.7 ** (mean_out - sigma_sq)
                lmean = 2.7 ** (mean_out + sigma_sq/2)

                esig_sq = 2.7 ** sigma_sq
                skewness = (esig_sq + 2) * (esig_sq - 1) ** (1/2)

                lower = mode / 2
                upper = lmean + skewness
                constraints.append(
                    # And(sigma_out > 0, lower < upper, lower <= out, out <= upper)
                    And(sigma_out > 0, mean_out / 2 <= out,  out <= mean_out * 4)
                )

            spaces.append(SubSpace(
                tree = LognormTree(
                    mean = mean.tree,
                    sigma = sigma.tree
                ),   
                dlen = mean.dlen + sigma.dlen,
                outcol = outcol,
                constraints = mean.constraints + sigma.constraints + constraints
            ))
        scope_lognorm()

        def scope_uniform():
            outcol = self.fresh_column(plate_size)
            low = self.to_expr(plates_in_scope, vars_in_scope)
            high = self.to_expr(plates_in_scope, vars_in_scope)

            constraints = []
            for out, low_out, high_out in zip(outcol.mean, low.outcol.mean, high.outcol.mean): 
                constraints.append(
                    And(low_out < high_out, low_out <= out, out <= high_out)
                )

            spaces.append(SubSpace(
                tree = UniformTree(
                    low = low.tree,
                    high = high.tree
                ), 
                dlen = low.dlen + high.dlen,
                outcol = outcol,
                constraints = low.constraints + high.constraints + constraints
            ))
        scope_uniform()


        def scope_halfnorm():
            outcol = self.fresh_column(plate_size)
            scale = self.to_expr(plates_in_scope, vars_in_scope)

            constraints = []
            for out, scl in zip(outcol.mean, scale.outcol.mean): 
                lower = RealVal(0)
                upper = 3 * scl 
                constraints.append(And(scl > 0, lower <= out, out <= upper))


            spaces.append(SubSpace(
                tree = HalfnormTree(
                    scale = scale.tree,
                ),   
                dlen = scale.dlen,
                outcol = outcol,
                constraints = scale.constraints + constraints
            ))
        scope_halfnorm()

        def scope_direct():
            outcol = self.fresh_column(plate_size)
            content = self.to_expr(plates_in_scope, vars_in_scope)
            
            constraints = [
                v == out
                for out, v in zip(outcol.mean, content.outcol.mean)
            ] + content.constraints 

            spaces.append(SubSpace(
                tree = DirectTree(
                    content = content.tree,
                ),   
                dlen = content.dlen,
                outcol = outcol,
                # outcol = content.outcol,
                constraints = constraints,
            ))
        scope_direct()

        # Combine

        return self.combine(spaces)


    def to_exts(self, plates_in_scope : dict[str, Column], 
                vars_in_scope : dict[str, Column]) -> SearchSpace[ExtsOption]:

        spaces : list[SubSpace[ExtsOption]] = []

        def scope_nil():
            spaces.append(SubSpace(
                tree = NilExtTree(), 
                dlen = RealVal(1),
                outcol = Column(
                    [RealVal(0)] * len(self.input_data), 
                    []
                ),
                constraints = [],
            ))
        scope_nil()

        if self.exts_fuel > 0: 
            self.exts_fuel -= 1

            def scope_cons_ext():
                head = self.to_ext(plates_in_scope, vars_in_scope)
                tail = self.to_exts(plates_in_scope, vars_in_scope)


                assert len(head.outcol.mean) == len(tail.outcol.mean)
                outcol_mean = [h + t for h,t in zip(head.outcol.mean, tail.outcol.mean)]
                outcol_align = [] 
                if len(head.outcol.align) > 0 and len(tail.outcol.align) > 0 : 
                    outcol_align = [h + t for h,t in zip(head.outcol.align, tail.outcol.align)]
                elif len(head.outcol.align) > 0:
                    outcol_align = head.outcol.align
                elif len(tail.outcol.align) > 0 : 
                    outcol_align = tail.outcol.align


                spaces.append(SubSpace(
                    tree = ConsExtTree(
                        head = head.tree,
                        tail = tail.tree
                    ), 
                    dlen = RealVal(1),
                    outcol = Column(
                        outcol_mean, outcol_align
                    ),
                    constraints = head.constraints + tail.constraints, 
                ))
            scope_cons_ext()

        return self.combine(spaces)

    def to_expr(self, plates_in_scope : dict[str, Column], 
                vars_in_scope : dict[str, Column]) -> SubSpace[ExprTree]:
        base = self.to_prod(plates_in_scope, vars_in_scope)
        exts = self.to_exts(plates_in_scope, vars_in_scope)



        outcol_mean = [h + t for h,t in zip(base.outcol.mean, exts.outcol.mean)]
        outcol_align = [] 
        if len(base.outcol.align) > 0 and len(exts.outcol.align) > 0 : 
            outcol_align = [h + t for h,t in zip(base.outcol.align, exts.outcol.align)]
        elif len(base.outcol.align) > 0:
            outcol_align = base.outcol.align
        elif len(exts.outcol.align) > 0 : 
            outcol_align = exts.outcol.align

        outcol = Column(outcol_mean, outcol_align)


        dlen = base.dlen + exts.dlen

        tree = ExprTree(
            base = base.tree,
            exts = exts.tree
        )


        return SubSpace(tree, 
                        dlen, outcol, 
                        base.constraints + exts.constraints)

    def to_factors(self, plates_in_scope : dict[str, Column], 
                vars_in_scope : dict[str, Column]) -> SearchSpace[FactorsOption]:

        spaces : list[SubSpace[FactorsOption]] = []

        def scope_nil():
            spaces.append(SubSpace(
                tree = NilFactorTree(), 
                dlen = RealVal(1),
                outcol = Column(
                    [RealVal(1)] * len(self.input_data), 
                    []
                ),
                constraints = []
            ))
        scope_nil()


        if self.factors_fuel > 0: 
            self.factors_fuel -= 1

            def scope_cons():
                head = self.to_factor(plates_in_scope, vars_in_scope)
                tail = self.to_factors(plates_in_scope, vars_in_scope)


                outcol_mean = [h * t for h,t in zip(head.outcol.mean, tail.outcol.mean)]
                outcol_align = [] 
                if len(head.outcol.align) > 0 and len(tail.outcol.align) > 0 : 
                    outcol_align = [h * t for h,t in zip(head.outcol.align, tail.outcol.align)]
                elif len(head.outcol.align) > 0:
                    outcol_align = head.outcol.align
                elif len(tail.outcol.align) > 0 : 
                    outcol_align = tail.outcol.align

                spaces.append(SubSpace(
                    tree = ConsFactorTree(
                        head = head.tree,
                        tail = tail.tree
                    ), 
                    dlen = RealVal(1),
                    outcol = Column(
                        outcol_mean, outcol_align
                    ),
                    constraints=head.constraints + tail.constraints
                ))
            scope_cons()

        return self.combine(spaces)

    def to_prod(self, 
                plates_in_scope : dict[str, Column], 
                vars_in_scope : dict[str, Column]
    ) -> SubSpace[ProdTree]:

        base = self.to_atom(plates_in_scope, vars_in_scope)
        factors = self.to_factors(plates_in_scope, vars_in_scope)

        outcol_mean = [h + t for h,t in zip(base.outcol.mean, factors.outcol.mean)]
        outcol_align = [] 
        if len(base.outcol.align) > 0 and len(factors.outcol.align) > 0 : 
            outcol_align = [h * t for h,t in zip(base.outcol.align, factors.outcol.align)]
        elif len(base.outcol.align) > 0:
            outcol_align = base.outcol.align
        elif len(factors.outcol.align) > 0 : 
            outcol_align = factors.outcol.align

        outcol = Column(outcol_mean, outcol_align)


        dlen = base.dlen + factors.dlen

        tree = ProdTree(
            base = base.tree,
            factors = factors.tree
        )

        return SubSpace(tree, dlen, outcol, 
                    base.constraints + factors.constraints
                    )

    def to_ext(self, 
               plates_in_scope : dict[str, Column], 
               vars_in_scope : dict[str, Column]) -> SearchSpace[ExtOption]:
        spaces : list[SubSpace[ExtOption]] = []

        def scope_plus():
            arg = self.to_prod(plates_in_scope, vars_in_scope)
            spaces.append(SubSpace(
                tree = PlusTree(
                    arg = arg.tree
                ), 
                dlen = arg.dlen,
                outcol = arg.outcol,
                constraints = arg.constraints
            ))
        scope_plus()

        def scope_minus():
            arg = self.to_prod(plates_in_scope, vars_in_scope)
            spaces.append(SubSpace(
                tree = MinusTree(
                    arg = arg.tree
                ),   
                dlen = arg.dlen ,
                outcol = Column(
                    [-1 * output for output in arg.outcol.mean], 
                    []
                ),
                constraints=arg.constraints
            ))
        scope_minus()

        # Combine
        return self.combine(spaces)

    def to_factor(self, 
        plates_in_scope : dict[str, Column], 
        vars_in_scope : dict[str, Column]) -> SearchSpace[FactorOption]:


        spaces : list[SubSpace[FactorOption]] = []

        def scope_mul():
            arg = self.to_atom(plates_in_scope, vars_in_scope)
            spaces.append(SubSpace(
                tree = MulTree(
                    arg = arg.tree
                ),   
                dlen = arg.dlen, 
                outcol = arg.outcol,
                constraints=arg.constraints
            ))
        scope_mul()

        def scope_div():
            arg = self.to_atom(plates_in_scope, vars_in_scope)

            spaces.append(SubSpace(
                tree = DivTree(
                    arg = arg.tree
                ), 
                dlen = arg.dlen,
                outcol = Column(
                    [1 / output for output in arg.outcol.mean],
                    []
                ),
                constraints=arg.constraints
            ))
        scope_div()

        # Combine
        return self.combine(spaces)


    def to_atom(self, 
        plates_in_scope : dict[str, Column], 
        vars_in_scope : dict[str, Column]) -> SearchSpace[AtomOption]:
        spaces : list[SubSpace[AtomOption]] = []

        if self.expr_fuel > 0:
            self.expr_fuel -= 1

            def scope_paren():
                content = self.to_expr(plates_in_scope, vars_in_scope)
                spaces.append(SubSpace(
                    tree = ParenTree(
                        content = content.tree
                    ),   
                    dlen = content.dlen, 
                    outcol = content.outcol,
                    constraints = content.constraints
                ))
            scope_paren()


        def scope_mean():
            for plate, col in plates_in_scope.items():
                spaces.append(SubSpace(
                    tree = MeanTree(plate),     
                    dlen = RealVal(1),
                    outcol = Column(col.mean, []),
                    constraints=[]
                ))
        scope_mean()

        def scope_align():
            for plate, col in plates_in_scope.items():
                spaces.append(SubSpace(
                    tree = MeanTree(plate),     
                    dlen = RealVal(1),
                    outcol = Column(col.align, []),
                    constraints = []
                ))
        scope_align()


        def scope_float():
            weight = self.fresh_weight()
            spaces.append(SubSpace(
                tree = FloatTree(
                    content = weight 
                ),   
                dlen = RealVal(1),
                outcol = Column([weight] * len(self.input_data), []),
                constraints=[]
            ))

        scope_float()

        def scope_params():
            for id in self.params:
                pid = self.params.index(id) 

                spaces.append(SubSpace(
                    tree = IdTree(content = id),   
                    dlen = RealVal(1),
                    outcol = Column([RealVal(row[pid].item()) for row in self.input_data], []),
                    constraints = []
                ))
        scope_params()

        # def scope_id():
        #     for var, weights in vars_in_scope.items():
        #         spaces.append(SubSpace(
        #             tree = IdTree(content = var),   
        #             dlen = RealVal(1),
        #             outcol = weights,
        #             constraint=And()
        #         ))
        # scope_id()

        # Combine
        return self.combine(spaces)

### End Spacer ###


class Extractor:
    def __init__(self, model):
        self.model = model

    def choose(self, choices : Choices[T]) -> T:

        for control_key, subtree in choices.items():
            if ("%s" % self.model[control_key]) == "True":
                return subtree

        raise Exception("choose: no satisfactory choice") 

    def from_body(self, choices : Choices[BodyOption]) -> str:
        tree = self.choose(choices)
        if isinstance(tree, SampleTree):
            return f'    {tree.name} ~ {self.from_dist(tree.src)};\n' + self.from_body(tree.contin)
        elif isinstance(tree, PlateTree):
            return (
                f'    {tree.name} # {tree.size} ~ {self.from_dist(tree.src)};\n' + 
                self.from_body(tree.contin)
            )
        elif isinstance(tree, FinalTree):
            return f'    {self.from_dist(tree.content)}'
        else:
            raise Exception("Extractor.from_body") 

    def from_dist(self, choices : Choices[DistroOption]) -> str:
        tree = self.choose(choices)

        if isinstance(tree, NormalTree):
            return f'normal({self.from_expr(tree.mean)}, {self.from_expr(tree.sigma)})'

        elif isinstance(tree, LognormTree):
            return f'lognorm({self.from_expr(tree.mean)}, {self.from_expr(tree.sigma)})'

        elif isinstance(tree, UniformTree):
            return f'uniform({self.from_expr(tree.low)}, {self.from_expr(tree.high)})'

        elif isinstance(tree, HalfnormTree):
            return f'halfnorm({self.from_expr(tree.scale)})'

        elif isinstance(tree, DirectTree):
            return f'@({self.from_expr(tree.content)})'
        else:
            raise Exception("Extractor.from_dist") 

    def from_expr(self, tree : ExprTree):
        prod = self.from_prod(tree.base)
        exts = self.from_exts(tree.exts)

        return prod + exts 

    def from_prod(self, tree : ProdTree):
        atom = self.from_atom(tree.base)
        factors = self.from_factors(tree.factors)
        return atom + factors 

    def from_exts(self, choices : Choices[ExtsOption]):
        tree = self.choose(choices)
        if isinstance(tree, NilExtTree):
            return ''
        elif isinstance(tree, ConsExtTree):
            return self.from_ext(tree.head) + self.from_exts(tree.tail)
        else:
            raise Exception("Extractor.from_exts") 
    
    def from_ext(self, choices : Choices[ExtOption]):
        tree = self.choose(choices)
        if isinstance(tree, PlusTree):
            return ' + ' + self.from_prod(tree.arg) 
        elif isinstance(tree, MinusTree):
            return ' - ' + self.from_prod(tree.arg) 
        else:
            raise Exception("Extractor.from_ext") 

    def from_factors(self, choices : Choices[FactorsOption]):
        tree = self.choose(choices)
        if isinstance(tree, NilFactorTree):
            return ''
        elif isinstance(tree, ConsFactorTree):
            return self.from_factor(tree.head) + self.from_factors(tree.tail)
        else:
            raise Exception("Extractor.from_factors") 

    def from_factor(self, choices : Choices[FactorOption]):
        tree = self.choose(choices)
        if isinstance(tree, MulTree):
            return ' * ' + self.from_atom(tree.arg) 
        elif isinstance(tree, DivTree):
            return ' / ' + self.from_atom(tree.arg) 
        else:
            raise Exception("Extractor.from_factor") 

    def from_atom(self, choices : Choices[AtomOption]):
        tree = self.choose(choices)
        if isinstance(tree, ParenTree):
            return '(' + self.from_expr(tree.content)  + ')'
        elif isinstance(tree, MeanTree):
            return f'mean({tree.plate})'
        elif isinstance(tree, AlignTree):
            return f'align({tree.plate})'
        elif isinstance(tree, IdTree):
            return tree.content
        elif isinstance(tree, FloatTree):
            v = "%s" % self.model[tree.content]
            if v != "None":
                return v 
            else:
                return tree.content.decl().name()
        else:
            raise Exception("Extractor.from_atom") 
### End Extractor ###

def synthesize_body(search_space : SearchSpace[BodyOption], output_data) -> tuple[str, float]:
    s = Solver()

    loss_var = Real('_loss_')
    noise_total = Sum([
        noise(output.item(), prediction) 
        for output, prediction in zip(output_data, search_space.outcol.mean)
    ])

    s.add(search_space.constraints)
    s.add(loss_var == search_space.dlen + noise_total)

    model = None
    solve_max = 5 

    new_loss = 0

    print('------- checking SAT')
    while solve_max > 0 and s.check() == sat:
        model = s.model()
        print(f'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        new_loss = eval('%s' % model[loss_var])

        print(f'new_loss: {new_loss}')
        s.add(loss_var < model[loss_var])
        print(f'constraints added')
        solve_max = solve_max - 1


    if model:
        extractor = Extractor(model)
        source_code = extractor.from_body(search_space.tree) 

        return source_code, new_loss
    else:
        raise Exception("synthesize body: no satisfactory body") 


def noise(expected : Any, actual : Any):
    ## for probability p, and noise n
    ## - log p = n
    ## log p = - n
    ## p = e(-n)
    ## p = 1/e^n
    # return ((actual - expected) ** 2)
    return ((actual - expected) ** 2) / 2
    # return  If(actual > expected, (actual - expected) / 2, (expected - actual) /2)
    # return  If(actual > expected, actual - expected, expected - actual)


def generate_function(file, data=None):
    ast = parse_from_file(file)
    if ast.body:
        return learn_posteriors(ast, data)
    elif ast.spec and data != None:
        header_source_code = util.read(file)

        fields = ast.spec.args
        result_name = ast.spec.result
        params : list = ast.params

        data_size = len(data)
        max_samples = min(10, data_size)
        span = math.floor(data_size/max_samples)
        indicies = torch.tensor([span * i for i in range(max_samples)])
        reduced_data = torch.index_select(data, 0, indicies)
        print('%%%%%%%%%%%%%%%')
        print(reduced_data)
        print('%%%%%% reduced data above %%%%%%%')

        small_input_data = extract_fields(params, fields, reduced_data)
        small_output_data = extract_field(result_name, fields, reduced_data)

        print('------- building search space')
        spacer = Spacer(params, small_input_data)
        search_space = spacer.to_body({}, {})

        body_source_code, synth_loss = synthesize_body(search_space, small_output_data)
        synth_mean_loss = synth_loss / len(small_output_data)

        source_code = header_source_code + '\n' + body_source_code
        print(f'''
##### SYNTHESIZED CODE #####
{source_code}
#############################
        ''')

        if body_source_code:
            ast = parse_from_str(source_code)
            stoch = learn_posteriors(ast, data)
            return stoch


            # input_data = extract_fields(params, fields, data)
            # output_data = extract_field(result_name, fields, data)

    #         input_cols = [input_data[:,i] for i in range(len(params))] 
    #         prediction = stoch.multi(*input_cols)
    #         total_loss = 0 
    #         for expected, actual_distro in zip(output_data, prediction):
    #             total_loss += (actual_distro.mean() - expected) ** 2
    #         stoch_mean_loss = total_loss / len(output_data) 

    #         scaled_loss = stoch_mean_loss * len(small_output_data) 
    #         print(f'''
    # ##### LOSS SUMMARY #####
    # $- synth_mean_loss: {synth_mean_loss}
    # $- stoch_mean_loss: {stoch_mean_loss}

    # $- synth_loss: {synth_loss}
    # $- scaled_loss: {scaled_loss}
    # #############################
    #         ''')


        return None
    else:
        return None