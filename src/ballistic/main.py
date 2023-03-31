import util

from textx import metamodel_from_file
parser = metamodel_from_file(util.resource('grammars/bll.tx'))

def parse_from_file(path):
    return parser.model_from_file(path)

def generate_from_ast(ast):
    input_name = next(param.name for param in ast.params)
    return f'''
def model({", ".join([param.name for param in ast.params])}):
    {generate_from_body(input_name, ast.body)}
    '''
#     a = pyro.param("a", lambda: torch.randn(()))
#     b_a = pyro.param("bA", lambda: torch.randn(()))
#     b_r = pyro.param("bR", lambda: torch.randn(()))
#     b_ar = pyro.param("bAR", lambda: torch.randn(()))
#     sigma = pyro.param("sigma", lambda: torch.ones(()), constraint=constraints.positive)

#     mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness

#     with pyro.plate("data", len(ruggedness)):
#         return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)

# def foo(ica, r):
#     svi_samples = predictive(torch.tensor([ica]), torch.tensor([r]), log_gdp=None)
#     svi_gdp = svi_samples["obs"]
#     return svi_gdp[:, 0]

def generate_from_body(input_name, body):
    if body.__class__.__name__ == "Bind":
        return f'''
    {body.name} = {generate_from_dist(body.name, body.src)}
    {generate_from_body(input_name, body.dst)}
        '''
    else: 
        return f'''
    with pyro.plate("data", len({input_name})):
        return {generate_from_dist("obs", body)}
        '''

def generate_from_dist(name, dist):
    if dist.__class__.__name__ == "Normal":
        return f'pyro.sample("{name}", dist.Normal({generate_from_expr(dist.mean)}, {generate_from_expr(dist.sigma)}))'
    else:
        assert dist.__class__.__name__ == "Direct"
        return f'{generate_from_expr(dist.content)}'

def generate_from_expr(expr):
    return f'{expr}'

if __name__ == "__main__":
    program = parse_from_file(util.resource('examples/hello.bll'))
    print('-----------------------------------------------------')
    print(generate_from_ast(program))
    print('-----------------------------------------------------')
    # print(f'''
    # -------------------------
    # {util.resource("bll.tx")}
    # -------------------------
    # {util.project_path()}
    # -------------------------
    # ''')
    # pass