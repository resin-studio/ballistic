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

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.4')

pyro.enable_validation(True)
pyro.set_rng_seed(1)
logging.basicConfig(format='%(message)s', level=logging.INFO)

# Set matplotlib settings
# %matplotlib inline
# plt.style.use('default')


# # DATA
# DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
# data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
# df = data[["cont_africa", "rugged", "rgdppc_2000"]]

# df = df[np.isfinite(df.rgdppc_2000)]
# df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

# train = torch.tensor(df.values, dtype=torch.float)
# is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]


# # DATA PLOT
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
# african_nations = df[df["cont_africa"] == 1]
# non_african_nations = df[df["cont_africa"] == 0]
# sns.scatterplot(x=non_african_nations["rugged"],
#                 y=non_african_nations["rgdppc_2000"],
#                 ax=ax[0])
# ax[0].set(xlabel="Terrain Ruggedness Index",
#           ylabel="log GDP (2000)",
#           title="Non African Nations")
# sns.scatterplot(x=african_nations["rugged"],
#                 y=african_nations["rgdppc_2000"],
#                 ax=ax[1])
# ax[1].set(xlabel="Terrain Ruggedness Index",
#           ylabel="log GDP (2000)",
#           title="African Nations");

# # plt.show()


# # SIMPLE MODEL

# def simple_model(is_cont_africa, ruggedness, log_gdp=None):
#     a = pyro.param("a", lambda: torch.randn(()))
#     b_a = pyro.param("bA", lambda: torch.randn(()))
#     b_r = pyro.param("bR", lambda: torch.randn(()))
#     b_ar = pyro.param("bAR", lambda: torch.randn(()))
#     sigma = pyro.param("sigma", lambda: torch.ones(()), constraint=constraints.positive)

#     mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness

#     with pyro.plate("data", len(ruggedness)):
#         return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)

# print('--------------')
# print(simple_model(is_cont_africa, ruggedness))
# print('--------------')

# # Bayesian model
def model(is_cont_africa, ruggedness, log_gdp=None):
    a = pyro.sample("a", dist.Normal(0., 10.))
    b_a = pyro.sample("bA", dist.Normal(0., 1.))
    b_r = pyro.sample("bR", dist.Normal(0., 1.))
    b_ar = pyro.sample("bAR", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))

    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness

    with pyro.plate("data", len(ruggedness)):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)

# pyro.render_model(model, model_args=(is_cont_africa, ruggedness, log_gdp), render_distributions=True)


pyro.clear_param_store()

# These should be reset each training loop.
auto_guide = pyro.infer.autoguide.AutoNormal(model)
adam = pyro.optim.Adam({"lr": 0.02})  # Consider decreasing learning rate.
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

# losses = []
# for step in range(1000 if not smoke_test else 2):  # Consider running for more steps.
#     loss = svi.step(is_cont_africa, ruggedness, log_gdp)
#     losses.append(loss)
#     if step % 100 == 0:
#         logging.info("Elbo loss: {}".format(loss))

# plt.figure(figsize=(5, 2))
# plt.plot(losses)
# plt.xlabel("SVI step")
# plt.ylabel("ELBO loss");
# # plt.show()


# # Sample from trained guide the learned latent variables/distributions
# with pyro.plate("samples", 800, dim=-1):
#     samples = auto_guide(is_cont_africa, ruggedness)

# gamma_within_africa = samples["bR"] + samples["bAR"]
# gamma_outside_africa = samples["bR"]

# fig = plt.figure(figsize=(10, 6))
# sns.histplot(gamma_within_africa.detach().cpu().numpy(), kde=True, stat="density", label="African nations")
# sns.histplot(gamma_outside_africa.detach().cpu().numpy(), kde=True, stat="density", label="Non-African nations", color="orange")
# fig.suptitle("Density of Slope : log(GDP) vs. Terrain Ruggedness");
# plt.xlabel("Slope of regression line")
# plt.legend()
# # plt.show()


# # construct the function to predict (forward evaluation)
# # functools.partial(simple_model, log_gdp=log_gdp)
# # num_samples is used to estimate the distribution at each input/output pair
# predictive = pyro.infer.Predictive(model, guide=auto_guide, num_samples=100)
# svi_samples = predictive(is_cont_africa[0:1], ruggedness[0:1], log_gdp=None)
# svi_gdp = svi_samples["obs"]
# # print(svi_gdp)
# # construct a function from scalar to distribution
# def foo(ica, r):
#     svi_samples = predictive(torch.tensor([ica]), torch.tensor([r]), log_gdp=None)
#     svi_gdp = svi_samples["obs"]
#     return svi_gdp[:, 0]

# print("---------------------------------------")
# print(foo(is_cont_africa[0], ruggedness[0]))
# print(len(foo(is_cont_africa[0], ruggedness[0])))
# print(foo(is_cont_africa[0], ruggedness[0]).mean().item())

# svi_samples = predictive(is_cont_africa, ruggedness, log_gdp=None)
# svi_gdp = svi_samples["obs"]

## note: len(is_cont_africa) == len(svi_gdp[0])

# predictions = pd.DataFrame({
#     "cont_africa": is_cont_africa,
#     "rugged": ruggedness,
#     "y_mean": svi_gdp.mean(0).detach().cpu().numpy(),
#     "y_perc_5": svi_gdp.kthvalue(int(len(svi_gdp) * 0.05), dim=0)[0].detach().cpu().numpy(),
#     "y_perc_95": svi_gdp.kthvalue(int(len(svi_gdp) * 0.95), dim=0)[0].detach().cpu().numpy(),
#     "true_gdp": log_gdp,
# })
# african_nations = predictions[predictions["cont_africa"] == 1].sort_values(by=["rugged"])
# non_african_nations = predictions[predictions["cont_africa"] == 0].sort_values(by=["rugged"])

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
# fig.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)

# ax[0].plot(non_african_nations["rugged"], non_african_nations["y_mean"])
# ax[0].fill_between(non_african_nations["rugged"], non_african_nations["y_perc_5"], non_african_nations["y_perc_95"], alpha=0.5)
# ax[0].plot(non_african_nations["rugged"], non_african_nations["true_gdp"], "o")
# ax[0].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="Non African Nations")

# ax[1].plot(african_nations["rugged"], african_nations["y_mean"])
# ax[1].fill_between(african_nations["rugged"], african_nations["y_perc_5"], african_nations["y_perc_95"], alpha=0.5)
# ax[1].plot(african_nations["rugged"], african_nations["true_gdp"], "o")
# ax[1].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="African Nations");

# plt.show()


# if __name__ == "__main__":
#     pass


print(dist.Normal(0, 10).sample())