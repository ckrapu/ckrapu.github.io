---
layout: post
title: "Poverty Bayes: fitting million-parameter models for pennies with serverless MCMC"
date: 2026-05-27
description: Fitting a million-parameter Bayesian model on Modal with PyMC and NumPyro
tags: statistics python pymc modal gpu
categories: tutorials
---

It's a good time to be an applied probabilist. The deep learning revolution has led to tremendous improvements in the $ / flops department, and we Bayesians can easily hop on this train! During grad school, I used to spend nights and weekends babysitting MCMC runs on my GeForce Titan XP running in my bedroom (by the way, thank you [NVIDIA Academic Grant Program](https://www.nvidia.com/en-us/industries/higher-education-research/academic-grant-program/)) while simultaneously trying to keep the waste heat from cooking me as I slept. If you are a newcomer to this field, rejoice in the knowledge that all this suffering is a thing of the past. A slew of companies are rushing to the fore with user-friendly platforms for renting GPUs. For prototyping, I really enjoy working with Modal since I'm cheap and I'm too lazy to keep managing my own fleet.

In this post, I'll show a workflow for using GPU-based inference on Modal for a model which is very large by the standards of Bayesian statisticians $$(\vert \theta\vert \gt 10^6)$$, by deploying to a datacenter GPU and renting it only for a short time.

## Model & data

We'll use synthetic data for this example.

 I've chosen a hierarchical logistic regression for this post since it has a non-conjugate likelihood, appears commonly in practice, and can easily be assigned more parameters by increasing the number of covariates and/or the number of groups. 
 
 Let $$i$$, $$g$$, and $$k$$ denote the indices over observations, groups, and covariates. Furthermore, let $$x_i \in \mathbb{R}^{20}$$ be the covariate vector for observation $$i$$, and let $$g_i \in \{1,\ldots,100000\}$$ identify its group. The data-generating process uses population-level slopes $$\beta_k$$, group intercept deviations $$\alpha_g$$, and group slope deviations $$\gamma_{gk}$$. The binary outcome is generated from

$$
Y_i \sim \operatorname{Bernoulli}(p_i),
\qquad
\operatorname{logit}(p_i)
= \alpha + \alpha_{g_i} + \sum_{k=1}^{K} x_{ik}(\beta_k + \gamma_{g_i k}).
$$

Essentially, this is a logistic regression with random slopes for 20 covariates and a random intercept for each of 100,000 groups in the data.
 
We'll use a non-centered parameterization for the group effects. The prior specification is

$$
\begin{aligned}
\alpha &\sim \operatorname{Normal}(0, 1.5), \\
\beta_k &\sim \operatorname{Normal}(0, 1), \\
\sigma_\alpha &\sim \operatorname{HalfNormal}(1), \\
\sigma_{\gamma,k} &\sim \operatorname{HalfNormal}(0.5), \\
z_{\alpha,g} &\sim \operatorname{Normal}(0, 1), \\
z_{\gamma,gk} &\sim \operatorname{Normal}(0, 1), \\
\alpha_g &= \sigma_\alpha z_{\alpha,g}, \\
\gamma_{gk} &= \sigma_{\gamma,k} z_{\gamma,gk}.
\end{aligned}
$$

The code below produces a synthetic dataset; we can control the overall sparsity of the response with the value of `α_true`.

```python
import numpy as np

RANDOM_SEED = 827
rng = np.random.default_rng(RANDOM_SEED)

N = 1_000_000 # Number of data points
G = 100_000   # Number of groups
K = 20      # Number of covariates / features

group_idx = rng.integers(0, G, size=N, dtype=np.int64)
X = rng.normal(size=(N, K)).astype(np.float32)

α_true = np.float32(-1.0)
β_true = rng.normal(0.0, 0.45, size=K).astype(np.float32)
σ_α_true = np.float32(0.80)
σ_γ_true = rng.uniform(0.15, 0.35, size=K).astype(np.float32)
α_group_true = rng.normal(0.0, σ_α_true, size=G).astype(np.float32)
γ_group_true = rng.normal(0.0, σ_γ_true, size=(G, K)).astype(np.float32)

η = α_true + α_group_true[group_idx] + np.sum(X * (β_true + γ_group_true[group_idx]), axis=1)
p = 1.0 / (1.0 + np.exp(-η))
y = rng.binomial(1, p).astype(np.int64)

print(f"Average of y: {np.mean(y):.2f}")
```

```text
Average of y: 0.36
```

Here, we define our model in PyMC. This is a fairly standard model definition without much nuance or many tricks.

```python
import pymc as pm
import pytensor
import pytensor.tensor as pt

pytensor.config.floatX = "float32"

def build_model(X, y, group_idx):

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    group_idx = np.asarray(group_idx, dtype=np.int64)

    N, K = X.shape
    G = int(group_idx.max()) + 1
    coords = {
        "obs": np.arange(N),
        "group": np.arange(G),
        "covariate": np.arange(K),
    }

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X", X, dims=("obs", "covariate"))
        group_lookup = pt.as_tensor_variable(group_idx, name="group_lookup")

        α = pm.Normal("α", np.float32(0.0), np.float32(1.5))
        β = pm.Normal("β", np.float32(0.0), np.float32(1.0), dims="covariate")
        σ_α = pm.HalfNormal("σ_α", np.float32(1.0))
        σ_γ = pm.HalfNormal("σ_γ", np.float32(0.5), dims="covariate")

        z_α = pm.Normal("z_α", np.float32(0.0), np.float32(1.0), dims="group")
        z_γ = pm.Normal("z_γ", np.float32(0.0), np.float32(1.0), dims=("group", "covariate"))
        α_group = σ_α * z_α 
        γ_group = σ_γ * z_γ

        η = α + α_group[group_lookup] + pm.math.sum(X_data * (β + γ_group[group_lookup]), axis=1)
        pm.Bernoulli("Y", logit_p=η, observed=y, dims="obs")

    return model
```

To illustrate why this model could be challenging to fit using only a CPU, we profile the gradient evaluation time. The `dlogp` function represents $$\frac{\partial}{\partial \theta}[\log \tilde{p}]$$ with $$\tilde{p}$$ denoting the unnormalized posterior density; it may be calculated up to 1000 times per sample with the evaluation count increasing with the difficulty of the problem in terms of posterior geometry and/or nonlinearity. Hamiltonian Monte Carlo, the workhorse of most modern Bayesian programming frameworks, requires these gradient evaluations to draw Monte Carlo samples.

The code cell below runs the gradient a few times and records the time taken.

```python
import time

def median_runtime(fn, n=5):
    fn()
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), times

local_model = build_model(X, y, group_idx)
local_point = local_model.initial_point()

with local_model:
    dlogp_fn = local_model.compile_dlogp()

local_grad_median, local_grad_times = median_runtime(lambda: dlogp_fn(local_point))
print(f"Median grad eval time is {local_grad_median:.2f} seconds")
```

```text
Median grad eval time is 0.20 seconds
```

If we try to extrapolate to the time required for running a full chain of 1000 samples, we find that assuming ~250 grad evals per sample, we would need 50 seconds per sample and 50,000 seconds (~14 hours) to run a full chain! This simply won't do.

Let's deploy this remotely and see what we get!

## Running remotely on Modal

For those of you unfamiliar with the magical world of serverless GPU, please take a look at Modal. I will probably never be wealthy enough to outright own a beautiful server rack of datacenter GPUs, so this is the best I can get. Basically, Modal provides APIs for spinning up jobs on GPUs with very little friction and little downtime.

To make this example run nicely on GPU, we'll make a few adjustments. First, we'll toggle it to use the Jax + NumPyro backend to work out-of-the-box with an NVIDIA GPU.

```python
import modal

modal_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "arviz==1.1.0",
        "jax[cuda12]==0.7.2",
        "numpy==2.3.5",
        "numpyro==0.19.0",
        "pandas==2.3.3",
        "pymc==6.0.1",
    )
)

app = modal.App("i-should-be-working-right-now", image=modal_image)
```

We'll set a few environment flags for the float precision and the devices, define a helper to benchmark the execution time, and apply some Jax-isms to prep the compute graph and evaluate it.

```python
def remote_model(X, y, group_idx):
    import os

    os.environ["JAX_PLATFORMS"] = "cuda"

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytensor

    pytensor.config.floatX = "float32"
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    group_idx = np.asarray(group_idx, dtype=np.int64)

    gpu_devices = jax.devices("gpu")
    gpu_check = jax.device_put(jnp.ones((512, 512), dtype=jnp.float32), gpu_devices[0]).sum().block_until_ready()
    model = build_model(X, y, group_idx)

    info = {
        "N": X.shape[0],
        "G": int(group_idx.max()) + 1,
        "K": X.shape[1],
        "jax_backend": jax.default_backend(),
        "jax_gpu_devices": [str(device) for device in gpu_devices],
        "gpu_check_sum": float(gpu_check),
        "x_dtype": str(X.dtype),
        "value_var_dtypes": {var.name: var.dtype for var in model.value_vars},
    }
    print(f"JAX backend: {info['jax_backend']}; GPU devices: {info['jax_gpu_devices']}")
    return model, info


def median_runtime(fn, n=5, synchronize=None):
    import time

    import numpy as np

    if synchronize is None:
        synchronize = lambda result: None

    def timed_run():
        start = time.perf_counter()
        result = fn()
        synchronize(result)
        return time.perf_counter() - start

    synchronize(fn())
    times = [timed_run() for _ in range(n)]
    return float(np.median(times)), times


def wait_jax(result):
    import jax

    jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)


@app.function(gpu="A100", timeout=2 * 60 * 60)
def profile_dlogp(X, y, group_idx, n_evals=5):

    import jax
    import jax.numpy as jnp
    from pymc.sampling.jax import get_jaxified_logp

    model, info = remote_model(X, y, group_idx)
    point = model.initial_point()

    with model:
        pytensor_dlogp = model.compile_dlogp()

    pytensor_median, pytensor_times = median_runtime(lambda: pytensor_dlogp(point), n_evals)

    values = [jnp.asarray(point[var.name]) for var in model.value_vars]
    jax_loss = get_jaxified_logp(model)
    jax_dlogp = jax.jit(jax.value_and_grad(jax_loss))

    wait_jax(jax_dlogp(values))
    jax_median, jax_times = median_runtime(lambda: jax_dlogp(values), n_evals, wait_jax)

    profile = {
        "median_pytensor_dlogp_eval_seconds": pytensor_median,
        "median_jax_dlogp_eval_seconds": jax_median,
    }
    print(f"Median dlogp eval: PyTensor={pytensor_median:.3f}s, JAX={jax_median:.3f}s")
    return {**info, **profile}


@app.function(gpu="A100", timeout=2 * 60 * 60)
def fit_hierarchical_logistic(X, y, group_idx, draws=100, tune=100, random_seed=RANDOM_SEED):
    import time
    import arviz as az
    import pymc as pm

    model, info = remote_model(X, y, group_idx)

    with model:
        start = time.perf_counter()
        idata = pm.sample(
            draws,
            tune=tune,
            chains=1,
            nuts_sampler="numpyro",
            random_seed=random_seed,
            var_names=["α", "β", "σ_α", "σ_γ"],
            progressbar=False,
        )
        elapsed_seconds = time.perf_counter() - start

    β_mean = idata.posterior["β"].mean(dim=("chain", "draw")).to_numpy()

    return {
        **info,
        "elapsed_seconds": elapsed_seconds,
        "summary": az.summary(idata, var_names=["α", "β", "σ_α", "σ_γ"]),
        "β_mean": β_mean.tolist(),
    }
```

With all of this helper logic written, we can finally deploy to the cloud! We will start by running a short job to just profile the logp gradient function.

```python
print("Launching Modal GPU dlogp profile")
with app.run():
    profile_result = profile_dlogp.remote(X, y, group_idx)
print(f"Finished running; profile results: {profile_result}")
```

```text
Launching Modal GPU dlogp profile
Finished running; profile results: {'N': 1000000, 'G': 100000, 'K': 20, 'jax_backend': 'gpu', 'jax_gpu_devices': ['cuda:0'], 'gpu_check_sum': 262144.0, 'x_dtype': 'float32', 'value_var_dtypes': {'α': 'float32', 'β': 'float32', 'σ_α_log__': 'float32', 'σ_γ_log__': 'float32', 'z_α': 'float32', 'z_γ': 'float32'}, 'median_pytensor_dlogp_eval_seconds': 0.2271286229999987, 'median_jax_dlogp_eval_seconds': 0.0013876599999989025}
```

Interesting - the gradient takes 0.001 seconds on GPU and 0.22 seconds on CPU. That is around a 200x speedup!

Next, we run the Markov chain to completion and retrieve the results.

```python
print("Launching Modal GPU run")
with app.run():
    result = fit_hierarchical_logistic.remote(X, y, group_idx)

print(f"MCMC run finished in {result["elapsed_seconds"]:.2f} seconds")
```

```text
MCMC run finished in 272.16 seconds
```

Modal helpfully lists [their prices per second](https://modal.com/pricing); an A100 runs for $0.000583 / second, meaning this run cost me around 15 cents from start to finish.

## Parameter Recovery

We can see all the sampler diagnostics in the posterior summary. We'd need to run more chains to get the $$\hat{R}$$ value for this model.

```python
import pandas as pd

posterior_summary = pd.DataFrame(result["summary"])
posterior_summary.iloc[0:5]
```

```text
  parameter   mean     sd  eti89_lb  eti89_ub  ess_bulk  ess_tail  r_hat  \
0         α -0.980  0.015    -0.997    -0.952     1.510     5.445    NaN   
1      β[0] -0.275  0.005    -0.282    -0.266     2.042     5.659    NaN   
2      β[1]  0.630  0.010     0.612     0.641     1.627     5.374    NaN   
3      β[2] -0.358  0.007    -0.366    -0.345     2.390     5.593    NaN   
4      β[3]  0.359  0.006     0.348     0.366     1.834     5.607    NaN   

   mcse_mean  mcse_sd  
0      0.012    0.008  
1      0.004    0.003  
2      0.008    0.005  
3      0.005    0.003  
4      0.005    0.003
```

A quick diagnostic is to compare the posterior mean of the fixed effects with the values used to simulate the data. With $$10^6$$ observations, this model has more than enough data to estimate the fixed-effects.

```python
import matplotlib.pyplot as plt

blog_colors = {
    "background": "#1c1c1d",
    "text": "#e8e8e8",
    "accent": "#2698ba",
    "point": "#ffffff",
}
β_posterior_mean = np.asarray(result["β_mean"])
limits = [
    min(β_true.min(), β_posterior_mean.min()) - 0.05,
    max(β_true.max(), β_posterior_mean.max()) + 0.05,
]

fig, ax = plt.subplots(figsize=(4.8, 3.6))
fig.patch.set_facecolor(blog_colors["background"])
ax.set_facecolor(blog_colors["background"])
ax.scatter(β_true, β_posterior_mean, s=36, color=blog_colors["point"], alpha=0.82)
ax.plot(limits, limits, color=blog_colors["accent"], linewidth=1.5)
ax.set_xlim(limits); ax.set_ylim(limits); ax.grid(False)
ax.set_xlabel("True fixed effect", color=blog_colors["text"]); ax.set_ylabel("Posterior mean fixed effect", color=blog_colors["text"])
ax.tick_params(colors=blog_colors["text"])
for spine in ax.spines.values():
    spine.set_color(blog_colors["text"])
```

<img src="{{ site.baseurl }}/images/2026-05-27-poverty-bayes-serverless-mcmc/cell-22-output-0.png" alt="Notebook output" width="620"/>

Nice! The true value and posterior mean estimates line up perfectly. A job well done for MCMC. 

I think that Bayes on GPU is tremendously undervalued. If this interests you or you have ideas to chat about, drop me a line.
