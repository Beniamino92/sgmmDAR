# sggmDAR



**Sparse Gaussian Graphical modeling with Discrete Autoregressive processes**  

Julia code for Bayesian estimation of **time-varying Gaussian graphical models** under **latent regime switching** with **discrete autoregressive (DAR) dynamics** and **automatic order/complexity learning**.

## What is it?

`sggmDAR` fits multivariate time series with an unobserved, finite set of connectivity “states.”  
Key pieces:

- **State dynamics:** A **Discrete Autoregressive (DAR)** process governs regime switching with order \(P\) learned from the data via a **zero-inducing cumulative shrinkage prior** (once a lag goes to zero, all higher lags are forced to zero).
- **Number of states:** Overfitted mixture with a **sparsity-inducing Dirichlet prior** estimates the effective number of regimes \( \hat{M} \).
- **Graphs within states:** **Graphical Horseshoe (GHS)** priors shrink off-diagonal precision entries toward zero to recover sparse conditional dependence (precision) matrices state-by-state.
- **Inference:** A custom MCMC with birth/death moves for DAR order, block updates for precision matrices, and a forward-backward–style update for the hidden sequence. Post-processing handles label switching.
- **Use cases:** Dynamic brain connectivity (fMRI), multi-sensor networks, macro/finance systems—any setting where **graphs evolve through recurring states** rather than continuously.

<p align="center"> <img src="https://github.com/Beniamino92/sgmmDAR/blob/main/images/main.png" width="500"/> </p>



The full methodology appears in:

> B. Hadj-Amar, A. M. Bornstein, M. Guindani, M. Vannucci (2025), *Discrete Autoregressive Switching Processes with Cumulative Shrinkage Priors for Graphical Modeling of Time Series Data*, **Journal of Computational and Graphical Statistics (JCGS)**.

---

## Repository layout
```
sggmDAR/
├─ src/ # (if you package-ize) core Julia code
├─ include/
│ ├─ utilites_sggmDAR.jl # utilities (I/O, helpers, diagnostics)
│ └─ structures.jl # data structures / types
├─ sgmmDAR_tutorial.jl # end-to-end tutorial script (examples & plots)
└─ README.md # you're reading it
```

## R (for post-processing and plots)

Several post-processing steps (such as label-switching resolution, correlation/graph visualizations, and figure assembly) are performed in **R**.  
This can be done directly in R, or from Julia using the [`RCall.jl`](https://github.com/JuliaInterop/RCall.jl) package.

### Required R packages

Install the following R packages before running the post-processing:

```r
install.packages(c(
  "corrplot",
  "label.switching",
  "gdata",
  "gridExtra",
  "ggplot2",
  "Matrix",
  "RColorBrewer",
  "matrixcalc"
))
```



## Model summary 

- **Observations:**  
  $y_t \mid \gamma_t = j \sim \mathcal{N}_D(\mu_j,\ \Omega_j^{-1})$, with sparse precision matrix $\Omega_j$ for each state $j$.

- **Hidden states:**  
  $\gamma_t \in \{1,\dots,M\}$, with $M$ learned via an overfitted mixture and a sparse Dirichlet prior on the innovations.

- **Switching rule (DAR-$P$):**  
  $p(\gamma_t = i \mid \gamma_{t-1:t-P}) = \sum_{l=1}^{P} \phi_l \mathbf{1}\{\gamma_{t-l}=i\} + \phi_0 \pi_i,\ \ \phi_0 = 1 - \sum_{l=1}^{P} \phi_l.$


- **Order learning:**  
  A modified stick-breaking prior with **cumulative shrinkage** on $\{\phi_l\}$ forces higher lags to zero after the first inactive lag—thus estimating the effective lag order $\hat{P}$ without RJMCMC.

- **Graph recovery:**  
  A **Graphical Horseshoe prior** on the off-diagonal entries of each precision matrix $\Omega_j$ (with global & local shrinkage) combined with credibility-interval selection yields sparse conditional graphs for each regime.


## Tutorial snapshots

Below are a few quick snippets from `sgmmDAR_tutorial.jl`. See the script for end-to-end details.

### 1) Fit the model

```julia
# Inputs prepared earlier:
#   obs :: Matrix{Float64}  # T × D time series
#   n_states :: Int         # overfitted maximum number of regimes
#   P_max :: Int            # maximum DAR order
#   n_MCMC :: Int           # total iterations
#   hyper_parms :: NamedTuple or Dict with priors

# Run Gibbs sampler (~10 minutes depending on settings)
mvShrinkageDAR_fit = sggmDAR_GibbsSampler(obs, n_states, P_max, n_MCMC, hyper_parms)
```

### 2) Extract (burn + thin)

```julia
burn_MCMC = 5_000
thin_MCMC = 10

fit_extract = sggmDAR_ExtractFit(
    mvShrinkageDAR_fit;
    burn_MCMC = burn_MCMC,
    thin_MCMC = thin_MCMC,
    plt = false,
)
```

### 3) Reshape + (optionally) relabel states

```julia
relabel = true  # use ECR or similar relabeling internally
fit_reshaped = ReshapeFit(fit_extract; relabel = relabel)
```

### 4) Posterior estimates of K and M

```julia
# Posterior estimates of K and P
K̂ = fit_reshaped[:bayes_est][:K̂]
P̂ = fit_reshaped[:bayes_est][:P̂]
```

### 5) Posterior predictive draws + plot

```julia
posteriorPredictive = sggmDAR_PosteriorPredictive(mvShrinkageDAR_fit, K̂;
                                                         n_draw = 100,
                                                         relabel = relabel,
                                                         modal = true,
                                                         plt = false)
sggmDAR_PlotPosteriorPredictive(obs, K̂, posteriorPredictive[:ŷ], posteriorPredictive[:γ̂])
```
<p align="center"> <img src="https://github.com/Beniamino92/sgmmDAR/blob/main/images/post_pred.png" width="500"/> </p>

### 6) Estimated partial correlations (state-specific) 

``` julia
plotHeat_multiple(Ω_part_est)
```
<p align="center"> <img src="https://github.com/Beniamino92/sgmmDAR/blob/main/images/partial_corr.png" width="500"/> </p>



