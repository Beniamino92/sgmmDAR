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

## Model summary (one-page intuition)

- **Observations:**  
  y_t | γ_t = j ~ N_D(μ_j, Ω_j⁻¹), with sparse precision matrix Ω_j for each state j.

- **Hidden states:**  
  γ_t ∈ {1, …, M}, with M learned via an overfitted mixture and a sparse Dirichlet prior on the innovations.

- **Switching rule (DAR-P):**  
  p(γ_t = i | γ_{t-1:t-P}) = Σ_{l=1}^P φ_l · 1{γ_{t-l} = i} + φ_0 · π_i,  
  where φ_0 = 1 − Σ_{l=1}^P φ_l.

- **Order learning:**  
  A modified stick-breaking prior with **cumulative shrinkage** on {φ_l} forces higher lags to zero after the first inactive lag—thus estimating the effective lag order P̂ without RJMCMC.

- **Graph recovery:**  
  A **Graphical Horseshoe prior** on the off-diagonal entries of each precision matrix Ω_j (with global & local shrinkage) combined with credibility-interval selection yields sparse conditional graphs for each regime.




