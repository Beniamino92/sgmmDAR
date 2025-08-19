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


