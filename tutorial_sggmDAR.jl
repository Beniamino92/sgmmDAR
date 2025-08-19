# ------------------------------------------------------------------------------
# Set working directory
# ------------------------------------------------------------------------------

# Change this path to where the code is located
path_folder = "/Users/beniamino/Dropbox/Research/Project_Michele_Marina/Code/sggmDAR/for_Jacob/"
cd(path_folder)

# Make sure the following R packages are installed before running:
# corrplot, label.switching, gdata, gridExtra, ggplot2, Matrix, RColorBrewer, matrixcalc


# ------------------------------------------------------------------------------
# Load Packages
# ------------------------------------------------------------------------------

using LinearAlgebra, Random
using StatsBase, StatsFuns, Distributions
using ProgressMeter, BenchmarkTools, MCMCDiagnostics
using DataFrames, CSV, FreqTables, StructArrays
using Plots, Colors, Measures
using RCall, MATLAB
using Clustering, CovarianceEstimation

include("include/structures.jl")
include("include/utilities_sggmDAR.jl")

# ------------------------------------------------------------------------------
# Simulation setup
# ------------------------------------------------------------------------------

Random.seed!(20)
N = 1000               # Length of time series
D = 10                 # Dimensionality
K_true = 5             # True number of states
P_true = 2             # True DAR order

# True DAR parameters
ϕ_true = [0.1, 0.75, 0.15, 0.0, 0.0]
p_true = [6/10, 1/10, 1/10, 1/10, 1/10]

# Allocate containers
μ_true         = zeros(D, K_true)
Σ_true         = zeros(D, D, K_true)
Ω_true         = zeros(D, D, K_true)
Ω_part_true    = zeros(D, D, K_true)
corr_true      = Array{Float64}(undef, D, D, K_true)

# Precision matrix: Identity
Σ_true[:, :, 1]      = I(D)
Ω_true[:, :, 1]      = Matrix(I(D))
Ω_part_true[:, :, 1] = getPartialCorrelation(Ω_true[:, :, 1])
corr_true[:, :, 1]   = getCorrelation(Σ_true[:, :, 1])

# Precision matrix: Star
precision = getPrecision_Star(D)
Ω_true[:, :, 2]      = precision[:Ω]
Σ_true[:, :, 2]      = Hermitian(inv(Ω_true[:, :, 2]))
Ω_part_true[:, :, 2] = precision[:Ω_part]
corr_true[:, :, 2]   = getCorrelation(Σ_true[:, :, 2])

# Precision matrix: Hub
n_blocks = 5
precision = getPrecision_Hub(D, n_blocks)
Ω_true[:, :, 3]      = precision[:Ω]
Σ_true[:, :, 3]      = Hermitian(inv(Ω_true[:, :, 3]))
Ω_part_true[:, :, 3] = precision[:Ω_part]
corr_true[:, :, 3]   = getCorrelation(Σ_true[:, :, 3])

# Precision matrix: AR2
precision = getPrecision_AR2(D)
Ω_true[:, :, 4]      = precision[:Ω]
Σ_true[:, :, 4]      = Hermitian(inv(Ω_true[:, :, 4]))
Ω_part_true[:, :, 4] = precision[:Ω_part]
corr_true[:, :, 4]   = getCorrelation(Σ_true[:, :, 4])

# Precision matrix: Random
Random.seed!(108)
precision = getPrecision_Random(D)
Ω_true[:, :, 5]      = precision[:Ω]
Σ_true[:, :, 5]      = Hermitian(inv(Ω_true[:, :, 5]))
Ω_part_true[:, :, 5] = precision[:Ω_part]
corr_true[:, :, 5]   = getCorrelation(Σ_true[:, :, 5])

# True discovery coefficients and vectorized partial correlation
discovery_true     = get_active_coeff(Ω_true; true_values = true)
discovery_vec_true = discovery_true[:vec]
Ω_part_vec_true    = get_Ω_vec(Ω_part_true)

# Plot true precision and partial correlation matrices
plotHeat_multiple(Ω_true)
plotHeat_multiple(Ω_part_true)

# Generate state-dependent means
for kk = 1:K_true
    μ_true[:, kk] = rand(MultivariateNormal(collect(range(-5, 5, length = D) .+ kk), I(D)))
    μ_true[:, kk] = shuffle(μ_true[:, kk])
end

# Combine into true model parameters
parms_true = Model_Parms(mvNormalParms(μ_true, Σ_true), DAR_Parms(ϕ_true, p_true))

# Simulate observed data
obs, obs_std, γ_seq = Simulate_Data(N, parms_true; scale = true, plt = false)


# ------------------------------------------------------------------------------
# Gibbs Sampler
# ------------------------------------------------------------------------------

# MCMC settings
P_max     = 4
n_states  = 8
n_MCMC    = Int64(4e3)
burn_MCMC = Int64(0.3 * n_MCMC)
thin_MCMC = 1
n_min     = 8

# Prior hyperparameters
α_ϕ_AR         = [10, 2]
α_ϕ_innovations = [2, 20]
α_p             = fill(0.0001, n_states)
R_0             = (1/10) * Matrix(I(D))
μ_0             = fill(0.0, D)
w               = 1
max_steps       = 10

hyper_parms = Hyper_Parms(α_ϕ_AR, α_ϕ_innovations, α_p, μ_0, R_0, w, max_steps, n_min)

# Run Gibbs sampler (takes ~10 min)
mvShrinkageDAR_fit = sggmDAR_GibbsSampler(obs, n_states, P_max, n_MCMC, hyper_parms)

# Extract thinned and burned samples
fit_extract = sggmDAR_ExtractFit(mvShrinkageDAR_fit;
                                        burn_MCMC = burn_MCMC,
                                        thin_MCMC = thin_MCMC,
                                        plt = false)

# Reshape and relabel posterior draws
relabel = true
fit_reshaped = ReshapeFit(fit_extract; relabel = relabel)

# Posterior estimates of K and P
K̂ = fit_reshaped[:bayes_est][:K̂]
P̂ = fit_reshaped[:bayes_est][:P̂]

# Posterior frequency tables
freqtable(fit_reshaped[:posterior_sample][:K]) / size(fit_reshaped[:posterior_sample][:K], 1)
freqtable(fit_reshaped[:posterior_sample][:P]) / size(fit_reshaped[:posterior_sample][:P], 1)

# Posterior predictive draws
posteriorPredictive = sggmDAR_PosteriorPredictive(mvShrinkageDAR_fit, K̂;
                                                         n_draw = 100,
                                                         relabel = relabel,
                                                         modal = true,
                                                         plt = false)

# Plot posterior of p and ϕ
multiHistogram(fit_reshaped[:posterior_sample][:ϕ], ϕ_true, xlimit = [0, 1])
multiHistogram(fit_reshaped[:posterior_sample][:p], p_true, xlimit = [0, 1])

# Match true and estimated precision matrices
Ω_part_est_unlab = fit_reshaped[:bayes_est][:Ω̂_part]
idx_matching      = RetrieveTrueLabel_Mat(Ω_part_est_unlab, Ω_part_true)
Ω_part_est        = Ω_part_est_unlab[:, :, idx_matching]

# Plot estimated partial correlations
plotHeat_multiple(Ω_part_est)

# Extract estimated active coefficients
discovery_est_temp = get_active_coeff(fit_reshaped[:posterior_sample][:Ω];
                                      α_start = 0.025, α_end = 0.975,
                                      true_values = false)
discovery_est = discovery_est_temp[:mat][:, :, idx_matching]
plotHeat_multiple(discovery_est)

# Plot posterior predictive signal
sggmDAR_PlotPosteriorPredictive(obs, K̂, posteriorPredictive[:ŷ], posteriorPredictive[:γ̂])

# Relabel estimated state sequence
state_seq_est = posteriorPredictive[:γ̂]
state_seq_est_relabeled = copy(state_seq_est)
for kk in 1:K̂
    state_seq_est_relabeled[findall(state_seq_est .== idx_matching[kk])] .= kk
end

# Compute and plot state probabilities
stateProbs = getStateProbablities(obs, fit_reshaped)
path_file = path_folder * "stateProbs.pdf"
plotStateProbs(stateProbs; path_file = path_file, width = 15, height = 5)
