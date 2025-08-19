mutable struct mvNormalParms
  μ::Array # mean vector (all states)
  Σ::Array # covariance matrices (all states)
end

mutable struct DAR_Parms
  ϕ::Array # DAR coefficients (including 1 - sum(ϕ))
  p::Array # probs innovations
end

mutable struct ShrinkageDAR_Parms
  v::Array # stick breaking weights (e.g. [v_0, v_1, 1, NaN])
  z::Array # indicator stick breaking or end stick (e.g. [false, true, true] )
  p::Array # probs innovations
end

mutable struct GHS
    Ω::Matrix
	  Σ::Matrix
    Λ²::Matrix
	  ν::Matrix
    τ²::Real
	  ξ::Real

	  GHS() = new()
end


mutable struct EmissionParms
	mean::Array
	precision::Vector{GHS}
end

struct PosteriorSample
	emissions::Vector{EmissionParms}
	ShrinkageDAR::Vector{ShrinkageDAR_Parms}
	stateSeq::Array
	classProb::Array
end


struct mvShrinkageDAR_Fit
	posteriorSample::PosteriorSample
	obs::Matrix
end


struct Model_Parms
  mvNorm::mvNormalParms
  DAR::DAR_Parms
end


mutable struct Hyper_Parms
  # α_ϕ::Array # [ϕ, (1-sum(ϕ))] ~ Dirichlet(α_ϕ)

  α_ϕ_AR::Array # ϕ_1, …, ϕ_p ~ Beta(α_ϕ_AR[1], α_ϕ_AR[1])
  α_ϕ_innovations::Array # ϕ_0 ~ Beta(α_ϕ_innovations[1], α_ϕ_innovations[2])
  α_p::Array # p ~ Dirichlet(α_p)

  # μ | Σ ~ Normal(μ_0, inv(R_0))
  μ_0::Array # prior mean
  R_0::Matrix # prior precision

  # slice sampler
  w::Real
  max_steps::Int

  # minimum n obs in each state
  n_min::Int

end
