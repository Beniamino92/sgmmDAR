# -
function sggmDAR_GibbsSampler(obs, n_states, P_max, n_MCMC, hyper_parms)

	burn_MCMC_z = Int64(0.3*n_MCMC) # warm up for DAR sampler

    T = size(obs, 1)
    D = size(obs, 2)
    w = hyper_parms.w
    max_steps = hyper_parms.max_steps
    state_CartIdx_all = Dict((pp) => get_StateCartesianIndices(n_states, pp)
                             for pp in 1:P_max)
    K = copy(n_states)
    ind_all =  get_IndexesAll(D) # auxiliary index set for full conditionals GHS


    ShrinkageDAR_sample = [ShrinkageDAR_Parms(
							ones(Float64, P_max),
	                        zeros(Bool, P_max - 1),
                            Array{Float64}(undef, n_states-1)) for tt in 1:n_MCMC]
    emissionParms_sample = [EmissionParms(zeros(Float64, D, K),
                            [GHS() for kk in 1:K]) for tt in 1:n_MCMC]
    classProb_sample = Array{Float64}(undef, T, K, n_MCMC)
    stateSeq_sample =  Array{Int64}(undef, T, n_MCMC)
    P_sample = zeros(Int64, n_MCMC) # (auxuliary, byproduct of z) n of active components DAR


    # -- initial values DAR pars
    # ShrinkageDAR_sample[1].z = fill(true, P_max-1)

    ShrinkageDAR_sample[1].v = vcat([rand(Beta(1, 10)), 1],
                          fill(NaN, P_max - 2))								

    ShrinkageDAR_sample[1].v = vcat([rand(Beta(hyper_parms.α_ϕ_innovations[1],
                                     hyper_parms.α_ϕ_innovations[2])), 1],
                          fill(NaN, P_max - 2))
    ShrinkageDAR_sample[1].p = fill(1/(n_states), n_states-1)
    P_sample[1] = 1

    # -- initial values emissions
    # K_start = 2
    K_start = K
    init = kmeans(obs', K_start)
    idx_clust = init.assignments

    @inbounds for kk in 1:K_start
        emissionParms_sample[1].mean[:, kk] = init.centers[:, kk]
        emissionParms_sample[1].precision[kk].Σ = 1.0 * Matrix(I(D))
        emissionParms_sample[1].precision[kk].Ω =  Matrix(1.0*I(D))
        emissionParms_sample[1].precision[kk].Λ² = ones(Float64, D, D)
        emissionParms_sample[1].precision[kk].ν = ones(Float64, D, D)
        emissionParms_sample[1].precision[kk].τ² = 1.0
        emissionParms_sample[1].precision[kk].ξ = 1.0
    end

    # @inbounds for kk in (1):K
    #     temp = sampleEmissions_fromPrior(hyper_parms)
    #     emissionParms_sample[1].mean[:, kk] = temp[:mean]
    #     emissionParms_sample[1].precision[kk] = temp[:precision]
    # end



    # --- Gibbs Sampler

    @inbounds @showprogress for tt in 2:n_MCMC

        emissions_curr = emissionParms_sample[tt-1]
        Shrinkage_DAR_curr = ShrinkageDAR_sample[tt-1]
        P_curr = P_sample[tt-1]

        p_curr = copy(ShrinkageDAR_sample[tt-1].p)
        p_curr_star = [p_curr; 1-sum(p_curr)]
        ϕ_curr = get_ϕ(Shrinkage_DAR_curr.v, Shrinkage_DAR_curr.z)
        η = get_TransitionsArray(ϕ_curr[2:(P_curr + 1)], p_curr_star)

        # ----- draw states (forward backward)
		state_CartIdx = state_CartIdx_all[P_curr]
        temp = sampleStates(obs, emissions_curr, η, state_CartIdx)
        indSeq = temp[:indSeq]
        totSeq = temp[:totSeq]
        classProb_sample[:, :, tt] = temp[:classProb]
        stateSeq_sample[:, tt] = temp[:stateSeq]
        γ_est = temp[:stateSeq]

        #println("tt: $tt - ", totSeq)

        # ----- draw mean and GHS precision
        temp_emissions =  sggmDAR_sampleEmissions(obs, indSeq, totSeq, ind_all,
                                                emissions_curr, hyper_parms)
        emissionParms_sample[tt].mean = temp_emissions.mean
        emissionParms_sample[tt].precision = temp_emissions.precision

        # ----- draw Shrinkage DAR parms
        temp_ShrinkageDAR = sampleShrinkageDARparms(ShrinkageDAR_sample[tt-1],
                                                    γ_est,
                                                    hyper_parms,
													tt,
													burn_MCMC_z)
        ShrinkageDAR_sample[tt].z = temp_ShrinkageDAR[:parms].z
        ShrinkageDAR_sample[tt].v = temp_ShrinkageDAR[:parms].v
        ShrinkageDAR_sample[tt].p = temp_ShrinkageDAR[:parms].p
        P_sample[tt] = temp_ShrinkageDAR[:P]
    end

    posteriorSample = PosteriorSample(emissionParms_sample,
                                      ShrinkageDAR_sample,
                                      stateSeq_sample,
                                      classProb_sample)


    return mvShrinkageDAR_Fit(posteriorSample, obs)
end

# -
function sggmDAR_ExtractFit(mvShrinkageDAR_fit::mvShrinkageDAR_Fit;
								   burn_MCMC::Int64 = 1000, thin_MCMC::Int64 = 1,
								   plt::Bool = false)

	n_MCMC = length(mvShrinkageDAR_fit.posteriorSample.ShrinkageDAR)

	# - sample (burn + thin)
	posterior_sample = mvShrinkageDAR_fit.posteriorSample
	emissions_sample = posterior_sample.emissions[burn_MCMC:thin_MCMC:n_MCMC]
	DAR_sample = posterior_sample.ShrinkageDAR[burn_MCMC:thin_MCMC:n_MCMC]
	stateSeq_sample = posterior_sample.stateSeq[:, burn_MCMC:thin_MCMC:n_MCMC]
	classProb_sample = posterior_sample.classProb[:, :, burn_MCMC:thin_MCMC:n_MCMC]
	n_sample = length(emissions_sample)

	# - useful
	P_max = size(DAR_sample[1].v, 1)
	D = size(emissions_sample[1].mean, 1)
	K = size(emissions_sample[1].mean, 2)
	T = size(classProb_sample, 1)

	# sample n of states + modal number of states
	n_states_sample = [length(unique(stateSeq_sample[:, tt])) for tt in 1:n_sample]
	K_hat = mode(n_states_sample)
	



	# sample P (active DAR lags)
	P_sample = Array{Int64}(undef, n_sample)
	for tt in 1:n_sample
		if (length(findall(DAR_sample[tt].z .!= 0)) != 0)
			P_sample[tt] = findall(DAR_sample[tt].z .!= 0)[1]
		else
			P_sample[tt] = P_max
		end
	end

	# κ^2 from GHS
	κ²_sample = ones(Float64, D, D, K_hat, n_sample)
	for tt in 1:n_sample
		for kk in 2:K_hat
			temp = emissions_sample[tt].precision[kk].Λ²
			κ² =  1 ./ (1 .+ temp .* emissions_sample[tt].precision[kk].τ²)
			κ²_sample[:, :, kk, tt] = κ²
		end
	end

	# - modal number of DAR order + idxs
	P_hat = mode(P_sample)
	idxs_P_hat = findall(P_sample .== P_hat)

	# idxs:  number of DAR order and n. of states
	idxs_P_hat_K_hat =  findall((n_states_sample .== K_hat) .& (P_sample .== P_hat))
	n_modal = length(idxs_P_hat_K_hat) # n_sample included in modal conditioning

	#  stateSeq and classProb (condition on modal number of P and K) + mapping to 1:K_hat
	stateSeq_sample_modal = stateSeq_sample[:, idxs_P_hat_K_hat]
	idx_sort = zeros(Int64, K_hat, n_modal)
	unique_states_sort =  zeros(Int64, K_hat, n_modal)

	# idxs unique states, (conditioned on modal number of DAR order and n. of states)
	n_states_idxs_modal = zeros(Int64, K_hat, n_modal)
	for tt in 1:n_modal
		n_states_idxs_modal[:, tt] = unique(stateSeq_sample_modal[:, tt])
	end

	# relabling from 1 to K_hat
	for tt in 1:n_modal
		unique_states_sort[:, tt] = sort(unique(stateSeq_sample_modal[:, tt]))
		idx_sort[:, tt] = sortperm(unique(stateSeq_sample_modal[:, tt]))
		for kk in 1:K_hat
			stateSeq_sample_modal[stateSeq_sample_modal[:, tt] .==  unique_states_sort[kk, tt], tt] .= kk
		end
	end


	classProb_sample_modal_temp = classProb_sample[:, :, idxs_P_hat_K_hat]
	classProb_sample_modal = zeros(Float64, T, K_hat, n_modal)

	for tt in 1:n_modal
		unique_states = n_states_idxs_modal[:, tt]
		classProb_sample_modal[:, :, tt] = classProb_sample_modal_temp[:,
											 unique_states, tt]
		classProb_sample_modal[:, :, tt] =  classProb_sample_modal[:,
							:, tt]./sum(classProb_sample_modal[:, :, tt], dims = 2)

		classProb_sample_modal[:, :, tt] = classProb_sample_modal[:, idx_sort[:, tt], tt]

	end



	# stateSeq and DAR sample (conditioned on modal number of DAR order and n. of states)
	DAR_sample_modal = DAR_sample[idxs_P_hat_K_hat]

	# -
	κ²_sample_modal = κ²_sample[:, :, :, idxs_P_hat_K_hat]

	# posterior ϕ (conditioned on modal number of DAR order and n. of states)
	ϕ_sample_modal = zeros(Float64, P_hat+1, n_modal)
	for tt in 1:n_modal
		ϕ_sample_modal[:, tt] = get_ϕ(DAR_sample_modal[tt].v,
									  DAR_sample_modal[tt].z)[1:(P_hat+1)]
	end

	# posterior π (p) (conditioned on modal number of DAR order and n. of states)
	p_sample_modal = Array{Float64}(undef, K_hat, n_modal)
	p_sample_modal_temp = reduce(hcat, [DAR_sample[tt].p
							for tt in idxs_P_hat_K_hat ])
	p_sample_modal_temp = vcat(p_sample_modal_temp, [1 - sum(p_sample_modal_temp[:, tt])
							for tt in 1:length(idxs_P_hat_K_hat)]')
	for tt in 1:n_modal
		aux = n_states_idxs_modal[:, tt]
		p_sample_modal[:, tt] = p_sample_modal_temp[aux, tt]
		p_sample_modal[:, tt] = p_sample_modal[idx_sort[:, tt], tt]
	end

	# posterior emissions (conditioned on modal number of DAR order and n. of states)
	emissions_sample_modal = emissions_sample[idxs_P_hat_K_hat]

	μ_sample_modal = Array{Float64}(undef, D, K_hat, n_modal)
	Σ_sample_modal = Array{Float64}(undef, D, D, K_hat,  n_modal)
	Ω_sample_modal = Array{Float64}(undef, D, D, K_hat, n_modal)
	Ω_part_sample_modal =  Array{Float64}(undef, D, D, K_hat, n_modal)

	for tt in 1:n_modal
		unique_states = n_states_idxs_modal[:, tt]
		μ_sample_modal[:, :, tt] = emissions_sample_modal[tt].mean[:, unique_states]
		μ_sample_modal[:, :, tt] = μ_sample_modal[:, idx_sort[:, tt], tt]

		for kk in 1:K_hat
			Σ_sample_modal[:, :, kk, tt] = emissions_sample_modal[tt].precision[
															unique_states[kk]].Σ
			Ω_sample_modal[:, :, kk, tt] = emissions_sample_modal[tt].precision[
															unique_states[kk]].Ω
			Ω_part_sample_modal[:, :, kk, tt] = getPartialCorrelation(emissions_sample_modal[tt
															].precision[unique_states[kk]].Ω)
		end

		Σ_sample_modal[:, :, :, tt] = Σ_sample_modal[:, :, idx_sort[:, tt], tt]
		Ω_sample_modal[:, :, :, tt] = Ω_sample_modal[:, :, idx_sort[:, tt], tt]
		Ω_part_sample_modal[:, :, :, tt] = Ω_part_sample_modal[:, :, idx_sort[:, tt], tt]
	end

	# -- Bayes Estimates (conditioned on modal number of DAR order and n. of states)

	#  emissions parms
	μ̂ = reshape(mean(μ_sample_modal, dims = 3), D, K_hat)
	Σ̂ = reshape(mean(Σ_sample_modal, dims = 4), D, D, K_hat)
	Ω̂ = reshape(mean(Ω_sample_modal, dims = 4), D, D, K_hat)
	Ω̂_part = reshape(mean(Ω_part_sample_modal, dims = 4), D, D, K_hat)
	κ²_hat = reshape(mean(κ²_sample_modal, dims = 4), D, D, K_hat)


	#  DAR parms
	ϕ̂ = vec(mean(ϕ_sample_modal, dims = 2))
	p̂ = vec(mean(p_sample_modal, dims = 2))

	# class probs
	classProb_hat = reshape(mean(classProb_sample_modal, dims = 3), T, K_hat)


	# return values

	emissions_hat = Dict(:μ̂ => μ̂, :Σ̂ => Σ̂, :Ω̂ => Ω̂, :Ω̂_part => Ω̂_part,
						  :κ²_hat => κ²_hat)
	DAR_hat = Dict(:ϕ̂ => ϕ̂, :p̂ => p̂)
	bayes_estimates = Dict(:emissions => emissions_hat,
						   :DAR => DAR_hat,
						   :P̂ => P_hat,
						   :K̂ => K_hat,
						   :classProb => classProb_hat)
	posterior_modal = Dict(:μ => μ_sample_modal, :Σ => Σ_sample_modal,
						   :Ω => Ω_sample_modal,
						   :Ω_part => Ω_part_sample_modal,
						   :κ² => κ²_sample_modal,
						   :ϕ => ϕ_sample_modal, :p => p_sample_modal,
						   :classProb => classProb_sample_modal,
						   :stateSeq => stateSeq_sample_modal)
	posterior_n_components = Dict(:P => P_sample, :K => n_states_sample)
	posterior = PosteriorSample(emissions_sample, DAR_sample,
								stateSeq_sample, classProb_sample)


	return Dict(:posterior => posterior,
				:posterior_modal => posterior_modal,
				:posterior_n_components => posterior_n_components,
				:bayes_estimates => bayes_estimates
				)
end


# -
function ReshapeFit(fit_extract; relabel = true, stateSeq_true = nothing)

	if relabel
		fit_extract_relabeled = relabel_parameters(fit_extract; stateSeq_true = nothing)
	else
		fit_extract_relabeled = nothing
	end

	# -------- sample
	P_sample = fit_extract[:posterior_n_components][:P]
	K_sample = fit_extract[:posterior_n_components][:K]
	ϕ_sample = fit_extract[:posterior_modal][:ϕ]
	if relabel
		p_sample = fit_extract_relabeled[:sample][:p]
		μ_sample = fit_extract_relabeled[:sample][:emissions][:μ]
		Σ_sample = fit_extract_relabeled[:sample][:emissions][:Σ]
		Ω_sample = fit_extract_relabeled[:sample][:emissions][:Ω]
		κ²_sample = fit_extract_relabeled[:sample][:emissions][:κ²]
		Ω_part_sample = fit_extract_relabeled[:sample][:emissions][:Ω_part]
		classProb_sample =  fit_extract_relabeled[:sample][:classProb]
		stateSeq_sample = fit_extract_relabeled[:stateSeq_sample]

	else
		p_sample = fit_extract[:posterior_modal][:p]
		μ_sample = fit_extract[:posterior_modal][:μ]
		Σ_sample = fit_extract[:posterior_modal][:Σ]
		Ω_sample = fit_extract[:posterior_modal][:Ω]
		κ²_sample = fit_extract[:posterior_modal][:κ²]
		Ω_part_sample = fit_extract[:posterior_modal][:Ω_part]
		classProb_sample = fit_extract[:posterior_modal][:classProb]
		stateSeq_sample = fit_extract[:posterior_modal][:stateSeq]
	end

	# -- bayes estimates

	K̂ = fit_extract[:bayes_estimates][:K̂]
	P̂ = fit_extract[:bayes_estimates][:P̂]
	ϕ̂ = fit_extract[:bayes_estimates][:DAR][:ϕ̂]

	if relabel
		p̂ = fit_extract_relabeled[:estimates][:p_hat]
		μ̂ = fit_extract_relabeled[:estimates][:emissions_hat][:μ̂]
		Σ̂ = fit_extract_relabeled[:estimates][:emissions_hat][:Σ̂]
		Ω̂ = fit_extract_relabeled[:estimates][:emissions_hat][:Ω̂]
		κ²_hat = fit_extract_relabeled[:estimates][:emissions_hat][:κ²_hat]
		Ω̂_part = fit_extract_relabeled[:estimates][:emissions_hat][:Ω̂_part]
		classProb_hat = fit_extract_relabeled[:estimates][:classProb_hat]
	else
		p̂ = fit_extract[:bayes_estimates][:DAR][:p̂]
		μ̂ = fit_extract[:bayes_estimates][:emissions][:μ̂]
		Σ̂ = fit_extract[:bayes_estimates][:emissions][:Σ̂]
		Ω̂ = fit_extract[:bayes_estimates][:emissions][:Ω̂]
		κ²_hat = fit_extract[:bayes_estimates][:emissions][:κ²_hat]
		Ω̂_part = fit_extract[:bayes_estimates][:emissions][:Ω̂_part]
		classProb_hat =fit_extract[:bayes_estimates][:classProb]
	end

	posterior_sample = Dict(:P => P_sample, :K => K_sample,
				  			:ϕ => ϕ_sample, :p => p_sample,
				  			:μ => μ_sample, :Σ => Σ_sample,
							:Ω => Ω_sample,
							:κ² => κ²_sample,
				  			:Ω_part => Ω_part_sample,
				  			:classProb => classProb_sample,
				  			:stateSeq => stateSeq_sample)
	bayes_est = Dict(:P̂ => P̂, :K̂ => K̂, :ϕ̂ => ϕ̂, :p̂ => p̂,
					 :μ̂ => μ̂, :Σ̂ => Σ̂, :Ω̂ => Ω̂, :Ω̂_part => Ω̂_part,
					 :κ²_hat => κ²_hat,
					 :classProb_hat => classProb_hat)

	out = Dict(:posterior_sample => posterior_sample,
			   :bayes_est => bayes_est)

	return out
end


# -

function sggmDAR_PosteriorPredictive(mvShrinkageDAR_fit::mvShrinkageDAR_Fit, K;
	burn_MCMC = 100, thin_MCMC = 1,
	n_draw::Int64 = 20, relabel::Bool = true, modal::Bool = true, plt::Bool = true)

	obs = mvShrinkageDAR_fit.obs
	fit_extract = sggmDAR_ExtractFit(mvShrinkageDAR_fit,
	                 burn_MCMC = burn_MCMC, thin_MCMC = thin_MCMC,
	                 plt = false)

	fit_reshaped = ReshapeFit(fit_extract; relabel = relabel,
	 						  stateSeq_true = nothing)
	n_sample_modal = size(fit_reshaped[:posterior_sample][:μ], 3)
	n_sample_raw = length(fit_extract[:posterior].emissions)

	D = size(fit_reshaped[:posterior_sample][:μ], 1)
	T = size(obs, 1)
	K̂ = fit_reshaped[:bayes_est][:K̂]
	P̂ = fit_reshaped[:bayes_est][:P̂]
	state_CartIdx_all = Dict((pp) => get_StateCartesianIndices(K̂, pp)
							 for pp in 1:P̂)
	if (n_draw > n_sample_modal) stop("n_draw is larger than n_MCMC") end


	# DAR parms
	ϕ̂ = fit_reshaped[:bayes_est][:ϕ̂]
	p̂ = fit_reshaped[:bayes_est][:p̂]
	# emission parms
	μ̂ = fit_reshaped[:bayes_est][:μ̂]
	Σ̂ = fit_reshaped[:bayes_est][:Σ̂]
	Ω̂_part = fit_reshaped[:bayes_est][:Ω̂_part]
	# - aux GHS (emissions)

	GHS_hat = [GHS() for kk in 1:K̂]
	for kk in 1:K̂
		GHS_hat[kk].Σ = Σ̂[:, :, kk]
	end

	# most likely state sequence:
	γ̂ = sampleStates(obs, EmissionParms(μ̂, GHS_hat),
					 get_TransitionsArray(ϕ̂[2:(P̂+1)], p̂),
					 state_CartIdx_all[P̂]; maximize = true)

	# -
	ŷ = Array{Float64}(undef, T, D, n_draw)
	idxs_modal = sample(1:n_sample_modal, n_draw)
	idxs_raw = sample(1:n_sample_raw, n_draw)

	println("...Sampling from Posterior Predictive...")
	@inbounds @showprogress for tt in 1:n_draw
		if modal
			γ_temp = fit_reshaped[:posterior_sample][:stateSeq][:, idxs_modal[tt]]
			μ_temp = fit_reshaped[:posterior_sample][:μ][:, :, idxs_modal[tt]]
			Σ_temp = fit_reshaped[:posterior_sample][:Σ][:, :, :, idxs_modal[tt]]
			for n in 1:T
				ŷ[n, :, tt] = rand(MultivariateNormal(μ_temp[:,
									γ_temp[n]], Σ_temp[:, :, γ_temp[n]]))
			end
		else
			γ_temp = fit_extract[:posterior].stateSeq[:, idxs_raw[tt]]
			μ_temp = fit_extract[:posterior].emissions[idxs_raw[tt]].mean
			GHS_temp = fit_extract[:posterior].emissions[idxs_raw[tt]].precision
			for n in 1:T
				ŷ[n, :, tt] = rand(MultivariateNormal(μ_temp[:, γ_temp[n]],
									 GHS_temp[γ_temp[n]].Σ))
			end
		end
	end

	n_states_final = ifelse(modal, K̂, K)
	#n_states_final = K̂
	if plt
		sggmDAR_PlotPosteriorPredictive(obs, n_states_final, ŷ, γ̂; path_file = nothing)
	end
	signal_hat = reshape(mean(ŷ, dims = 3), T, D)

	return Dict(:ŷ => ŷ,
				:signal_hat => signal_hat,
				:γ̂ => γ̂)
end


# - ?
function sggmDAR_PlotPosteriorPredictive(obs::Matrix, n_states::Int,
									   ŷ::Array, γ̂::Vector{Int}; path_file = nothing,
									   ylabels = nothing)

    N = size(obs, 1)
    D = size(obs, 2)
    K = copy(n_states)

	save = ifelse(!isnothing(path_file), true, false)
	change_ylabels = ifelse(!isnothing(ylabels), true, false)
    changePoints = vcat(0, findall(diff(γ̂) .!= 0), N)
	# colorPalette = palette(:roma, length(unique(γ̂)))
    # colorPalette = palette(:roma, K)
	#colorPalette = palette(:lighttest, K)
	colorPalette = palette(:rainbow1, K)
	# candidate: colorPalette = palette(:seaborn_bright6, K)
	#colorPalette = palette(:jet, K)


	if (D <= 20)

		n_col_plots = Int64(ceil(D / 5))
		plots = plot(layout = (5,  n_col_plots))

		for jj = 1:n_col_plots
			for dd = 1:5
				if (((dd + 5*(jj-1))) <= D)

					if change_ylabels
						ylab = ylabels[dd + 5*(jj-1)]
					else
						ylab = "\$ y_{$(dd + 5*(jj-1))}\$"
					end
					for tt = 1:size(ŷ, 3)
						plot!(
							plots[dd, jj],
							ŷ[:, (dd + 5*(jj-1)), tt],
							color = :grey,
							legend = :topright,
							label = "",
							alpha = 0.02,
							xlims = (1, N),
							xticks = [1, N],
							xtickfont = font(4),
							ytickfont = font(4),
							yguidefontsize=8,
							dpi = 1000,
						)
					end
					scatter!(
						plots[dd, jj],
						1:N,
						obs[:, (dd + 5*(jj-1))],
						marker_z = γ̂,
						legend = nothing,
						colorbar = false,
						markersize = 0.7,
						alpha = 2,
						c = colorPalette,
						ylabel = ylab,
						markerstrokewidth = 0,
					)
					for ii = 1:(length(changePoints)-1)
						z_span = [changePoints[ii] + 1, changePoints[ii+1]]
						vspan!(
							plots[dd, jj],
							z_span,
							color = colorPalette.colors[γ̂[changePoints[ii+1]]],
							alpha = 0.15,
							labels = :none,
							dpi = 1000
						)
					end
				else
					plot!(plots[dd, jj],
						  legend=false, grid=false,
						  foreground_color_subplot=:white)
				end
			end
		end
		if save
			savefig(plots, path_file * ".png")
		end
		display(plots)

	else

		D_aux = 0

		while (D_aux <= D)

			n_col_plots = 4
			plots = plot(layout = (5,  n_col_plots))

			for jj = 1:n_col_plots
				for dd = 1:5
					if ((((dd+D_aux) + 5*(jj-1))) <= D)
						for tt = 1:size(ŷ, 3)
							plot!(
								plots[dd, jj],
								ŷ[:, ((dd+D_aux) + 5*(jj-1)), tt],
								color = :grey,
								legend = :topright,
								label = "",
								alpha = 0.3,
								xlims = (1, N),
								xticks = [1, N],
								xtickfont = font(4),
								ytickfont = font(4),
								dpi = 600
							)
						end
						scatter!(
							plots[dd, jj],
							1:N,
							obs[:, ((dd+D_aux) + 5*(jj-1))],
							marker_z = γ̂,
							legend = nothing,
							colorbar = false,
							markersize = 0.3,
							alpha = 0.3,
							c = colorPalette,
							ylabel = "\$ y_{$((dd+D_aux) + 5*(jj-1))}\$",
							markerstrokewidth = 0,
							dpi = 600
						)
						for ii = 1:(length(changePoints)-1)
							z_span = [changePoints[ii] + 1, changePoints[ii+1]]
							vspan!(
								plots[dd, jj],
								z_span,
								color = colorPalette.colors[γ̂[changePoints[ii+1]]],
								alpha = 0.2,
								labels = :none,
								dpi = 600
							)
						end
					else
						plot!(plots[dd, jj],
							  legend=false, grid=false,
							  foreground_color_subplot=:white)
					end
				end
			end
			savefig(plots, path_file *  "$D_aux.png")
			display(plots)

			D_aux += 20

			if (D_aux < D)
				println("press enter to see the next plot ")
				readline()
			end


		end

	end
end

# - 
function sggmDAR_PlotPosteriorPredictive_Single(obs::Matrix, n_states::Int, ŷ::Array, 
                                              γ̂::Vector{Int}; path_file = nothing, 
                                              ylabels = nothing, state_seq_true = nothing)

    
    N = size(obs, 1)
    D = size(obs, 2)
    K = copy(n_states)

    save = ifelse(!isnothing(path_file), true, false)
    plot_state_seq_true = ifelse(!isnothing(state_seq_true), true, false)
    change_ylabels = ifelse(!isnothing(ylabels), true, false)
    changePoints = vcat(0, findall(diff(γ̂) .!= 0), N)
    # colorPalette = palette(:roma, length(unique(γ̂)))
    # colorPalette = palette(:roma, K)
    #colorPalette = palette(:lighttest, K)
    colorPalette = palette(:rainbow1, K)
    # candidate: colorPalette = palette(:seaborn_bright6, K)
    #colorPalette = palette(:jet, K)

    if plot_state_seq_true 
        state_seq_temp = state_seq_true
    else 
        state_seq_temp = γ̂
    end 

    plots = plot(layout = (1, 1))
	plot!(plots, size=(1000,600))

    for dd in 1:D 
        for tt = 1:size(ŷ, 3)
            plot!(
                plots[1, 1],
                ŷ[:, dd, tt],
                color = :grey,
                legend = :topright,
                label = "",
                alpha = 0.01,
                xlims = (-10, N+10),
                xticks = [1, 500, 1000, 1500, 2000],
                xtickfont = font(10),
                ytickfont = font(10),
                yguidefontsize=13,
                dpi = 1000,
                ylab = "y",
                xlab = "Time"
            )
        end
        scatter!(
        plots[1, 1],
        1:N,
        obs[:, dd],
        marker_z = state_seq_temp,
        legend = nothing,
        colorbar = false,
        markersize = 0.8,
        alpha = 2,
        c = colorPalette,
        markerstrokewidth = 0,
        )

    end 



    for ii = 1:(length(changePoints)-1)
        z_span = [changePoints[ii] + 1, changePoints[ii+1]]
        vspan!(
            plots[1, 1],
            z_span,
            color = colorPalette.colors[γ̂[changePoints[ii+1]]],
            alpha = 0.08,
            labels = :none,
            dpi = 1000
        )
    end 

    if save
        savefig(plots, path_file * ".png")
    end
    display(plots)

end 



# - ?
function logLikelihoodSystem(obs, γ, μ, Σ, η)

	N = size(obs, 1)
	K̂ = size(η, 1)
	P̂ = length(size(η)) - 1

	N_star = length((P̂+1):N)
	temp_transitions = Array{Float64}(undef, N_star)
	temp_emissions = Array{Float64}(undef, N_star)

	ii = 1
	for nn in (P̂+1):N
		transition_prob = [η[CartesianIndex(Tuple(vcat(γ[nn-P̂:(nn-1)], jj)))] for jj in 1:K̂]
		temp_transitions[ii] = log(transition_prob[γ[nn]])
		temp_emissions[ii] = log(pdf(MultivariateNormal(μ[:, γ[nn]], Σ[:, :, γ[nn]]), obs[nn, :]))
		ii += 1
	end

	out = sum(temp_transitions .+ temp_emissions)
	return out
end

# -
function tracesConvergence(obs, fit_reshaped; skip_trace = 5, plt = true,
	 					   path_file = nothing)

	save_plt = ifelse(!isnothing(path_file), true, false)
	n_sample = size(fit_reshaped[:posterior_sample][:stateSeq], 2)
	trace_llk = zeros(0)

	println("...Computing Traces Likelihood System ")
	@showprogress for tt in  1:skip_trace:n_sample
		γ = fit_reshaped[:posterior_sample][:stateSeq][:, tt]
		μ = fit_reshaped[:posterior_sample][:μ][:, :, tt]
		Σ = fit_reshaped[:posterior_sample][:Σ][:, :, :, tt]
		η = get_TransitionsArray(fit_reshaped[:posterior_sample][:ϕ][2:end, tt],
							 	 fit_reshaped[:posterior_sample][:p][:, tt])
	    append!(trace_llk, logLikelihoodSystem(obs, γ, μ, Σ, η))
	end

	if plt
		plt_out = plot(trace_llk, legend =nothing, xlabel = "Iterations", ylabel = "Log-Likelihood",
			 xtickfont = font(10), ytickfont = font(10),
			 dpi = 300)
	    display(plt_out)
		if save_plt
			savefig(plt_out, path_file)
		end
	end

	return trace_llk
end




# -- sample Shrinkage parms
function sampleShrinkageDARparms(parms::ShrinkageDAR_Parms,
                                 γ::Array,
                                 hyper_parms::Hyper_Parms,
								 tt::Int,
								 burn_MCMC_z::Int)

    n_states = length(parms.p)
    w = hyper_parms.w
    max_steps = hyper_parms.max_steps
    p_curr = parms.p
    p_curr_star = [p_curr; 1-sum(p_curr)]
    ϕ_curr = get_ϕ(parms.v, parms.z)

    # -- update z
    temp_parms = ShrinkageDAR_Parms(parms.v, parms.z, parms.p)
	if tt < burn_MCMC_z
		temp = ShrinkageDAR_sample_z_AllMoves(temp_parms, γ, hyper_parms)
	else
		temp = ShrinkageDAR_sample_z(temp_parms, γ, hyper_parms)
	end
    v_curr = temp[:v]
    z_new = temp[:z]
    P_new = temp[:P]

    # - update v (slice sampler - one at time )
    v_out = copy(v_curr)
    @inbounds for jj in 1:P_new
        y = log_posterior_v_and_z_and_p(γ, v_out, z_new,
                                        p_curr, hyper_parms) - rand(Exponential(1))
        interval = get_interval("v", jj, γ, v_out, z_new, p_curr,  y, w, max_steps, hyper_parms)
        v_new = shrink_and_sample("v", jj, γ, v_out, z_new, p_curr, y, interval, hyper_parms)
        v_out[jj] = copy(v_new)
    end
    v_new = copy(v_out)

    # - draw π (slice sampler - one at time)
    @inbounds for jj in 1:n_states
        y = log_posterior_v_and_z_and_p(γ, v_new, z_new, p_curr, hyper_parms) - rand(Exponential(1))
        interval = get_interval("p", jj, γ, v_new, z_new, p_curr,  y, w, max_steps, hyper_parms)
        p_new = shrink_and_sample("p", jj, γ, v_new, z_new, p_curr, y, interval, hyper_parms)
        p_curr[jj] = copy(p_new)
    end
    p_new = copy(p_curr)

    parms_new = ShrinkageDAR_Parms(v_new, z_new, p_new)

    return Dict(:parms => parms_new, :P => P_new)
end


# -
function sggmDAR_sampleEmissions(obs::Matrix, indSeq::Matrix, totSeq::Vector,
							   ind_all::Matrix,
                               emissions::EmissionParms, hyper_parms::Hyper_Parms)

    μ_0 = hyper_parms.μ_0
    R_0 = hyper_parms.R_0
	n_min = hyper_parms.n_min


    N = size(obs, 1)
    D = size(obs, 2)
    K = length(totSeq)

    out = EmissionParms(zeros(Float64, D, K), [GHS() for kk in 1:K])

    @inbounds for kk in 1:K
        if (totSeq[kk] > n_min)
            N_state = totSeq[kk]
            obsInd = @view(indSeq[1:N_state, kk])
            obs_state = obs[obsInd, :]
            GHS_state = emissions.precision[kk]
            temp = sampleEmissions(obs_state, GHS_state, hyper_parms, ind_all)

            out.mean[:, kk] = temp[:mean]
            out.precision[kk] = temp[:precision]
        else
            temp = sampleEmissions_fromPrior(hyper_parms)
            out.mean[:, kk] = temp[:mean]
            out.precision[kk] = temp[:precision]
        end
    end

    return out
end

# - sample z ( birth death within)
function ShrinkageDAR_sample_z(parms::ShrinkageDAR_Parms, γ::Array, hyper_parms::Hyper_Parms)

    z_curr = parms.z
    v_curr = parms.v
    p_curr = parms.p
	P_curr = numberActiveComponents(z_curr)
	P_max = length(v_curr)


    if (P_curr == 1)
        # ------------------------- only within or birth  ---------------------- #

		U = rand()
		if (U < 0.5) 
			z_new = z_curr
			v_new = v_curr
			P_new = P_curr
		else 
			P_birth = P_curr + 1
			z_birth = copy(z_curr); z_birth[P_curr] = false
			v_birth = copy(v_curr)
			if P_birth != P_max
				v_birth[P_birth + 1] = 1
			end
			v_birth[P_birth] = rand(Beta(hyper_parms.α_ϕ_AR[1], hyper_parms.α_ϕ_AR[2]))

			# - birth
			logpost_prop = log_posterior_v_and_z_and_p(γ, v_birth, z_birth, p_curr, hyper_parms)
			# - curr
			logpost_curr = log_posterior_v_and_z_and_p(γ, v_curr, z_curr, p_curr, hyper_parms)
			# proposal ratio: 
			logprop_ratio = log(1) - log(pdf(Beta(hyper_parms.α_ϕ_AR[1], hyper_parms.α_ϕ_AR[2]), v_birth[P_birth]))
	
			MH_ratio = min(1, exp(logpost_prop - logpost_curr + logprop_ratio))
			U = rand()
	
			if (U < MH_ratio) 
				z_new = z_birth
				v_new = v_birth
				P_new = P_birth
			else 
				z_new = z_curr
				v_new = v_curr
				P_new = P_curr
			end 
		end 



    elseif (P_curr == P_max)
        # ------------------------- only within or death ------------------------- #

		U = rand()
		if (U < 0.5) 
			z_new = z_curr
			v_new = v_curr
			P_new = P_curr
		else 
			P_death = P_curr - 1
			z_death = copy(z_curr); z_death[P_death] = true
			v_death = copy(v_curr); v_death[P_curr] = 1
	
			# - prop
			logpost_prop = log_posterior_v_and_z_and_p(γ, v_death, z_death, p_curr, hyper_parms)
			# - curr
			logpost_curr = log_posterior_v_and_z_and_p(γ, v_curr, z_curr, p_curr, hyper_parms)
			# proposal ratio: 
			logprop_ratio = log(pdf(Beta(hyper_parms.α_ϕ_AR[1], hyper_parms.α_ϕ_AR[2]), v_curr[P_curr])) - log(1)
		
			MH_ratio = min(1, exp(logpost_prop - logpost_curr + logprop_ratio))
			U = rand()
	
			if (U < MH_ratio) 
				z_new = z_death
				v_new = v_death
				P_new = P_death
			else 
				z_new = z_curr
				v_new = v_curr
				P_new = P_curr
			end 
		end 

    else
        # ------------------------- within or (birth/death)   -------------------- #

		U = rand()
		if (U < 0.5)
			# within
			z_new = z_curr
			v_new = v_curr
			P_new = P_curr
		else 
			U = rand() # birth/death 
			if (U < 0.5) 
				# birth
				P_birth = P_curr + 1
				v_birth = copy(v_curr)
				z_birth = copy(z_curr); z_birth[P_curr] = false
				if (P_birth != P_max)
    				v_birth[P_birth + 1] = 1;
				end
				v_birth[P_birth] = rand(Beta(hyper_parms.α_ϕ_AR[1], hyper_parms.α_ϕ_AR[2]))
				# - prop
				logpost_prop = log_posterior_v_and_z_and_p(γ, v_birth, z_birth, p_curr, hyper_parms)
				# - curr
				logpost_curr = log_posterior_v_and_z_and_p(γ, v_curr, z_curr, p_curr, hyper_parms)
				# proposal ratio: 
				logprop_ratio = log(1) - log(pdf(Beta(hyper_parms.α_ϕ_AR[1], hyper_parms.α_ϕ_AR[2]), v_birth[P_birth]))
	
				MH_ratio = min(1, exp(logpost_prop - logpost_curr + logprop_ratio))
				U = rand()

				if (U < MH_ratio) 
    				z_new = z_birth
    				v_new = v_birth
    				P_new = P_birth
				else 
    				z_new = z_curr
    				v_new = v_curr
    				P_new = P_curr
				end 
			else 
				
				P_death = P_curr - 1
				z_death = copy(z_curr); z_death[P_death] = true
				v_death = copy(v_curr); v_death[P_curr] = 1
		
				# - prop
				logpost_prop = log_posterior_v_and_z_and_p(γ, v_death, z_death, p_curr, hyper_parms)
				# - curr
				logpost_curr = log_posterior_v_and_z_and_p(γ, v_curr, z_curr, p_curr, hyper_parms)
				# proposal ratio: 
				logprop_ratio = log(pdf(Beta(hyper_parms.α_ϕ_AR[1], hyper_parms.α_ϕ_AR[2]), v_curr[P_curr])) - log(1)
			
				MH_ratio = min(1, exp(logpost_prop - logpost_curr + logprop_ratio))
				U = rand()
		
				if (U < MH_ratio) 
					z_new = z_death
					v_new = v_death
					P_new = P_death
				else 
					z_new = z_curr
					v_new = v_curr
					P_new = P_curr
				end 
			end 
		end 
    end

    out = Dict(:z => z_new, :v => v_new, :P => P_new)

    return out
end


# - sample z (all possible moves)
function ShrinkageDAR_sample_z_AllMoves(parms::ShrinkageDAR_Parms, γ::Array,
                                        hyper_parms::Hyper_Parms)

    z_curr = parms.z
    v_curr = parms.v
    p_curr = parms.p
    P_curr = numberActiveComponents(z_curr)
    P_max = length(v_curr)

    z_prop_ALL = zeros(Bool, P_max, P_max-1)
    v_prop_ALL = ones(Float64, P_max, P_max)

    # -- z prop ALL
    for jj in 1:(P_max-1)
        z_prop_ALL[jj, jj:end] = ones(Bool, P_max-jj)
    end

    # -- v prop ALL
    v_prop_ALL[P_curr, : ] = v_curr
    # - deaths
    for jj in (P_curr-1):-1:1
        v_prop_ALL[jj, :] = v_prop_ALL[jj+1, :]
        v_prop_ALL[jj, jj+1] = 1.0
        if (jj != (P_max-1))
            v_prop_ALL[jj, jj+2] = NaN
        end
    end
    # - births
    for jj in (P_curr+1):P_max
        v_prop_ALL[jj, :] = v_prop_ALL[jj-1, :]
        v_prop_ALL[jj, jj] = rand(Beta(hyper_parms.α_ϕ_AR[1], hyper_parms.α_ϕ_AR[2]))
        if (jj != P_max)
            v_prop_ALL[jj, jj+1] = 1.0
        end
    end

    # -- evaluating probabibilities  z
    z_logprobs_ALL = zeros(Float64, P_max)
    for jj in 1:P_max
        z_logprobs_ALL[jj] = log_posterior_v_and_z_and_p(γ, v_prop_ALL[jj, :],
                                                         z_prop_ALL[jj, :], p_curr, hyper_parms)
    end
    # -- sampling z_new
    z_aux =  sample(1:P_max,
                    ProbabilityWeights(exp.(z_logprobs_ALL .- logsumexp(z_logprobs_ALL))))


    z_new  = z_prop_ALL[z_aux, :]
    v_new = v_prop_ALL[z_aux, :]
    P_new = numberActiveComponents(z_new)

    out = Dict(:z => z_new, :v => v_new, :P => P_new)

    return out
end


# -
function sampleEmissions(obs_state::Array, GHS_state::GHS,
                         hyper_parms::Hyper_Parms, ind_all::Array)

    N_state = size(obs_state, 1)
    D = size(obs_state, 2)
    μ_0 = hyper_parms.μ_0
    R_0 = hyper_parms.R_0

    ȳ = vec(mean(obs_state, dims = 1))
    if N_state == 1
        obs_centered = copy(obs_state)
    else
        obs_centered = obs_state - repeat(ȳ, outer = [1, N_state])'
    end
    S = obs_centered' * obs_centered

    # current values
    Ω = copy(GHS_state.Ω)
    Σ = copy(GHS_state.Σ)
    Λ² = copy(GHS_state.Λ²)
    ν = copy(GHS_state.ν)
    τ² = copy(GHS_state.τ²)
    ξ = copy(GHS_state.ξ)


    @inbounds for ii in 1:D

        # auxiliary
        ind = ind_all[:, ii]
        Σ_11 = Σ[ind, ind]
        σ_12 = Σ[ind, ii]
        σ_22 = Σ[ii, ii]

        s_21 = S[ind, ii]
        s_22 = S[ii, ii]
        λ²_12 = Λ²[ind, ii]
        ν_12 = ν[ind, ii]

        # - sample γ and β
        γ = rand(Gamma((N_state/2+1), 2/s_22)) # random gamma with shape=n/2+1, rate=s_22/2
        inv_Ω_11 = Σ_11 - σ_12 * σ_12' / σ_22
        inv_C = s_22*inv_Ω_11 + Diagonal(1 ./ (λ²_12 * τ²))
        inv_C_chol = Matrix(cholesky(inv_C).U)

        μ_i = - inv_C \ s_21
        β = μ_i + inv_C_chol \ rand(Normal(), D - 1)
        ω_12 = β
        ω_22 = γ + β' * inv_Ω_11 * β

        # -  sample λ_sq and ν
        rate = (ω_12.^2)/(2*τ²) + 1 ./ν_12
        λ²_12 = 1 ./ [rand(Gamma(1, 1/rate[d])) for d in 1:(D-1)]
        ν_12 = 1 ./ [rand(Gamma(1, 1/(1+1/λ²_12[d]))) for d in 1:(D-1)]

        # -  update Ω, Σ, Λ_sq, \n
        Ω[ii, ind] = ω_12
        Ω[ind, ii] = ω_12
        Ω[ii, ii] = ω_22
        temp = inv_Ω_11 * β
        Σ_11 = inv_Ω_11 + temp * temp' / γ
        σ_12 = -temp/γ
        σ_22 = 1/γ

        Σ[ind, ind] = Σ_11
        Σ[ii, ii] = σ_22
        Σ[ii, ind] = σ_12
        Σ[ind, ii] = σ_12

        Λ²[ii, ind] = λ²_12
        Λ²[ind, ii] = λ²_12
        ν[ii, ind] = ν_12
        ν[ind, ii] = ν_12
    end

    # sample τ and ξ
    ω_vector = Ω[tril(fill(true, (D, D)), -1)]
    λ²_vector = Λ²[tril(fill(true, (D, D)), -1)]
    rate = 1/ξ + sum(ω_vector.^2 ./ (2*λ²_vector))
    τ² = 1/rand(Gamma((D*(D-1)/2 + 1)/2, 1/rate))
    ξ = 1/rand(Gamma(1, 1/(1+1/τ²)))


    # drawn values
    precision_out = GHS()
    precision_out.Ω = Ω
    precision_out.Σ = Σ
    precision_out.Λ² = Λ²
    precision_out.ν = ν
    precision_out.τ² = τ²
    precision_out.ξ = ξ

    # - sample μ
    R_post = R_0  + N_state * Ω
    μ_post = R_post \ (R_0 * μ_0 + N_state * Ω * ȳ)
    U = Matrix(cholesky(R_post).U)
    mean_out = μ_post + U \ rand(Normal(), D)

    return Dict(:mean => mean_out, :precision => precision_out)
end


# -
function sampleEmissions_fromPrior(hyper_parms::Hyper_Parms)

	μ_0 = hyper_parms.μ_0
	R_0 = hyper_parms.R_0

	D = length(μ_0)
	precision_out = GHS()
	draw = true

	while draw

		ξ = rand(Gamma(0.5, 1))
		τ² =  rand(Gamma(0.5, ξ))

		ν = [rand(Gamma(0.5, 1)) for i in 1:Int64(D*(D+1)/2)]
		λ² = [rand(Gamma(0.5, ν[i])) for i in 1:Int64(D*(D+1)/2)]

		ν_mat = Symmetric([i<=j ? ν[Int64(j*(j-1)/2+i)] : 0 for i=1:(D), j=1:(D)])
		ν_mat[diagind(ν_mat)] .= 1.0
		λ²_mat = Symmetric([i<=j ? λ²[Int64(j*(j-1)/2+i)] : 0 for i=1:(D), j=1:(D)])
		λ²_mat[diagind(λ²_mat)] .= 1.0

		ω  = [rand(Normal(0, sqrt(λ²[ii] * τ²))) for ii in 1:Int64(D*(D+1)/2)]
		Ω = Symmetric([i<=j ? ω[Int64(j*(j-1)/2+i)] : 0 for i=1:(D), j=1:(D)])
		# Ω[diagind(Ω)] = rand(Uniform(0.05, 3), D)
		#Ω[diagind(Ω)] = rand(Uniform(1, 3), D)
        Ω[diagind(Ω)] = rand(Uniform(0.00001, 100), D)
		Σ = Matrix(Hermitian(inv(Ω)))

		if isposdef(Σ)
			precision_out.Σ =  Σ
			precision_out.Ω =  Ω
			precision_out.Λ² = λ²_mat
			precision_out.ν = ν_mat
			precision_out.τ² = τ²
			precision_out.ξ = ξ
			draw = false
		end
	end
	mean_out = rand(MultivariateNormal(μ_0, inv(R_0)))

	return Dict(:mean => mean_out, :precision => precision_out)
end



# -
function sampleStates(obs::Array, emissions::EmissionParms, η::Array,
					  state_CartIdx; maximize = false)

	N = size(obs, 1)
	K = size(η, 1)
	P = length(size(η)) - 1

	#
	γ = zeros(Int64, N)
	totSeq = zeros(Int64, K)
	indSeq = zeros(Int64, N, K)
	classProb = Array{Float64}(undef, N, K)

	# Compute likelihood of each observation under each parameter
	likelihood = mvGaussLikelihood(obs, emissions)
	# Compute backward messages
	partial_marg = backwardMessages(likelihood, η, state_CartIdx)[:partial_marg]


    if maximize
        state_est = Array{Int64}(undef, N)
        @inbounds for tt in 1:N
            if (any(tt .== 1:P))
                P_γ = @view(partial_marg[:, 1])
            else
                # slow - might want to vectorize(η) and find a way
                # to pick the indexes.
                transition_prob = [η[CartesianIndex(Tuple(vcat(γ[tt-P:(tt-1)], jj)))] for jj in 1:K]
                P_γ = transition_prob .* @view(partial_marg[:, tt])
            end
            γ[tt] = argmax(P_γ)

        end
        return γ

    else
        @inbounds for tt in 1:N

            #println(tt)

            if (P == 1) # as in HMM
                if (any(tt .== 1:P))
                    P_γ = @view(partial_marg[:, 1])
                else
                    P_γ = @view(η[γ[tt-1], :]) .* @view(partial_marg[:, tt])
                end
            else
                if (any(tt .== 1:P))
                    P_γ = @view(partial_marg[:, 1])
                else
                    # slow - might want to vectorize(η) and find a way
                    # to pick the indexes.
                    transition_prob = [η[CartesianIndex(Tuple(vcat(γ[tt-P:(tt-1)], jj)))] for jj in 1:K]
                    P_γ = transition_prob .* @view(partial_marg[:, tt])
                end
            end

            classProb[tt, :] = P_γ/sum(P_γ)
            P_γ = cumsum(P_γ)
            γ[tt] = 1 + sum(P_γ[end]*rand() .> P_γ)

            # Add z_t to count vector
            totSeq[γ[tt]] = totSeq[γ[tt]] + 1
            indSeq[totSeq[γ[tt]], γ[tt]] = copy(tt)

        end
    end


	output = Dict(:stateSeq => γ, :totSeq => totSeq,
                  :indSeq => indSeq, :classProb => classProb)

    return output
end



function Plot_Data(observations, state_seq; path_file = nothing)

	T = size(observations, 1)
	D = size(observations, 2)
	K = length(unique(state_seq))
	save = ifelse(!isnothing(path_file), true, false)


	colorPalette = palette(:roma, K)

	if (D <= 20)

		n_col_plots = Int64(ceil(D/5))
		plots = plot(layout = (5,  n_col_plots))

		for jj in 1:n_col_plots
			for dd in 1:5
				if (((dd + 5*(jj-1))) <= D)
					scatter!(plots[dd, jj], 1:T,
							observations[:, (dd + 5*(jj-1))], marker_z = state_seq,
							markersize = 1.5, legend=nothing,
							ylabel = "\$ y_{$(dd + 5*(jj-1))}\$",
							c = colorPalette,
							colorbar = false,
							markerstrokewidth=0,
							xlims = (1, T),
							xticks = [1, T],
							xtickfont = font(4),
							ytickfont = font(4),
							dpi = 1000)
				else
					plot!(plots[dd, jj],
						  legend=false, grid=false,
						  foreground_color_subplot=:white)
				end
			end
		end

		if save
			savefig(plots, path_file)
		end
		display(plots)

	else
		D_aux = 0

		while (D_aux <= D)

			n_col_plots = 4
			plots = plot(layout = (5,  n_col_plots))
			gui(plots)

			for jj in 1:n_col_plots
				for dd in 1:5
					if ((((dd+D_aux) + 5*(jj-1))) <= D)
						scatter!(plots[dd, jj], 1:T,
								observations[:, ((dd+D_aux) + 5*(jj-1))], marker_z = state_seq,
								markersize = 1.5, legend=nothing,
								ylabel = "\$ y_{$((dd+D_aux) + 5*(jj-1))}\$",
								c = colorPalette,
								colorbar = false,
								markerstrokewidth=0,
								xlims = (1, T),
								xticks = [1, T],
								xtickfont = font(4),
								ytickfont = font(4),
								dpi = 1000)
					else
						plot!(plots[dd, jj],
							  legend=false, grid=false,
							  foreground_color_subplot=:white)
					end
				end
			end
			savefig(plots, chop(path_file, tail = 4)*"_$D_aux.png")
			display(plots)



			D_aux += 20

			if (D_aux < D)
				println("press enter to see the next plot ")
				readline()
			end
		end
	end
end

# -
function Simulate_Data(T::Int, parms::Model_Parms; scale::Bool = false, plt::Bool=true)

    D = size(parms.mvNorm.μ, 1)
    K = size(parms.mvNorm.μ, 2)
    state_seq = zeros(Int64, T)
    observations  = zeros(T, D)


    μ = parms.mvNorm.μ
    Σ = parms.mvNorm.Σ


    # - DAR parms
    ϕ = parms.DAR.ϕ
    if (sum(ϕ) != 1) error("sum ϕ must be one!") end
    probs_innovation = parms.DAR.p

    # - n lags and n states
    P = length(ϕ) - 1
    η = get_TransitionsArray(ϕ[2:(P+1)], probs_innovation)

    # init
    state_seq[1:P] .= sample(1:K, ProbabilityWeights(probs_innovation))
    state = state_seq[1:P]
    for ii in 1:P
        observations[ii, :] = rand(MvNormal(parms.mvNorm.μ[:, state[ii]],
                                            parms.mvNorm.Σ[:, :, state[ii]]))
    end

    # 2:N
    for tt in (P+1):T
        transition_prob = [η[CartesianIndex(Tuple(vcat(state_seq[tt-P:(tt-1)], jj)))] for jj in 1:K]
        state_seq[tt] = sample(1:K, ProbabilityWeights(transition_prob))
        observations[tt, :] = rand(MvNormal(μ[:, state_seq[tt]],
                                            Σ[:, :, state_seq[tt]]))
    end



    if scale
		temp = z_score(observations; only_sd = true)
        observations = temp[:data]
		std_out = temp[:std]
    end

    if plt
		Plot_Data(observations, state_seq)
    end

	if scale
		return observations, std_out, state_seq
	else
		return observations, state_seq
	end

  return observations, state_seq
end

# -
function generate_ϕ_ZIGD(P, hyper_parms::Hyper_Parms)

    α_ϕ_AR = hyper_parms.α_ϕ_AR
    α_ϕ_innovations = hyper_parms.α_ϕ_innovations

    # v = ones(Float64, P) # beta rv's
    v = fill(NaN, P)
    z = ones(Bool, P-1) # bernoulli rv's
    ϕ = zeros(Float64, P+1) # innovation + AR probs

    # ϕ_0 = 1 - sum([ϕ_1, …, ϕ_p])
    v[1] = rand(Beta(α_ϕ_innovations[1], α_ϕ_innovations[2]))
    ϕ[1] = v[1]

    # [ϕ_1, …, ϕ_p]
    p = 1
    while true
        ξ = sum(ϕ[1:p])
        z[p] = rand(Bernoulli(ξ))

        v[p+1] = ifelse(z[p], 1, rand(Beta(α_ϕ_AR[1], α_ϕ_AR[2])))
        ϕ[p+1] = v[p+1]*prod(1 .- v[1:p])

        if z[p] break end

        p+=1

        if !z[P-1]
            ϕ[P+1] = 1-sum(ϕ[1:P])
            break
        end

    end
    out = Dict(:ϕ => ϕ, :v => v, :z =>z)

    return out
end

# -
function get_ϕ(v::Vector{Float64}, z::Vector{Bool})
    P = length(z) + 1
    ϕ = zeros(Float64, P+1)

    ϕ[1] = v[1]
    p = 1
    while true
        ϕ[p+1] = v[p+1]*prod(1 .- v[1:p])
        if z[p]
            break
        end
        p+=1
        if p == P
            ϕ[p+1] = 1-sum(ϕ[1:p])
            break
        end
    end
    return ϕ
end

function getCorrelation(Σ::Matrix; plt = false)
    temp = Matrix(Diagonal(1 ./ sqrt.(Σ[diagind(Σ)])))
    corr = temp * Σ * temp
    corr = convert(Matrix{Float64}, corr)
    if plt
        R"""
        par(mfrow = c(1, 1))
        library("corrplot")
        corrplot($corr)
        """
    end
    return corr
end

function randomSparsePrecision(D, sparsity)

	R"""
	library(Matrix)
	library(matrixcalc)

	p <- $(D)
	sparsity <- $(sparsity)
	n_sparse <- ceiling(((p)*((p)-1)/2)*(sparsity))
	continue = TRUE

	while (continue == TRUE) {
		A <- rsparsematrix(nrow = p, ncol = p,
						   nnz = n_sparse,
						   rand.x = stats::rnorm)
		Precision <- A %*% t(A) + 0.05 * diag(rep(1, p))
		Precision <- as.matrix(Precision)
		if (is.positive.definite(Precision)) {
			continue = FALSE
		}
	}
	"""

	Ω = rcopy(R"Precision")
	return Ω
end


# - log likelik v, z, and p
function log_likelik_v_and_z_and_p(γ_seq::Array, v::Array, z::Array{Bool}, p::Array)

    T = length(γ_seq)
    p_star = [p; (1-sum(p))]
    ϕ = get_ϕ(v, z)

    # if (any(ϕ .< 0) || sum(ϕ) != 1)  return -Inf end
    # if (any(p .< 0) || sum(p) > 1)  return -Inf end

    if any(ϕ .== 0)
         P = findall(ϕ .== 0.0)[1] - 2
    else
         P = length(ϕ) - 1
    end
    ϕ_aux = ϕ[2:(P+1)]

    # if (any(ϕ .< 0) || sum(ϕ) > 1)  error("ϕ out of domain") end
    # if (any(p .< 0) || sum(p) > 1)  error("p out of domain") end

    ϕ_mat = Array{Float64}(undef, length((P+1):T), P)
    @inbounds for l in 1:P
        ϕ_mat[:, l] = ϕ_aux[l] * Vector{Int64}(γ_seq[(P+1):T] .==
                              γ_seq[(P - l + 1):(T-l)])
    end
    ϕ_mat = hcat(ϕ[1] * [p_star[jj] for jj in γ_seq[(P+1):T]], ϕ_mat)

    return sum(log.(sum(ϕ_mat, dims = 2)))
end



function log_prior_v_and_z(v::Vector, z::Vector{Bool}, hyper_parms::Hyper_Parms)


    α_ϕ_AR = hyper_parms.α_ϕ_AR
    α_ϕ_innovations = hyper_parms.α_ϕ_innovations
    ϕ = get_ϕ(v, z)

    #if (any(ϕ .< 0) || sum(ϕ) != 1)  return -Inf end
    # if (any(ϕ .< 0) || sum(ϕ) > 1)  return -Inf end

    out = 0.0
    # - v[1]
    out += log(pdf(Beta(α_ϕ_innovations[1], α_ϕ_innovations[2]), v[1]))
    # - z[1] | v[1]
    ξ = v[1]
    out += log(pdf(Bernoulli(ξ), z[1]))

    if !z[1]

        if any(z .== 1)
            P_star = findall(z .== 1)[1]
        else
            P_star = length(z)
        end

        for jj in 2:P_star
            # - v[j] | z[j-1]
            out += (1 - z[jj-1]) * log(pdf(Beta(α_ϕ_AR[1], α_ϕ_AR[2]), v[jj])) # + z[1]*log(1)
            # - z[j] | v[1], …, v[j] and z[j-1]
            # ξ = ifelse(z[jj-1] == 0, sum(ϕ[1:jj]), 1) #
            ξ = sum(ϕ[1:jj]) #
            out += z[jj-1]*log(pdf(Bernoulli(1), z[jj])) + (1-z[jj-1])*log(pdf(Bernoulli(ξ), z[jj]))
        end

        if !any(z .== 1)
            out += (1 - z[P_star]) * log(pdf(Beta(α_ϕ_AR[1], α_ϕ_AR[2]), v[P_star+1])) # + z[1]*log(1)
        end
    end

    return out
end


# - log prior p (remark: p is without 'last' component)
function log_prior_p(p::Array, α_p::Array)
    p_star = [p; (1-sum(p))]
    #if (any(p_star .< 0) || sum(p_star) != 1)  return -Inf end

    return sum(α_p .* log.(p_star))
end

# -
function log_posterior_v_and_z_and_p(γ_seq::Array, v::Array, z::Array{Bool},
                                      p::Array, hyper_parms::Hyper_Parms)

    ϕ = get_ϕ(v, z)
    if (any(ϕ .< 0) || sum(ϕ) != 1)  return -Inf end
    if (any(p .< 0) || sum(p) > 1)  return -Inf end

    log_lik = log_likelik_v_and_z_and_p(γ_seq, v, z, p)
    log_prior_1 = log_prior_v_and_z(v, z, hyper_parms)
    log_prior_2 = log_prior_p(p, hyper_parms.α_p)

    return (log_lik + log_prior_1 + log_prior_2)
end




# -
function get_interval(var_name::String, jj::Int, γ_seq::Array, v::Array,
                      z::Array,
                      p::Array, y::Real, w::Real, max_steps::Int,
                      hyper_parms::Hyper_Parms)

    x = ifelse(var_name == "v", copy(v), copy(p))
    #println(x)
    if (var_name == "v")
        L = v[jj] - rand(Uniform(0, 1))
    else
        L = p[jj] - rand(Uniform(0, 1))
    end
    R = L + w

    # step out
    J = floor(max_steps * rand(Uniform(0, 1)))
    K = (max_steps-1) - J

    if var_name == "v"
        #println(x)
        x[jj] = L
        while (y < log_posterior_v_and_z_and_p(γ_seq, x, z, p, hyper_parms)) && (J>0)
            L = L - w
            J = J - 1
        end
        x[jj] = R
        while (y < log_posterior_v_and_z_and_p(γ_seq, x, z, p, hyper_parms)) && (K>0)
            R = R + w
            K = K - 1
        end
        return Dict(:L => L, :R => R)
    elseif var_name == "p"
        #println(x)
        x[jj] = L
        while (y < log_posterior_v_and_z_and_p(γ_seq, v, z, x, hyper_parms)) && (J>0)
            L = L - w
            J = J - 1
        end
        x[jj] = R
        while (y < log_posterior_v_and_z_and_p(γ_seq, v, z, x, hyper_parms)) && (K>0)
            R = R + w
            K = K - 1
        end
        return Dict(:L => L, :R => R)
    end
end

# -
function shrink_and_sample(var_name::String, jj::Int, γ_seq::Array, v::Array,
                           z::Array,
                           p::Array, y::Real, interval::Dict,
                           hyper_parms::Hyper_Parms)

    x = ifelse(var_name == "v", copy(v), copy(p))
    L = interval[:L]
    R = interval[:R]

    if var_name == "v"
        while true
            #println("ciao")
            x_prop = rand(Uniform(L, R))
            x[jj] = x_prop
            if (y < log_posterior_v_and_z_and_p(γ_seq, x, z, p, hyper_parms))
                return x_prop
            elseif (x_prop > v[jj])
                R = x_prop
            elseif (x_prop < v[jj])
                L = x_prop
            end
        end
    elseif var_name == "p"
        while true
            #println("ciao")
            x_prop = rand(Uniform(L, R))
            x[jj] = x_prop
            if (y < log_posterior_v_and_z_and_p(γ_seq, v, z, x, hyper_parms))
                return x_prop
            elseif (x_prop > p[jj])
                R = x_prop
            elseif (x_prop < p[jj])
                L = x_prop
            end
        end
    end
end


# -
function get_TransitionsArray(ϕ::Array, p::Array)

    P = length(ϕ)
    K = length(p)
    ϕ_aux = 1 - sum(ϕ)
    ϕ_rev = reverse(ϕ)
    η_out = Array{Float64}(undef, Tuple([K; fill(K, P)]))

     @inbounds Threads.@threads for index_temp in CartesianIndices(η_out)
        out = 0.0
        for jj in P:-1:1
            if (index_temp[P+1] == index_temp[jj])
                out += ϕ_rev[jj]
            end
        end
        out += p[index_temp[P+1]]*ϕ_aux
        η_out[index_temp] = out
    end

    return η_out
end


# - K = n_states + 1
function get_StateCartesianIndices(K::Int, P::Int)

    η_temp = Array{Float64}(undef, Tuple([K; fill(K, P)]))
    η_cartesianIndex = CartesianIndices(η_temp)
    state_CartesianIndex = [Vector{CartesianIndex{P+1}}(undef, K^P) for ii = 1:K]

    @inbounds Threads.@threads for state in 1:K
        ii = 1
        for idx in η_cartesianIndex
            if (Tuple(idx)[P] == state)
                state_CartesianIndex[state][ii] = idx
                ii += 1
            end
        end
    end
    return state_CartesianIndex
end


# -
function mvGaussLikelihood(obs::Array, emissions::EmissionParms)

    μ = emissions.mean

    N = size(obs, 1)
    K = size(μ, 2)
    log_likelihood = Array{Float64}(undef, K, N)


    @inbounds for kk in 1:K
        # maybe I can take account for kk = 1 is Spike
        log_likelihood[kk, :] = logpdf(MultivariateNormal(μ[:, kk],
                                       emissions.precision[kk].Σ), obs')
    end

    normalizer = [maximum(@view(log_likelihood[:, n])) for n =1:N]
    log_likelihood = log_likelihood - reshape(repeat(normalizer,
                                        inner = [K, 1]), K, N)
    likelihood = exp.(log_likelihood)

    return likelihood
end

# -
function mvGaussLogLikelihood(obs::Array, emissions::EmissionParms)

    μ = emissions.mean

    N = size(obs, 1)
    K = size(μ, 2)
    log_likelihood = Array{Float64}(undef, K, N)


    @inbounds for kk in 1:K
        # maybe I can take account for kk = 1 is Spike
        log_likelihood[kk, :] = logpdf(MultivariateNormal(μ[:, kk],
                                       emissions.precision[kk].Σ), obs')
    end

    return log_likelihood
end


# -
function backwardMessages(likelihood::Matrix, η::Array,
                          state_CartIdx)

    N = size(likelihood, 2)
    K = size(η, 1)
    P = length(size(η)) - 1

    bwds_msg = ones(Float64, K, N)
    partial_marg = zeros(K, N)

    if (P == 1)

        @inbounds for tt in (N-1):-1:1
            # Mutiplying likelihood by incoming message
            partial_marg[:, tt+1] = @view(likelihood[:, tt+1]) .* @view(bwds_msg[:, tt+1])
            # Integrate out z_t
            bwds_msg[:, tt] = η * @view(partial_marg[:, tt+1])
            bwds_msg[:, tt] = @view(bwds_msg[:, tt])/sum(@view(bwds_msg[:, tt]));
        end
    else
        @inbounds for tt in (N-1):-1:1
            partial_marg[:, tt+1] = @view(likelihood[:, tt+1]) .* @view(bwds_msg[:, tt+1])

            bwds_msg[:, tt] = [sum(η[state_CartIdx[state]] .* repeat(@view(
                                partial_marg[:, tt+1]), inner = K^(P-1)))
                               for state in 1:K]
            bwds_msg[:, tt] = @view(bwds_msg[:, tt])/sum(@view(bwds_msg[:, tt]))
        end
    end

    # compute marginal for first time point
    partial_marg[:, 1] = @view(likelihood[:, 1]) .* @view(bwds_msg[:, 1])

    out = Dict(:partial_marg => partial_marg, :bwds_msg => bwds_msg)
    return out
end

# - !!!! so far works on only for P = 1, 2
function forwardMessages(obs::Array, emissions::EmissionParms, η::Array)

	K̂ = size(η)[1]
	N = size(obs)[1]
	P̂ = length(size(η)) - 1

	log_likelihood = mvGaussLogLikelihood(obs, emissions)
	log_DAR_fwds_msg = zeros(Float64, K̂, K̂, N)
	log_fwds_msg = zeros(Float64, K̂, N)
	fwds_msg = zeros(Float64, K̂, N)

	if P̂ == 1

		#  initial state:
		log_fwds_msg[:, P̂] .= log_likelihood[:, P̂]
		log_fwds_msg[:, P̂] = log_fwds_msg[:, P̂] .- logsumexp(log_fwds_msg[:, P̂])
		fwds_msg[:, P̂] = exp.(log_fwds_msg[:, P̂] .- logsumexp(log_fwds_msg[:, P̂]))


		# recursion
		for tt in (P̂+1):N
			for jj in 1:K̂
				out = 0.0
				for ii in 1:K̂
					out += exp(log(η[ii, jj]) + log_likelihood[jj, tt-1] + log_fwds_msg[ii, tt-1])
				end
				log_fwds_msg[jj, tt] = log(out)
			end
			log_fwds_msg[:, tt] = log_fwds_msg[:, tt] .- logsumexp(log_fwds_msg[:, tt])
			fwds_msg[:, tt] = exp.(log_fwds_msg[:, tt])
		end

	elseif P̂ ==2
		log_DAR_fwds_msg = zeros(Float64, K̂, K̂, N)
		# - initial (log) DAR forward messages
		for tt in 1:P̂
			for kk in 1:K̂
				log_DAR_fwds_msg[kk, :, tt] .= log_likelihood[kk, tt]
			end
			# normalizing, considering all elements
			log_DAR_fwds_vec = reshape(log_DAR_fwds_msg[:, :, tt], K̂*K̂)
			log_DAR_fwds_msg[:, :, tt] = reshape(log_DAR_fwds_vec .- logsumexp(log_DAR_fwds_vec), K̂, K̂)
		end

		# - (log) DAR forward messages recursion
		for tt in (P̂ + 1):N
			for ii in 1:K̂
				for jj in 1:K̂
					out = 0.0
					for ll in 1:K̂
						out += exp(log(η[ll, ii, jj]) + log_likelihood[ii, tt-1] + log_DAR_fwds_msg[jj, ll, tt-1])
					end
					log_DAR_fwds_msg[ii, jj, tt] = log(out)
				end
			end
			log_DAR_fwds_vec = reshape(log_DAR_fwds_msg[:, :, tt], K̂*K̂)
			log_DAR_fwds_msg[:, :, tt] = reshape(log_DAR_fwds_vec .- logsumexp(log_DAR_fwds_vec), K̂, K̂)
		end


		# - (log) forward messages (sum over) + normalization
		for tt in 1:N
			for jj in 1:K̂
				log_fwds_msg[jj, tt] = logsumexp(log_DAR_fwds_msg[jj, :, tt])
			end
			fwds_msg[:, tt] = exp.(log_fwds_msg[:, tt] .- logsumexp(log_fwds_msg[:, tt]))
		end
	elseif P̂ == 3
		log_DAR_fwds_msg = zeros(Float64, K̂, K̂, K̂, N)
	end




	return fwds_msg
end

# -
function getStateProbablities(obs, fit_reshaped)

	P̂ = fit_reshaped[:bayes_est][:P̂]
	K̂ = fit_reshaped[:bayes_est][:K̂]
	ϕ̂ = fit_reshaped[:bayes_est][:ϕ̂]
	p̂ = fit_reshaped[:bayes_est][:p̂]
	N = size(obs, 1)

	emissions_hat = EmissionParms(fit_reshaped[:bayes_est][:μ̂],
	 							 [GHS() for kk in 1:K̂])
	for kk in 1:K̂
		emissions_hat.precision[kk].Σ = fit_reshaped[:bayes_est][:Σ̂][:, :, kk]
	end
	likelihood = mvGaussLikelihood(obs, emissions_hat)

	state_CartIdx = get_StateCartesianIndices(K̂, P̂)
	η = get_TransitionsArray(ϕ̂[2:(P̂ + 1)], p̂)

	fwds_msg = forwardMessages(obs, emissions_hat, η)
	bwds_msg = backwardMessages(likelihood, η, state_CartIdx)[:bwds_msg]

	local_prob = zeros(Float64, K̂, N)
	log_local_aux = zeros(Float64, K̂)

	for tt in 1:N
		for jj in 1:K̂
			log_local_aux[jj] = log(fwds_msg[jj, tt]) + log(bwds_msg[jj, tt])
		end
		local_prob[:, tt] = exp.(log_local_aux .- logsumexp(log_local_aux))
	end

	return local_prob
end

# - (super slow...)
function plotStateProbs(stateProbs; path_file = nothing,
						width = nothing, height = nothing)

	K = size(stateProbs, 1)
	#colorPalette = palette(:lighttest, K)
	colorPalette = palette(:rainbow1, K)
	# candidate: colorPalette = palette(:seaborn_bright6, K)
	#colorPalette = palette(:jet, K)
	colorPalette_hex = ["#"*hex(colorPalette.colors.colors[kk]) for kk in 1:K]
	save = ifelse(!isnothing(path_file), true, false)
	plt_legend = ["state " * string(jj) for jj in 1:K]


	R"""

	library("RColorBrewer")

	save = ($save)
	state_probs = $(stateProbs)
	path_file = $(path_file)
	width = $(width)
	height = $(height)
	plt.legend = $(plt_legend)

	K <- nrow(state_probs)
	N <- ncol(state_probs)
	time <- 1:N

	# z_col <- brewer.pal(n = K, name = "Accent")
	z_col <- $(colorPalette_hex)

	if (save) {
		pdf(file = path_file,
			width = width,
			height = height)
	}

	plot(1:N, type = "n",  ylab = "Prob State",
		
	     cex = 0.5, las = 1,
	     xlim = c(1, N),
	     ylim = c(0,1), cex.lab = 1.3, cex.axis = 1.3, xlab = "Time")

	plot_p <- matrix(NA,nrow=N-1,ncol=2*K)
	a <- 0

	for (n in 2:N) {

	  for (j in 1:K) {
	    plot_p[n-1, (j*2-1)] <- state_probs[j, n-1]
	    plot_p[n-1, j*2] <- state_probs[j, n]
	    # if	(j==1){col_states<- z_col[1]}
	    # if	(j==2){col_states<- z_col[2]}
	    # if	(j==3){col_states<- z_col[3]}
	    # if	(j==4){col_states<- z_col[4]}
		# if	(j==4){col_states<- z_col[4]}
		# if	(j==4){col_states<- z_col[4]}
		col_states <- z_col[j]

	    # (at some point need to make function to
	    # generalize for all K, like this is redundant)
	    if	(j==1){
	      point_1<-a
	      point_2<-point_1+plot_p[n-1,(j*2-1)]
	      point_4<-a
	      point_3<-point_4+plot_p[n-1,(j*2)]	}

	    if	(j==2){
	      point_1<-a+plot_p[n-1,(j-1)*2-1]
	      point_2<-point_1+plot_p[n-1,(j*2-1)]
	      point_4<-a+plot_p[n-1,(j-1)*2]
	      point_3<-point_4+plot_p[n-1,(j*2)]	}

	    if	(j==3){
	      point_1<-a+plot_p[n-1,(j-2)*2-1]+plot_p[n-1,(j-1)*2-1]
	      point_2<-point_1+plot_p[n-1,(j*2-1)]
	      point_4<-a+plot_p[n-1,(j-2)*2]+plot_p[n-1,(j-1)*2]
	      point_3<-point_4+plot_p[n-1,(j*2)]}
	    if (j==4) {
	      point_1 <- a+ plot_p[n-1,(j-3)*2-1] + plot_p[n-1,(j-2)*2-1]+plot_p[n-1,(j-1)*2-1]
	      point_2 <- point_1+plot_p[n-1,(j*2-1)]
	      point_4 <- a + plot_p[n-1,(j-3)*2] + plot_p[n-1,(j-2)*2]+plot_p[n-1,(j-1)*2]
	      point_3 <- point_4+plot_p[n-1,(j*2)]}
	    if (j ==5) {
	      point_1 <- a+ plot_p[n-1,(j-4)*2-1] + plot_p[n-1,(j-3)*2-1]+plot_p[n-1,(j-2)*2-1] +plot_p[n-1,(j-1)*2-1]
	      point_2 <- point_1+plot_p[n-1,(j*2-1)]
	      point_4 <- a + plot_p[n-1,(j-4)*2] + plot_p[n-1,(j-3)*2]+plot_p[n-1,(j-2)*2] + +plot_p[n-1,(j-1)*2]
	      point_3 <- point_4+plot_p[n-1,(j*2)]}
		if (j ==6) {
		  point_1 <- a+ plot_p[n-1,(j-5)*2-1] + plot_p[n-1,(j-4)*2-1] + plot_p[n-1,(j-3)*2-1]+plot_p[n-1,(j-2)*2-1] +plot_p[n-1,(j-1)*2-1]
		  point_2 <- point_1+plot_p[n-1,(j*2-1)]
		  point_4 <- a + plot_p[n-1,(j-5)*2] + plot_p[n-1,(j-4)*2] + plot_p[n-1,(j-3)*2]+plot_p[n-1,(j-2)*2] + +plot_p[n-1,(j-1)*2]
		  point_3 <- point_4+plot_p[n-1,(j*2)]}


	    polygon(c(time[n-1],time[n-1],time[n],time[n]),
	            c(point_1,point_2,point_3,point_4),col=scales::alpha(col_states, 0.8), border=NA)
	    lines(c(time[n-1],time[n]),c(point_2,point_3), col=scales::alpha(col_states, 0.5))
	  }
	}

	legend("topleft",
			legend=plt.legend,
			col = z_col,
			fill = z_col,
			bty = "n",
			cex = 2.5)

	if (save) {
		dev.off()
	}
	"""
end



# - ?
function NormalInversWishart_rng(μ::Vector, κ::Real, ν::Real, S::Matrix)

	Σ_draw = rand(InverseWishart(ν, S))
	μ_draw = rand(MultivariateNormal(μ, Σ_draw/κ))
	D = size(μ_draw, 1)

	output = Dict(:μ => μ_draw, :Σ => Σ_draw)
	return output
end

# -
function Effective_Sample_Size(out_MCMC::Array)

    temp = ifelse(size(out_MCMC, 2) == 1, out_MCMC, out_MCMC')
    R"""
    library(coda)
    ESS = effectiveSize(as.mcmc($temp))
    """
    ESS = Int64.(floor.(rcopy(R"ESS")))

    return     ESS./size(temp, 1)
end

# -
function Plot_Multiple_Simulations(T::Int, parms::Model_Parms; n_simul::Int = 10)

	  n_states = length(parms.DAR_parms.p)
      P = length(parms.DAR_parms.ϕ)
      n_col_plots = Int64(ceil(n_simul/5))
      plots = plot(layout = (5,  n_col_plots), top_margin = 0.1cm)
      #colorPalette = palette(:lightrainbow, n_states + 1)
      # colorPalette = palette(:tab10, n_states + 1)
      #colorPalette = palette(:matter, n_states + 1)
      colorPalette = palette(:roma, n_states)

	  μ = parms.mvNorm.μ
	  σ = parms.mvNorm.σ
	  ϕ = parms.DAR_parms.ϕ
	  p = parms.DAR_parms.p

	  emissions = mvNormalParms(μ, σ)
	  η = get_TransitionsArray(ϕ, p)
      state_CartIdx = get_StateCartesianIndices(n_states, P)

      for jj in 1:n_col_plots
          for dd in 1:5
              observations, state_seq = Simulate_Data(T, parms_true; plt = false)
			  γ_est = sampleStates(observations, emissions, η, state_CartIdx)[:stateSeq]
			  change_points = vcat(0, findall(diff(γ_est) .!= 0), T)

              if (((dd + 5*(jj-1))) <= n_simul)
                  plot!(plots[dd, jj],
                        observations,
                        color = :grey,
                        legend = :topright,
                        label = "",
                        alpha = 0.3,
                        xlims = (1, T),
                        xticks = [1, T],
                        xtickfont = font(4),
                        ytickfont = font(4),
                        yguidefontsize = 6,
                        dpi = 300,
                  )
                  scatter!(plots[dd, jj], 1:T,
                          observations, marker_z = γ_est,
                          markersize = 1, legend=nothing,
                          ylabel = "\$ \\beta_t \$",
                          c = colorPalette,
                          colorbar = false,
                          markerstrokewidth=0,
                          xlims = (1, T),
                          xticks = [1, T],
                          xtickfont = font(4),
                          ytickfont = font(4),
                          dpi = 300)
                  for ii = 1:(length(change_points)-1)
                        z_span = [change_points[ii], change_points[ii + 1]]
                        vspan!(
                              plots[dd, jj],
                              z_span,
                              color = colorPalette.colors[γ_est[change_points[ii+1]]],
                              alpha = 0.2,
                              labels = :none,
                              dpi = 300
                        )
                  end
              else
                  plot!(plots[dd, jj],
                        legend=false, grid=false,
                        foreground_color_subplot=:white)
              end
          end
      end

      plot!(plots[1, 1], top_margin = 0.9cm)
      annotate!(plots[1, 1], floor(T) + 10, 17,
               text("\$ P = $P, \\quad \\phi = $(parms.DAR_parms.ϕ) \$",
               :black, :center, 10, :bold))
      path_fig = "plots/simul_P$P"* "_phi$(parms.DAR_parms.ϕ)" * ".png"
      savefig(plots, path_fig)
      Plots.display(plots)
end

function RandomCovarianceMatrix(D)

	A = randn(D,D)
	A = A'*A
	A = (A + A')/2
	scaling = sort(A[diagind(A)], rev = true)[3]
	Σ = A/scaling

	return Σ

end


# -
function get_IndexesAll(D)
    ind_all = zeros(Int64, D-1, D)
    for i in 1:D
        if i == 1
            ind = collect(2:D)
        elseif i == D
            ind = collect(1:(D-1))
        else
            ind = [1:i-1; i+1:D]
        end
        ind_all[:, i] = ind
    end
    ind_all
end


# - dim(out) = (m*p + k) * (m*p + k)
function drawSparseCovariance(m::Int, p::Int, k::Int)

    R"""
    library(Matrix)

    m <- $m
    p <- $p
    k <- $k

    ## build sample sparse covariance matrix
    Q1 <- tril(kronecker(Matrix(rnorm(p*p),p,p),diag(m)))
    Q2 <- cbind(Q1,Matrix(0,m*p,k))
    Q3 <- rbind(Q2,cbind(Matrix(rnorm(k*m*p),k,m*p),Diagonal(k)))
    V <- tcrossprod(Q3)
    """
    return(rcopy(R"as.matrix(V)"))
end

# -
function z_score(data::Array; only_sd::Bool = true)
	N = size(data, 1)
	D = size(data, 2)
	out = Array{Float64}(undef, N, D)
	std_out = Array{Float64}(undef, D)

    if only_sd
        for dd in 1:D
            out[:, dd] = (data[:, dd]) ./ std(data[:, dd])
			std_out[dd] = std(data[:, dd])
        end
    else
        for dd in 1:D
            out[:, dd] = (data[:, dd] .- mean(data[:, dd])) ./ std(data[:, dd])
			std_out[dd] = std(data[:, dd])
        end
    end

	return Dict(:data => out, :std => std_out)
end

# -
function numberActiveComponents(z_curr)
    P_max = length(z_curr) + 1
    if (length(findall(z_curr .!= 0)) != 0)
        P_curr = findall(z_curr .!= 0)[1]
    else
        P_curr = P_max
    end
    return P_curr
end


# -
function plotHeat_single(Omega; Xlab = "", Ylab = "", Main_lab = "", limit = [-2, 2])

	R"""
	Omega = $(Omega)
	Xlab = $(Xlab)
	Ylab = $(Ylab)
	Main = $(Main_lab)
	limit = $(limit)

	plot.heat(Omega,Xlab,Ylab,Main,limit)

	"""
end

R"""

library(ggplot2)

plot.heat <- function(Omega,Xlab,Ylab,Main,limit, show_legend = T){
  Omega = as.matrix(Omega)
  colnames(Omega)<-NULL
  rownames(Omega)<-NULL
  x = reshape2::melt(data.frame(Omega,ind=c(nrow(Omega):1)),id="ind")
  colnames(x)<-c("X1","X2","value")
  p_X_heat = ggplot(data = x, aes(x=X2, y=X1, fill=value)) +
	theme_bw() +
	xlab(Xlab) +
	ylab(Ylab) +
	ggtitle(Main) +
	theme(axis.title=element_text(size=14,face="bold"),
		  axis.text.x=element_blank(),
		  axis.ticks.x=element_blank(),
		  axis.text.y=element_blank(),
		  axis.ticks.y=element_blank())+
	scale_fill_gradient2(limits=limit) + 
	if (show_legend) {
		geom_tile(show.legend = T) 
	} 
	else {
		geom_tile(show.legend = F) 
	}
	# theme(legend.position="bottom")
  return(p_X_heat)
}
"""


# - 
function plotHeat_multiple(Omega_all; path_file = nothing, width = nothing, height = nothing, by_row = false, 
						   show_legend = true)

	save = ifelse(!isnothing(path_file), true, false)
	limit = [floor(minimum(Omega_all)),
			 ceil(maximum(Omega_all))]


	R"""
	library(gridExtra)

	save = ($save)
	path_file = ($path_file)
	Omega_all = ($Omega_all)
	limit <- ($limit)
	width = ($width)
	height = ($height)
	show_legend = ($show_legend)

	if (save) {
		pdf(file = path_file,
		    width = width,
			height = height)
	}


	K = dim(Omega_all)[3]
	plots_list <- list()

	for (kk in 1:K) {
		plots_list[[kk]] = plot.heat(Omega_all[, , kk], Xlab = "", Ylab = "",
								Main = "", limit = limit, show_legend = show_legend)
	}

	if ($by_row) {
		do.call(grid.arrange, c(plots_list, nrow= K))
	} else {
		do.call(grid.arrange, c(plots_list, ncol= K))
	}


	if (save) {
		dev.off()
	}
	"""
end

# -
function getPartialCorrelation(Omega)
	R"""
	conc2pcor <- function(Omega) {
		ans <- -cov2cor(Omega)
		diag(ans) <- 1
		return(ans)
	}

	Omega = ($Omega)
	out = conc2pcor(Omega)
	"""
	return rcopy(R"as.matrix(out)")
end

# -
function getPrecision_Star(D)

	Omega = zeros(Float64, D, D)
	for i in 1:D
		for j in 1:D
			if (i == 1 || j == 1)
				Omega[i, j] = -1/sqrt(D)
			end
		end
	end

	Omega[diagind(Omega)] .= 1
	Omega_partial = getPartialCorrelation(Omega)

	return (Dict(:Ω => Omega, :Ω_part => Omega_partial))
end

# -
function getPrecision_Hub(D, n_blocks)

	if ((D % n_blocks) != 0) stop("D mod n_group has to be equal to zero ") end
	n_group = Int64(D/n_blocks)

	Omega = zeros(Float64, D, D)
	idxs = [Dict( :elements => ((i-1)*n_group+1):(i*n_group)) for i in 1:n_blocks]

	for ll in 1:n_blocks
		idx_group = collect(idxs[ll][:elements])
		aux = idx_group[1]

		for i in idx_group
			for j in idx_group
				if (i == aux || j == aux)
					Omega[i, j] = -2/sqrt(D)
				end
			end
		end
	end

	Omega[diagind(Omega)] .= 1
	Omega_partial = getPartialCorrelation(Omega)

	return (Dict(:Ω => Omega, :Ω_part => Omega_partial))
end

# -
function getPrecision_AR2(D)
	Omega = zeros(Float64, D, D)
	for ii in 1:D
		for jj in 1:D
			if (jj == (ii-1) || jj == (ii + 1))
				Omega[ii, jj] = 0.5
			elseif (jj == (ii-2) || jj == (ii + 2))
				Omega[ii, jj] = 0.25
			end
		end
	end

	Omega[diagind(Omega)] .= 1
	Omega_partial = getPartialCorrelation(Omega)

	return (Dict(:Ω => Omega, :Ω_part => Omega_partial))
end

# -
function UniformOffdiagonalGenerate()
	U = rand()
	if U <= 0.5
		out = rand(Uniform(-1, -0.4))
	else
		out = rand(Uniform(0.4, 1))
	end
	return out
end

# -
function getPrecision_Random(D)

	n_active =  Int64(floor((3/2)*D))

	while true

		Omega = zeros(Float64, D, D)

		R"""
		idxs_upper_tri <- which(upper.tri($Omega, diag = FALSE), arr.ind=T)
		"""
		idxs_upper_tri = rcopy(R"idxs_upper_tri")
		n_elements = size(idxs_upper_tri, 1)
		idxs_active = sample(1:n_elements, n_active, replace = false)

		for ii in idxs_active
			temp = idxs_upper_tri[ii, :]
			Omega[temp[1], temp[2]] = UniformOffdiagonalGenerate()
		end

		R"""
		library(gdata)
		Omega = $Omega
		lowerTriangle(Omega) = upperTriangle(Omega, byrow=TRUE)
		"""

		Omega = rcopy(R"Omega")
		Omega[diagind(Omega)] .= 1

		for jj in 1:D
			aux = deleteat!(collect(1:D), jj)
			column_sum = sum(abs.(Omega[aux, jj]));
			for ii in aux
				Omega[ii, jj] = Omega[ii, jj]/(1.1*column_sum)
			end
		end

		Omega = 0.5*(Omega + Omega')

		if isposdef(Omega)
			Omega_partial = getPartialCorrelation(Omega)
			return (Dict(:Ω => Omega, :Ω_part => Omega_partial))
		end
	end
end

function vec_to_symmetric(x::Array, D::Int64)
	R"""
	D <- $(D)
	x <- $(x)
	mat <- matrix(0, nrow = D, ncol = D)
	mat[lower.tri(mat, diag = TRUE)] <- x
	out <- Matrix::forceSymmetric(mat, uplo="L")
	"""
	out = rcopy(R"as.matrix(out)")
end


function relabel_parameters(fit_extract; stateSeq_true = nothing)

	p_sample = fit_extract[:posterior_modal][:p]
	μ_sample = fit_extract[:posterior_modal][:μ]
	Σ_sample = fit_extract[:posterior_modal][:Σ]
	Ω_sample = fit_extract[:posterior_modal][:Ω]
	κ²_sample = fit_extract[:posterior_modal][:Ω]
	Ω_part_sample = fit_extract[:posterior_modal][:Ω_part]
	classProb_sample = fit_extract[:posterior_modal][:classProb]
	stateSeq_sample = fit_extract[:posterior_modal][:stateSeq]

	n_sample = size(μ_sample, 3)
	K_hat = size(μ_sample, 2)
	D = size(μ_sample, 1)
	T = size(classProb_sample, 1)

	# μ_sample_perm = zeros(Float64, D, K_hat)
	# Σ_sample_perm = zeros(Float64, D, D, K_hat)
	# Ω_sample_perm = zeros(Float64, D, D, K_hat)

	μ_temp = zeros(Float64, n_sample, K_hat, D)
	Σ_temp = zeros(Float64, n_sample, K_hat, Int64(D * (D + 1) / 2))
	Ω_temp = zeros(Float64, n_sample, K_hat, Int64(D * (D + 1) / 2))
	κ²_temp = zeros(Float64, n_sample, K_hat, Int64(D * (D + 1) / 2))
	Ω_part_temp = zeros(Float64, n_sample, K_hat, Int64(D * (D + 1) / 2))
	classProb_temp = zeros(Float64, n_sample, T, K_hat)

	# - reshaping μ
	for kk in 1:K_hat
		μ_temp[:, kk, :] = μ_sample[:, kk, :]'
	end

	# - reshaping Ω_part
	for kk in 1:K_hat
		for tt in 1:n_sample
			Σ_aux = Σ_sample[:, :, kk, tt]
			Ω_aux = Ω_sample[:, :, kk, tt]
			Ω_part_aux = Ω_part_sample[:, :, kk, tt]
			κ²_aux = κ²_sample[:, :, kk, tt]
			Σ_temp[tt, kk, :] = Σ_aux[tril!(trues(size(Σ_aux)))]
			Ω_temp[tt, kk, :] = Ω_aux[tril!(trues(size(Ω_aux)))]
			Ω_part_temp[tt, kk, :] = Ω_part_aux[tril!(trues(size(Ω_part_aux)))]
			κ²_temp[tt, kk, :] = κ²_aux[tril!(trues(size(κ²_aux)))]
		end
	end

	# - reshaping

	for kk in 1:K_hat
		classProb_temp[:, :, kk] = classProb_sample[:, kk, :]'
	end

	# reshaping stateSeq
	stateSeq_temp  = stateSeq_sample'

	# binding emissions
	emissions_temp = cat(μ_temp, Σ_temp, Ω_part_temp,  dims = 3)

	if (stateSeq_true != nothing)
		R"""
		library("label.switching")
		K_hat = ($K_hat)
		p = ($classProb_temp)
		mcmc = ($emissions_temp)
		z = ($stateSeq_temp)
		groundTruth = ($stateSeq_true)


		ls <- label.switching(method = "ECR-ITERATIVE-2",
							  K = K_hat,
							  p = p,
							  mcmc = mcmc,
							  z = z,
							  groundTruth = groundTruth)
		permutations = ls$permutations$`ECR-ITERATIVE-2`
		emissions_mat_permuted <- permute.mcmc(($emissions_temp),
								  ls$permutations$`ECR-ITERATIVE-2`)$output
		similarity = ls$similarity
		"""
	else
		R"""
		library("label.switching")
		K_hat = ($K_hat)
		p = ($classProb_temp)
		mcmc = ($emissions_temp)
		z = ($stateSeq_temp)

		ls <- label.switching(method = "ECR-ITERATIVE-2",
							  K = K_hat,
							  p = p,
							  mcmc = mcmc,
							  z = z)
		permutations = ls$permutations$`ECR-ITERATIVE-2`
		emissions_mat_permuted <- permute.mcmc(($emissions_temp),
								  ls$permutations$`ECR-ITERATIVE-2`)$output
		similarity = ls$similarity
		"""
	end

	permutations = rcopy(R"permutations")
	similarityMatrix = rcopy(R"similarity")
	emissionsPermuted = rcopy(R"emissions_mat_permuted")

	classProb_sample_perm = zeros(Float64, T, K_hat, n_sample)
	μ_sample_perm = zeros(Float64, D, K_hat, n_sample)
	Σ_sample_perm = zeros(Float64, D, D, K_hat, n_sample)
	Ω_sample_perm = zeros(Float64, D, D, K_hat, n_sample)
	κ²_sample_perm = zeros(Float64, D, D, K_hat, n_sample)
	Ω_part_sample_perm = zeros(Float64, D, D, K_hat, n_sample)
	p_sample_perm = zeros(Float64, K_hat, n_sample)

	# - reshaping (permuted) classProbs + innovations
	for tt in 1:n_sample
		κ²_sample_perm[:, :, :, tt] = κ²_sample[:, :, permutations[tt, :], tt]

		Ω_sample_perm[:, :, :, tt] = Ω_sample[:, :, permutations[tt, :], tt]

		classProb_sample_perm[:, :, tt] = classProb_temp[tt, :, permutations[tt, :]]

		p_sample_perm[:, tt] = p_sample[permutations[tt, :], tt]
		p_sample_perm[:, tt] = p_sample_perm[:, tt] ./ sum(p_sample_perm[:, tt])
	end
	classProb_hat_perm = reshape(mean(classProb_sample, dims = 3), T, K_hat)
	p_hat_perm = reshape(mean(p_sample_perm, dims = 2), K_hat)


	# - reshaping (permuted) emissions (μ, Σ, and Ω_part)

	D_mat = Int64(D*(D+1)/2)
	for tt in 1:n_sample
		for kk in 1:K_hat
			μ_sample_perm[:, kk, tt] = emissionsPermuted[tt, kk, 1:D]
			Σ_sample_perm[:, :, kk, tt] = vec_to_symmetric(emissionsPermuted[tt, kk,
											(D+1):(D+1 + D_mat - 1)], D)
			Ω_part_sample_perm[:, :, kk, tt] = vec_to_symmetric(emissionsPermuted[tt, kk,
											(D+1 + D_mat):(D+1 + 2*D_mat - 1)], D)
		end
	end



	# - bayes estimates emissions (permuted)

	#  μ
	μ̂_perm = reshape(mean(μ_sample_perm, dims = 3), D, K_hat)
	# Σ & Ω_part
	Σ̂_perm = reshape(mean(Σ_sample_perm, dims = 4), D, D, K_hat)
	Ω̂_perm = reshape(mean(Ω_sample_perm, dims = 4), D, D, K_hat)
	κ²_hat_perm = reshape(mean(κ²_sample_perm, dims = 4), D, D, K_hat)
	Ω̂_part_perm = reshape(mean(Ω_part_sample_perm, dims = 4), D, D, K_hat)

	emissions_hat_perm = Dict(:μ̂ => μ̂_perm, :Σ̂ => Σ̂_perm, :Ω̂ => Ω̂_perm,
							   :Ω̂_part => Ω̂_part_perm,
							   :κ²_hat => κ²_hat_perm)
	emissions_sample_perm = Dict(:μ => μ_sample_perm, :Σ => Σ_sample_perm,
								 :Ω => Ω_sample_perm,
								 :Ω_part => Ω_part_sample_perm,
								 :κ² => κ²_sample_perm)


	sample_perm = Dict(:emissions => emissions_sample_perm,
					   :p => p_sample_perm,
					   :classProb => classProb_sample_perm)
	estimates_perm = Dict(:emissions_hat => emissions_hat_perm,
						  :classProb_hat => classProb_hat_perm,
						  :p_hat => p_hat_perm)

	out = Dict(:sample => sample_perm,
			   :estimates => estimates_perm,
			   :stateSeq_sample => stateSeq_temp',
			   :similarity_mat => similarityMatrix)

    return out
end


function histogram_DAR(ϕ_sample; ϕ_true = ϕ_true, path_file = nothing, xlabel = "Sample")

	P_hat = size(ϕ_sample, 1)
	line_true = ifelse(!isnothing(ϕ_true), true, false)
	save_plt = ifelse(!isnothing(path_file), true, false)

	alpha_col = 0.9

	#colorPalette = palette(:roma, P_hat)
	# colorPalette = cgrad(:matter, P_hat, categorical = true)
	colorPalette = cgrad(:lighttest, P_hat, categorical=true)


	plt = histogram(ϕ_sample[1, :], bins = 20,
	                normalize = true, label = false,
	                dpi = 800,
	                xlabel = xlabel,
	                ylabel = "Density",
	                xtickfont = font(10),
	                xguidefontsize=18,
	                alpha = alpha_col,
	                c = colorPalette[1],
					bottom_margin=4.5mm)
	if line_true
	    vline!([ϕ_true[1]], color = :black, lw = 2, labels = :none,
	            line = :dash)
	end

	for jj in 2:(P_hat)
	    histogram!(ϕ_sample[jj, :], bins = 20, normalize = true,
	     label = false, c = colorPalette[jj], alpha = alpha_col)
	    if line_true
	        vline!([ϕ_true[jj]], color = :black, lw = 2, labels = :none,
	                dpi = 800, line = :dash)
	    end
	end
	if save_plt
	    savefig(plt, path_file)
	end
	display(plt)
end


function multiHistogram(X_sample, x_true; path_file =nothing, xlabel="", xlimit = [0, 1])

    M = size(X_sample, 1)
    line_true = ifelse(!isnothing(x_true), true, false)
    save_plt = ifelse(!isnothing(path_file), true, false)

    colorPalette = cgrad(:lighttest, M, categorical=true)

    plts = histogram(X_sample[1, :], bins = 20,
                normalize = true, label = false,
                dpi = 800,
                xlabel = xlabel,
                ylabel = "Density",
                xtickfont = font(10),
                xguidefontsize=18,
                alpha = 0.5,
                c = colorPalette[1], 
				xlims = xlimit)
    if line_true
        vline!([x_true[1]], color = :black, lw = 2, labels = :none,
                line = :dash)
    end
    for jj in 2:M
        histogram!(X_sample[jj, :], bins = 20, normalize = true,
         label = false, c = colorPalette[jj], alpha = 0.5)
        if line_true
            vline!([x_true[jj]], color = :black, 
                    lw = 2, labels = :none,
                    dpi = 800, line = :dash)
        end
    end
    if save_plt
        savefig(plts, path_file)
    end
    display(plts)
end 	

function traces_μ(μ_sample, D_max; path_file = nothing)

	save_plt = ifelse(!isnothing(path_file), true, false)

	D = size(μ_sample, 1)
	K_hat = size(μ_sample, 2)
	n_sample =  size(μ_sample, 3)
	plots = plot(layout = (D_max, K_hat), dpi = 300)

	for kk in 1:K_hat
		for dd in 1:D_max
			plot!(plots[dd, kk], μ_sample[dd, kk, :], color = :blue,
				  legend=nothing, label = "\$\\mu_{$kk,$dd}\$",
				  legendfontsize= 2,
				  titlefontsize = 7,
				  title = "\$\\mu_{$kk,$dd}\$",
				  yguidefontsize = 6,
				  xtickfont = font(4),
				  ytickfont = font(4),
				  xlims = (1, n_sample),
				  xticks = [1, n_sample],
				  dpi = 300)
		end
	end
	display(plots)
	if save_plt
		savefig(plots, path_file * "traces_mean.png")
	end
end

# -
function RMSE(obs, signal_hat)
	N = size(obs, 1)
	D = size(obs, 2)

	temp = 0.0
	for dd in 1:D
		temp += sum((signal_hat[:, dd] .- obs[:, dd]).^2)
	end
	temp = temp / N
	return temp
end


# -
function get_active_coeff(Ω_all; α_start = 0.025, α_end = 0.975, true_values=false)

	D = size(Ω_all, 1)
	K_hat = size(Ω_all, 3)
	discovery_mat_Ω = zeros(Int64, D, D, K_hat)

	if true_values
		for kk in 1:K_hat
			for ii in 1:D
				for ll in 1:D
					if Ω_all[ii, ll, kk] == 0
						discovery_mat_Ω[ii, ll, kk] = 0
					else
						discovery_mat_Ω[ii, ll, kk] = 1
					end
				end
			end
		end

	else
		for kk in 1:K_hat
			for ii in 1:D
				for ll in 1:D
					interval = quantile(Ω_all[ii, ll, kk, :], [α_start, α_end])
					if (interval[1] <= 0 <= interval[2])
						discovery_mat_Ω[ii, ll, kk] = 0
					else
						discovery_mat_Ω[ii, ll, kk] = 1
					end
				end
			end
		end
	end

	discovery_vec = zeros(Int64, Int64(D*(D-1)/2), K_hat)
	for kk in 1:K_hat
		temp = discovery_mat_Ω[:, :, kk]
		discovery_vec[:, kk] = temp[tril!(trues(size(temp)), -1)]
	end



	return Dict(:vec => discovery_vec, :mat => discovery_mat_Ω)
end

# - 
function ClassificationMeasure(predictions::Vector, true_values::Vector)
    if (all(predictions .== 0) & all(true_values .== 0))
        acc = 1.0
        spec = 1.0
        MCC = NaN
        F1 = 0.0
        sens = 0.0
        out = Dict(:acc => acc,
                   :spec => spec, 
                   :MCC => MCC,
                   :F1 => F1, 
                   :sens => sens)
        return out
    else
        R"""
    
        library("caret")
        predictions <- $(predictions)
        true_values <- $(true_values)
    
        if (all(true_values==0)) {
            true_values = factor(true_values)
            levels(true_values) = c(levels(true_values), "1")
        } else {
        true_values = factor(true_values)
        }

		

        if (all(predictions==0)) {
			predictions = factor(predictions)
			levels(predictions) = c(levels(predictions), "1")
		} else {
			predictions = factor(predictions)
		}
		
    
        conf_mat <- confusionMatrix(data = predictions,
                                    reference = true_values, 
                                    positive = "1")
        """
    
        conf_mat = rcopy(R"conf_mat")[:table]
        TN = conf_mat[1, 1]
        TP = conf_mat[2, 2]
        FP = conf_mat[2, 1]
        FN = conf_mat[1, 2]
    
        acc = (TP + TN)/ (TP + TN + FN + FP)
        spec = TN/(TN+FP)
        MCC = (TP*TN - FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
        F1 = TP/(TP + 0.5*(FP + FN))
        sens = TP/(TP+FN) # (also called recall)
    
        out = Dict(:acc => acc, 
                   :spec => spec, 
                   :MCC => MCC,
                   :F1 => F1, 
                   :sens => sens)
        return out
    end  
end 




# -
function square_loss(x, x_hat)
	sum((x .- x_hat).^2)
end



# - 
function RetrieveTrueLabel_Mat(X_est::Array, X_true::Array)
    
    K = size(X_true, 3)
    loss_mat = zeros(Float64, K, K)

    R"""
    loss_mat = matrix(NA, $K, $K)
    for (kk in 1:$K) {
        for (jj in 1:$K) {
            loss_mat[kk, jj] = norm($X_true[ , , kk] - $X_est[, , jj], type = c("F"))
        }
    }
    """
    loss_mat = rcopy(R"loss_mat")

    idx_out = zeros(Int64, K)

    kk = 1;
    loss_mat[kk, :]
    candidate = argmin(loss_mat[kk, :])
    idx_out[kk] = candidate

    for kk in 2:K 
        loss_mat_temp = loss_mat[kk, :]
        loss_mat_temp[idx_out[1:(kk-1)]] .= Inf
        candidate = argmin(loss_mat_temp)
        idx_out[kk] = candidate
    end 

    return idx_out
end 


# -
function get_Ω_vec(Ω)
	K_hat = size(Ω, 3)
	D = size(Ω, 2)
	Ω_vec = zeros(Float64, Int64(D*(D-1)/2), K_hat)

	for kk in 1:K_hat
		temp = Ω[:, :, kk]
		Ω_vec[:, kk] = temp[tril!(trues(size(temp)), -1)]
	end
	return Ω_vec
end



function ClassificationMeasure_AllStates(discovery_vec_est::Array, discovery_vec_true::Array)

    K_hat = size(discovery_vec_est, 2)
    col_names = [:acc, :spec, :MCC, :F1, :sens]
    col_types = [Float64, Float64, Float64, Float64, Float64]
    named_tuple = (; zip(col_names, type[] for type in col_types )...)
    measure_df = DataFrame(named_tuple) # 0×2 DataFrame

    for jj in 1:K_hat 
        measure_dict = ClassificationMeasure(discovery_vec_est[:, jj], 
                                             discovery_vec_true[:, jj])
        push!(measure_df, [x for x  in values(measure_dict)])
    end 

    return measure_df
end 


# - 
function RMSE_AllStates(Ω_part_vec_est, Ω_part_vec_true)

    K̂ = size(Ω_part_vec_est, 2)
    D_star = size(Ω_part_vec_est, 1)
    RMSE_all = zeros(Float64, K̂)

    for kk in 1:K̂ 
        RMSE_all[kk] = sqrt(sum((Ω_part_vec_true[:, kk] .- Ω_part_vec_est[:, kk]).^2)/D_star)
    end 

    return RMSE_all
end

# - mvnormal MLE
function mvNormalMLE(obs)
    D = size(obs, 2);
    N = size(obs, 1);
    μ_hat = vec(mean(obs, dims = 1))
    Σ_hat = zeros(D, D)
    for tt in 1:N
        Σ_hat = Σ_hat .+ (obs[tt, :] - μ_hat) * (obs[tt, :] - μ_hat)';
    end 
    Σ_hat /= N;

    out = Dict(:μ => μ_hat, :Σ => Σ_hat);
    return(out);
end




function glassoSlidingWindow(obs::Array; win_slide::Int, ρ::Real)

    N = size(obs, 1);
	N_slide = N-win_slide+1;
    D = size(obs, 2);

    μ_slide = zeros(D, N_slide);
    Σ_slide = zeros(D, D, N_slide);
	Ω_slide = zeros(D, D, N_slide);
	rho = copy(ρ);

    # - tapered window approach
    for tt in 1:N_slide
        obs_window = obs[(tt:(tt+win_slide-1)), :];
        parms_hat = mvNormalMLE(obs_window)
        μ_slide[:, tt] = parms_hat[:μ];
        Σ_slide[:, :, tt] = parms_hat[:Σ];
    end 

	# - glasso covariance (for each t)
	R"""
	library("glasso")
	"""
	for tt in 1:N_slide
		cov_slide = Σ_slide[:, :, tt];
		R"""
		prec_slide = glasso($cov_slide, rho= $rho)$wi
		"""
		Ω_slide[:, :, tt] = rcopy(R"prec_slide");
	end 

	# - partial correlation
	Ω_part_slide = zeros(D, D, N_slide);
	for tt in 1:N_slide 
    	Ω_part_slide[:, :, tt] = getPartialCorrelation(Ω_slide[:, :, tt] )
	end 


    out = Dict(:μ => μ_slide, :Σ => Σ_slide, :Ω => Ω_slide, 
			   :Ω_part => Ω_part_slide);
    return(out);
end 

# - 
function glasso(cov_mat::Array; ρ::Real)
	rho = copy(ρ);
	R"""
	out = glasso($cov_mat, rho= $rho)$wi
	"""
	return(rcopy(R"out"));
end 

# - 
function kmeanSlidingWindow(obs::Array, K::Int,  win_slide::Int, ρ::Real)

    N = size(obs, 1);
	N_slide = N-win_slide+1;
    D = size(obs, 2);
	D_star = Int64(D*(D-1)/2);

    slide_output = glassoSlidingWindow(obs, win_slide = win_slide, ρ = ρ);
    μ_slide = slide_output[:μ];
    Σ_slide = slide_output[:Σ];
    Ω_slide = slide_output[:Ω];
	Ω_part_slide = slide_output[:Ω_part];

    D_star = Int64(D*(D-1)/2);
    feature_slide_vec = zeros(Float64, D_star, N_slide);

    for tt in 1:N_slide
		temp = Ω_part_slide[:, :, tt]
		feature_slide_vec[:, tt] = temp[tril!(trues(size(temp)), -1)]
	end 

    kmeans_out = kmeans(feature_slide_vec, K);

    idx_clust = kmeans_out.assignments

	if (length(unique(idx_clust)) != K) 
		out = Dict(:Ω_part => NaN, 
               		:discovery => NaN, 
               		:γ => idx_clust, 
               		:μ => μ_slide,
					:accepted_output => false)
    	return out
	else 
		Ω_all = zeros(D, D, K);
		Ω_partial_all = zeros(D, D, K);
	
		for kk in 1:K 
			obs_state = obs[findall(idx_clust .== kk), :];
			Σ_state = mvNormalMLE(obs_state)[:Σ];
			Ω_all[:, :, kk] = glasso(Σ_state, ρ = ρ)
		end 
	
		# partial getCorrelation
		for kk in 1:K
			Ω_partial_all[:, :, kk] = getPartialCorrelation(Ω_all[:, :, kk]);
		end 
	
		discovery_all = Int.(Ω_partial_all .!=0)
		out = Dict(:Ω_part => Ω_partial_all, 
					:discovery => discovery_all, 
					:γ => idx_clust, 
					:μ => μ_slide, 
					:accepted_output => true)
		return out 
	end 



end


# - 
function getStatisticsClassification(classMeasures_all)
    K_true = size(classMeasures_all, 1)
    classMeasures_mean = zeros(Float64, K_true, 5)
    classMeasures_std = zeros(Float64, K_true, 5)

    for ii in 1:K_true 
        for jj in 1:5
            n_NaN = sum(isnan.(classMeasures_all[ii, jj, :]))
            idx_average = findall(.!isnan.(classMeasures_all[ii, jj, :]))
            if (n_NaN > Int(n_simul/2))
                classMeasures_mean[ii, jj] = NaN
                classMeasures_std[ii, jj] = NaN
            else 
                classMeasures_mean[ii, jj] = mean(classMeasures_all[ii, jj, idx_average])
                classMeasures_std[ii, jj] = std(classMeasures_all[ii, jj, idx_average])
            end 
        end 
    end 

    out = Dict(:mean => round.(Matrix(classMeasures_mean), digits = 3), 
               :std => round.(Matrix(classMeasures_std), digits = 3))

end 


# - 
function NetworkMetrics(weightedGraph::Matrix, adjacencyGraph::Matrix)

	R"""
	library("NetworkToolbox")

	Omega_part = ($weightedGraph)
	discovery = ($adjacencyGraph)

	Omega_part_aux = Omega_part
	Omega_part_aux[discovery == 0] = 0

	# - Closeness centrality of each node in a network
	closenessCentrality = closeness(Omega_part_aux, weighted = TRUE)

	# - Eigenvector centrality of each node in a network
	eigenCentrality = eigenvector(Omega_part_aux, weighted = TRUE)

	# - Hybrid centrality of each node in a network (betweenness centrality)
	hybridCentrality = hybrid(discovery, BC = "standard", beta = 0.1)

	# - Global clustering coefficient (CC) and Local clustering coefficient (CCi)
	clustCoeff_global = clustcoeff(discovery)$CC
	clustCoeff_local = clustcoeff(discovery)$CCi

	# - Average and Standard deviation of the weights in the network
	networkConnectivity_mean = conn(Omega_part_aux)$mean

	# - Degree of each node in a network
	nodesDegree = degree(discovery)

	# - Node impact
	nodesImpact = impact(discovery)

	# Computes global average shortest path length, local average shortest path length, eccentricity, and
	# diameter of a network
	pathLengths = pathlengths(Omega_part_aux, weighted = TRUE)
	averageShortestPath_global = pathLengths$ASPL
	averageShortestPath_local = pathLengths$ASPLi
	eccentricity = pathLengths$ecc
	diameter = pathLengths$diameter

	# Randomized Shortest Paths Betweenness Centrality 
	betweensCentrality_rsp = rspbc(discovery, beta = 0.01, comm = NULL)

	# smallworldness Small-worldness Measure
	SWM = qgraph::smallworldness(discovery, B = 1000, up = 0.995, lo = 0.005)
	smallWordlness = SWM[1]

	# within-community centrality for each node in the network
	withinCommunityCentrality = stable(discovery)

	# Node Strength
	strengthNodes = strength(discovery)

	# Transitivity 
	transitivityNet = transitivity(discovery)
	"""

	localMetrics = Dict(:closenessCentrality => rcopy(R"closenessCentrality"), 
					 :eigenCentrality => rcopy(R"eigenCentrality"), 
					 :hybridCentrality => rcopy(R"hybridCentrality"),
					 :clustCoeff_local => rcopy(R"clustCoeff_local"),
					 :nodesDegree => Int.(rcopy(R"nodesDegree")),
					 :nodesImpact => rcopy(R"nodesImpact"),
					 :averageShortestPath_local => rcopy(R"averageShortestPath_local"),
					 :eccentricity => rcopy(R"eccentricity"),
					 :betweensCentrality_rsp => rcopy(R"betweensCentrality_rsp"),
					 :withinCommunityCentrality => rcopy(R"withinCommunityCentrality"),
					 :strengthNodes => Int.(rcopy(R"strengthNodes"))
					 )

	globalMetrics = Dict(:clustCoeff_global => rcopy(R"clustCoeff_global"),
					:networkConnectivity_mean => rcopy(R"networkConnectivity_mean"),
					:averageShortestPath_global => rcopy(R"averageShortestPath_global"),
					:diameter => rcopy(R"diameter"),
					:smallWordlness => rcopy(R"smallWordlness"),
					:transitivityNet => rcopy(R"transitivityNet"))

	out = Dict(:local => localMetrics, 
		 	   :global => globalMetrics)

	return out; 
end 

# - 
function getDataFrame_LocalMetrics(local_metrics, mask_names)

	R"""
	local_metrics <- $local_metrics
	mask_names <- $mask_names
	D <- length(mask_names)
	n_metrics <- length(local_metrics)

	df <- data.frame(matrix(NA, nrow = length(local_metrics), ncol = D+1))
	colnames(df) <- c("Metric", mask_names)

	df[, 1] <- names(local_metrics)
	for (ii in 1:n_metrics) {
		df[ii, 2:(D+1)] <- local_metrics[[ii]]
	}
	"""

	return rcopy(R"df")
end 

# - 
function getDataFrame_GlobalMetrics(global_metrics)
	R"""
	global_metrics = $global_metrics
	n_metrics <- length(global_metrics)

	df <- data.frame(matrix(NA, nrow = length(global_metrics), ncol = 2))
	colnames(df) <- c("Metric", "Value")

	df[, 1] <- names(global_metrics)
	for (ii in 1:n_metrics) {
		df[ii, 2] <- global_metrics[[ii]]
	}
	
	"""
	return rcopy(R"df")
end 




# - 

function plotPosteriorPredictive(obs, n_states, ŷ,
                                 γ̂,
                                 width, height;
								 state_seq_true = nothing, 
                                 path_file = nothing, ylabels = nothing, 
                                 only_mean = false, add_mean = false)
    N = size(obs, 1)
    D = size(obs, 2)
    K = copy(n_states)
	ŷ_mean = reshape(mean(ŷ, dims = 3), N, D)

    save = ifelse(!isnothing(path_file), true, false)
    plot_state_seq_true = ifelse(!isnothing(state_seq_true), true, false)
    change_ylabels = ifelse(!isnothing(ylabels), true, false)
    changePoints = vcat(0, findall(diff(γ̂) .!= 0), N)
    # colorPalette = palette(:roma, length(unique(γ̂)))
    # colorPalette = palette(:roma, K)
    #colorPalette = palette(:lighttest, K)
    colorPalette = palette(:rainbow1, K)
    # candidate: colorPalette = palette(:seaborn_bright6, K)
    #colorPalette = palette(:jet, K)

    plots = plot(layout = (1, 1))
    plot!(plots, size=(width, height))

	if only_mean 
		for dd in 1:D 
			plot!(
				plots[1, 1],
				ŷ_mean[:, dd],
				color = :grey,
				legend = :topright,
				label = "",
				alpha = 0.3,
				ylims = (minimum(obs) - 0.3,  maximum(obs) + 0.3),
				xlims = (-10, N+10),
				xticks = [1, 500, 1000, 1500, 2000],
				xtickfont = font(10),
				ytickfont = font(10),
				yguidefontsize=13,
				dpi = 1000,
				ylab = "y",
				xlab = "Time"
				)
		end
	else 
		for dd in 1:D 
			for tt = 1:size(ŷ, 3)
				plot!(
					plots[1, 1],
					ŷ[:, dd, tt],
					color = :grey,
					legend = :topright,
					label = "",
					alpha = 0.01,
					ylims = (minimum(obs) - 0.3,  maximum(obs) + 0.3),
					xlims = (-10, N+10),
					xticks = [1, 500, 1000, 1500, 2000],
					xtickfont = font(10),
					ytickfont = font(10),
					yguidefontsize=13,
					dpi = 1000,
					ylab = "y",
					xlab = "Time"
					)
			end
	
		end 

		if add_mean  

			for dd in 1:D 
				plot!(
					plots[1, 1],
					ŷ_mean[:, dd],
					color = :grey,
					legend = :topright,
					label = "",
					alpha = 0.3,
					ylims = (minimum(obs) - 0.3,  maximum(obs) + 0.3),
					xlims = (-10, N+10),
					xticks = [1, 500, 1000, 1500, 2000],
					xtickfont = font(10),
					ytickfont = font(10),
					yguidefontsize=13,
					dpi = 1000,
					ylab = "y",
					xlab = "Time"
					)
			end

		end 
	end 


    for ii = 1:(length(changePoints)-1)
        z_span = [changePoints[ii] + 1, changePoints[ii+1]]
        vspan!(
            plots[1, 1],
            z_span,
            color = colorPalette.colors[γ̂[changePoints[ii+1]]],
            alpha = 0.12,
            labels = :none,
            dpi = 1000
        )
    end 
    if save
        savefig(plots, path_file * ".png")
    end
    display(plots)
end

# - 
function plotData(obs, n_states, state_seq_true, 
                                 width, height;
                                 path_file = nothing, ylabels = nothing, 
                                )

    N = size(obs, 1)
    D = size(obs, 2)
    K = copy(n_states)

    save = ifelse(!isnothing(path_file), true, false)
    change_ylabels = ifelse(!isnothing(ylabels), true, false)
    changePoints = vcat(0, findall(diff(state_seq_true) .!= 0), N)
    # colorPalette = palette(:roma, length(unique(γ̂)))
    # colorPalette = palette(:roma, K)
    #colorPalette = palette(:lighttest, K)
    colorPalette = palette(:rainbow1, K)
    # candidate: colorPalette = palette(:seaborn_bright6, K)
    #colorPalette = palette(:jet, K)

    plots = plot(layout = (1, 1))
    plot!(plots, size=(width, height))

    for dd in 1:D 
        plot!(
            plots[1, 1],
            obs[:, dd],
            color = :grey,
            legend = :topright,
            label = "",
            alpha = 0.3,
			ylims = (minimum(obs) - 0.3,  maximum(obs) + 0.3),
            xlims = (-10, N+10),
            xticks = [1, 500, 1000, 1500, 2000],
            xtickfont = font(10),
            ytickfont = font(10),
            yguidefontsize=13,
            dpi = 1000,
            ylab = "y",
            xlab = "Time"
            )
    end

    for ii = 1:(length(changePoints)-1)
        z_span = [changePoints[ii] + 1, changePoints[ii+1]]
        vspan!(
            plots[1, 1],
            z_span,
            color = colorPalette.colors[state_seq_true[changePoints[ii+1]]],
            alpha = 0.11,
            labels = :none,
            dpi = 1000
        )
    end 
    if save
        savefig(plots, path_file * ".png")
    end
    display(plots)

end


R"""
plotStateProbsR <- function(state_probs, plt_legend, z_col) {
    K <- nrow(state_probs)
    N <- ncol(state_probs)
    time <- 1:N
    plot(1:N, type = "n",  ylab = "Prob State", xaxt = "n", xlab = "",
	     cex = 0.5, las = 1,
	     xlim = c(1, N),
	     ylim = c(0,1), cex.lab = 1.3, cex.axis = 1.3)
    plot_p <- matrix(NA,nrow=N-1,ncol=2*K)
    a <- 0

    for (n in 2:N) {
        for (j in 1:K) {
            plot_p[n-1, (j*2-1)] <- state_probs[j, n-1]
            plot_p[n-1, j*2] <- state_probs[j, n]
            # if	(j==1){col_states<- z_col[1]}
            # if	(j==2){col_states<- z_col[2]}
            # if	(j==3){col_states<- z_col[3]}
            # if	(j==4){col_states<- z_col[4]}
            # if	(j==4){col_states<- z_col[4]}
            # if	(j==4){col_states<- z_col[4]}
            col_states <- z_col[j]
    
            # (at some point need to make function to
            # generalize for all K, like this is redundant)
            if	(j==1){
              point_1<-a
              point_2<-point_1+plot_p[n-1,(j*2-1)]
              point_4<-a
              point_3<-point_4+plot_p[n-1,(j*2)]	}
    
            if	(j==2){
              point_1<-a+plot_p[n-1,(j-1)*2-1]
              point_2<-point_1+plot_p[n-1,(j*2-1)]
              point_4<-a+plot_p[n-1,(j-1)*2]
              point_3<-point_4+plot_p[n-1,(j*2)]	}
    
            if	(j==3){
              point_1<-a+plot_p[n-1,(j-2)*2-1]+plot_p[n-1,(j-1)*2-1]
              point_2<-point_1+plot_p[n-1,(j*2-1)]
              point_4<-a+plot_p[n-1,(j-2)*2]+plot_p[n-1,(j-1)*2]
              point_3<-point_4+plot_p[n-1,(j*2)]}
            if (j==4) {
              point_1 <- a+ plot_p[n-1,(j-3)*2-1] + plot_p[n-1,(j-2)*2-1]+plot_p[n-1,(j-1)*2-1]
              point_2 <- point_1+plot_p[n-1,(j*2-1)]
              point_4 <- a + plot_p[n-1,(j-3)*2] + plot_p[n-1,(j-2)*2]+plot_p[n-1,(j-1)*2]
              point_3 <- point_4+plot_p[n-1,(j*2)]}
            if (j ==5) {
              point_1 <- a+ plot_p[n-1,(j-4)*2-1] + plot_p[n-1,(j-3)*2-1]+plot_p[n-1,(j-2)*2-1] +plot_p[n-1,(j-1)*2-1]
              point_2 <- point_1+plot_p[n-1,(j*2-1)]
              point_4 <- a + plot_p[n-1,(j-4)*2] + plot_p[n-1,(j-3)*2]+plot_p[n-1,(j-2)*2] + +plot_p[n-1,(j-1)*2]
              point_3 <- point_4+plot_p[n-1,(j*2)]}
            if (j ==6) {
              point_1 <- a+ plot_p[n-1,(j-5)*2-1] + plot_p[n-1,(j-4)*2-1] + plot_p[n-1,(j-3)*2-1]+plot_p[n-1,(j-2)*2-1] +plot_p[n-1,(j-1)*2-1]
              point_2 <- point_1+plot_p[n-1,(j*2-1)]
              point_4 <- a + plot_p[n-1,(j-5)*2] + plot_p[n-1,(j-4)*2] + plot_p[n-1,(j-3)*2]+plot_p[n-1,(j-2)*2] + +plot_p[n-1,(j-1)*2]
              point_3 <- point_4+plot_p[n-1,(j*2)]}
    
    
            polygon(c(time[n-1],time[n-1],time[n],time[n]),
                    c(point_1,point_2,point_3,point_4),col=scales::alpha(col_states, 0.8), border=NA)
            lines(c(time[n-1],time[n]),c(point_2,point_3), col=scales::alpha(col_states, 0.5))
          }
        }
    
        legend("topleft",
                legend=plt_legend,
                col = z_col,
                fill = z_col,
                bty = "n",
                cex = 1)

}
"""