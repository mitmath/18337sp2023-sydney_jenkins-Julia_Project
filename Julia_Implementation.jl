### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ c5204406-eac8-11ed-2916-83e3ac9c807c
begin 
	import Pkg
	Pkg.add("CSV")
	using CSV
	using LinearAlgebra
	using DataFrames
	Pkg.add("Distributions")
	using Distributions 
	Pkg.add("Plots")
	using Plots
	Pkg.add("LsqFit")
	using LsqFit
	Pkg.add("QuadGK")
	using QuadGK
	using Base.Threads
	Pkg.add("BenchmarkTools")
	using BenchmarkTools
end

# ╔═╡ a53ed098-5e7a-4f92-819b-605572f1167b
begin 
	#Get data
	time = Matrix(CSV.read("time_test.csv", DataFrame))
	flux = Matrix(CSV.read("flux_test.csv", DataFrame))
	e_flux = Matrix(CSV.read("e_flux_test.csv", DataFrame))
	t0 = Matrix(CSV.read("t0_test.csv", DataFrame))
	pdepths = Matrix(CSV.read("pdepth_test.csv", DataFrame))
	P = Matrix(CSV.read("P_test.csv", DataFrame))

	#Get rid of zeros -- I had used these to make all the arrays the same length to store in csv 
	function remove_zeros(A::Vector{Float64})
	    B = A[findall(!iszero, A)]
		return B
	end
end

# ╔═╡ 635e2e35-7b42-4612-b342-7cbfd3fa7fec
nthreads()

# ╔═╡ 8f9f0cb1-6354-42a5-a3b1-3f83a29960db
md"""
MCMC Code
"""

# ╔═╡ 3ed55dae-3565-4a4f-ba1f-da0d7ad22456
begin
	#MCMC functions
	
	function log_likelihood(y, x, params, σ, model, P)
		#Log-likelihood function
	    y_pred = model(x, params, P) #Predicted value using current params
	    ll = sum(logpdf.(Normal.(y_pred, σ), y))
	    return ll
	end
	
	function sampler(y, x, p0, priors, σ, model, n_iterations, chain, log_prior, P)
		#Define sampler
	    n_params = length(p0)
	    lp = log_likelihood(y, x, p0, σ, model, P)
	    lp += log_prior(p0..., priors)
	    for i in 2:n_iterations
	        p_new = chain[:, i-1] .+ rand(Normal(0, 0.1), n_params)
	        lp_new = log_likelihood(y, x, p_new, σ, model, P) + log_prior(p_new..., priors)
	        if lp_new - lp > log(rand())
	            chain[:, i] = p_new
	            lp = lp_new
	        else
	            chain[:, i] = chain[:, i-1]
	        end
	    end
	    return chain
	end
end

# ╔═╡ 9b209eb9-bf23-4ac1-bfde-8e0eba7dea68
md"""
Box Model
"""

# ╔═╡ 891545b2-da00-4543-a800-e4265a3422a9
begin 
	function simple_model(x, p, P)
		#Define Box model
		t0, depth, duration = p
	    box = ones(size(x))
	    box[(x .>= t0) .& (x .<= t0 + duration)] .= 1 - depth
	    return box
	end

	function log_prior_simple(t0, depth, duration, priors)
		#Define log priors
		t0_prior, depth_prior, duration_prior = priors
	    lp = logpdf(t0_prior, t0) + logpdf(depth_prior, depth) + logpdf(duration_prior, duration)
	    return lp
	end

	function fit_simple(i, parallel)
		#Remove zeros from lightcurve
		x_simple = remove_zeros(Matrix(time)[i,:])
		y_simple = remove_zeros(Matrix(flux)[i,:])
		e_y_simple = remove_zeros(Matrix(e_flux)[i,:])
		
		#Intial guess and priors
		p0_test = [t0[i]-.04, 0.05, .08]
		perts = [.1,.01,.02] #perturbations
		t0_prior_simple = Uniform(t0[i]-.2, t0[i]+.2) #Normal(t0[1], 1)
		depth_prior_simple = Uniform(0.01, .1)
		duration_prior_simple = Uniform(.01, .5) #Normal(.1, .1)
		priors = [t0_prior_simple, depth_prior_simple, duration_prior_simple]
	
		#Sampler parameters
		n_iters = 10000
		burnin = 2000 #20000
		nchains = 128
		
		#Make chains
		chains = zeros(length(p0_test), n_iters, nchains)
		params_init = [p0_test + randn(length(p0_test)) .* perts for i in 1:nchains]
		
		#Loop through chains -- parallel or serial options
		if parallel
			Threads.@threads for thd = 1:nthreads()
	           for chain_ct in 1:nchains
				    if mod(chain_ct,nthreads())+1==thd
						chains[:, 1,chain_ct] = params_init[chain_ct]
						chains[:,:,chain_ct] = sampler(y_simple, x_simple, p0_test, priors, e_y_simple, simple_model, n_iters, chains[:,:,chain_ct], log_prior_simple, 0)
					end
				end
	       end
		else 
			for chain_ct in 1:nchains
				chains[:, 1,chain_ct] = params_init[chain_ct]
				chains[:,:,chain_ct] = sampler(y_simple, x_simple, p0_test, priors, e_y_simple, simple_model, n_iters, chains[:,:,chain_ct], log_prior_simple, 0)
			end
		end
		
		#Calculate mean of each parameter
		t0_mean = mean(chains[1, burnin:end,:])
		depth_mean = mean(chains[2, burnin:end,:])
		duration_mean = mean(chains[3, burnin:end,:])
		fitparams_simple = [t0_mean, depth_mean, duration_mean]

		#Calculate goodness of fit using RMSE
		prediction_simple = simple_model(x_simple, fitparams_simple, P[i])
		rmse_simple = sqrt(sum((y_simple.-prediction_simple).^2)/length(y_simple)) 

		return fitparams_simple, rmse_simple
	end

	function iterate_simple(ndata, parallel)
		#Iterate through data
		all_rmse = zeros(ndata)
		for i in 1:ndata
			fitparams_simple_i, rmse_simple_i = fit_simple(i, parallel)
			all_rmse[i] = rmse_simple_i
		end
		return mean(all_rmse), std(all_rmse)
	end
end

# ╔═╡ 9552116f-8556-42cd-b5cc-107821798f82
begin 
	#Example serial fit of Box model, change false to true for parallelization
	@time fp2, rm2 = fit_simple(1,false)
end

# ╔═╡ 9e310f18-3a83-4e3d-9ea8-5eed8140b958
md"""
Complex Model
"""

# ╔═╡ cd842114-59f9-455e-952b-ff026bc3aae4
begin 
	#Functions for the complex model
	
	function log_prior_MA(t0, gam1, gam2, p, Ra, priors)
		#Define priors
		t0_prior, gam1_prior, gam2_prior, p_prior, Ra_prior = priors
	    lp = logpdf(t0_prior, t0) + logpdf(gam1_prior, gam1) + logpdf(gam2_prior, gam2) + logpdf(p_prior, p) + logpdf(Ra_prior, Ra)
	    return lp
	end
	
	function find_I(r, gam1, gam2)
	    mu = sqrt(1 - r^2)
	    return 1 - gam1*(1 - mu) - gam2*(1 - mu)^2
	end
	
	function find_F(p, z)
	    if (1+p) < z
	        lamb = 0
		elseif abs(1-p) < z && z <= (1+p)
	        coeff1 = p^2*acos((p^2+z^2-1)/(2*p*z))
	        coeff2 = acos((1-p^2+z^2)/(2*z))
	        coeff3 = sqrt((4*z^2 - (1+z^2-p^2)^2)/4)
	        lamb = (1/π) * (coeff1+coeff2-coeff3)
		elseif z <= (1-p)
	        lamb = p^2
	    else
	        lamb = 1
	    end
	    return 1 - lamb
	end
	
	function find_dFdr(p, z, r)
	    p2 = p/r  # Used for checking conditions
	    z2 = z/r
	    if (1+p2) < z2
	        lamb = 0
		elseif abs(1-p2) < z2 && z2 < (1+p2)
	        # Coeff1
	        t1 = p/(r*z*sqrt(1-((p^2-r^2+z^2)^2 / (4*p^2*z^2))))
	        k0 = acos((p^2+z^2-r^2)/(2*p*z))
	        t2 = 2*p^2*k0 / r^3
	        coeff1 = t1-t2
	        # Coeff2
	        num = p^2/r^2 - z^2/r^2 + 1
	        denom = 2*z*sqrt(1-((-p^2/r+z^2/r+r)^2 / (4*z^2)))
	        coeff2 = num/denom
	        # Coeff3
	        num = 2*(2*p^2/r^3 - 2*z^2/r^3) * (-p^2/r^2 + z^2/r^2 + 1) + (8*z^2/r^3)
	        denom = 4*sqrt(4*z^2 / r^2 - (-p^2 / r^2 + z^2 / r^2 + 1)^2)
	        coeff3 = num/denom
	        # Combine
	        lamb = (1/π) * (coeff1-coeff2+coeff3)
		elseif z2 < (1-p2)
	        lamb = -2*p^2 / r^3
	    else
	        lamb = 0
	    end
	    return -1 * lamb
	end
	
	function find_dFdr_r(p, z, r)
	    return find_F(p/r, z/r)*2*r + find_dFdr(p, z, r)*r^2
	end

	function Mandel_Agol_model(x, p, P)
		#Define Mandel Agol model
		t0, gam1, gam2, p, Ra = p
		w = 2π/P
		all_F = Float64[]
		for i in 1:length(x)
		    z = (1/Ra)*sqrt(sin(w*(x[i]-t0))^2)#+(cos(i)*cos(w*(x[i]-t0)))^2) 
		    I1, = quadgk(r -> 2*r*find_I(r, gam1, gam2), 0, 1)
		    I2, = quadgk(r -> find_dFdr_r(p, z, r)*find_I(r, gam1, gam2), 0, 1)
		    F = I2/I1
			push!(all_F, F)
		end
	    return all_F
	end
end


# ╔═╡ 27beb0df-3312-4ce2-a64d-4e1ec6f617a2
begin 
	#Now fit complex model 	
	function fit_MA(i, parallel)
		#Remove zeros from lightcurve
		x_MA = remove_zeros(Matrix(time)[i,:])
		y_MA = remove_zeros(Matrix(flux)[i,:])
		e_y_MA = remove_zeros(Matrix(e_flux)[i,:])
		
		#Initial guesses -- made smarter with Kepler results
		#params: t0, gam1, gam2, p (Rp/Rs), Ra (Rs/semimajor axis)
		t0_guess = t0[i]
		p_guess = pdepths[i]^.5
		p0_MA = [t0[i],.18,.15,pdepths[i]^.5,.05]
		perts_MA = [.01,.02,.02,.02,.1] #perturbations
	
		#Priors
		t0_prior_MA = Uniform(t0[i]-.2, t0[i]+.2) 
		gam1_prior = Uniform(-1, 1)
		gam2_prior = Uniform(-1, 1)
		p_prior = Uniform(0, .5)
		Ra_prior = Uniform(.001, 1)
		priors_MA = [t0_prior_MA, gam1_prior, gam2_prior, p_prior, Ra_prior]
	
		#Sampler parameters
		n_iters_MA = 10000
		burnin_MA = 2000
		nchains_MA = 128
		
		#Initialize chain
		chains_MA = zeros(length(p0_MA), n_iters_MA, nchains_MA)
		params_init_MA = [p0_MA + randn(length(p0_MA)) .* perts_MA for i in 1:nchains_MA]
		
		#Loop through chains -- parallel or serial
		if parallel
			Threads.@threads for thd = 1:nthreads()
	           for chain_ct in 1:nchains_MA
				    if mod(chain_ct,nthreads())+1==thd
						chains_MA[:, 1,chain_ct] = params_init_MA[chain_ct]
						chains_MA[:,:,chain_ct] = sampler(y_MA, x_MA, p0_MA, priors_MA, e_y_MA, Mandel_Agol_model, n_iters_MA, chains_MA[:,:,chain_ct], log_prior_MA, P[i])
					end
				end
	       end
		else 
			for chain_ct in 1:nchains_MA
				chains_MA[:, 1,chain_ct] = params_init_MA[chain_ct]
				chains_MA[:,:,chain_ct] = sampler(y_MA, x_MA, p0_MA, priors_MA, e_y_MA, Mandel_Agol_model, n_iters_MA, chains_MA[:,:,chain_ct], log_prior_MA, P[i])
			end
		end
		
		#Calculate mean of each parameter
		t0_mean_MA = mean(chains_MA[1, burnin_MA:end,:])
		gam1_mean = mean(chains_MA[2, burnin_MA:end,:])
		gam2_mean = mean(chains_MA[3, burnin_MA:end,:])
		p_mean = mean(chains_MA[4, burnin_MA:end,:])
		Ra_mean = mean(chains_MA[5, burnin_MA:end,:])
		fitparams_MA = [t0_mean_MA, gam1_mean, gam2_mean, p_mean, Ra_mean]

		#Calculate goodness of fit using RMSE
		prediction_MA = Mandel_Agol_model(x_MA, fitparams_MA, P[i])
		rmse_MA = sqrt(sum((y_MA.-prediction_MA).^2)/length(y_MA)) 

		return fitparams_MA, rmse_MA
	end

	function iterate_MA(ndata, parallel)
		#Iterate through data
		all_rmse = zeros(ndata)
		for i in 1:ndata
			fitparams_MA_i, rmse_MA_i = fit_MA(i, parallel)
			all_rmse[i] = rmse_MA_i
		end
		return mean(all_rmse), std(all_rmse)
	end
	
end

# ╔═╡ 497dcdf8-982c-4f07-9f61-d2cf9fce0497
begin 
	#Example serial fit of MA model, change false to true for parallelization
	@time fp3, rm3 = fit_MA(1,false)
end

# ╔═╡ Cell order:
# ╠═c5204406-eac8-11ed-2916-83e3ac9c807c
# ╠═a53ed098-5e7a-4f92-819b-605572f1167b
# ╠═635e2e35-7b42-4612-b342-7cbfd3fa7fec
# ╟─8f9f0cb1-6354-42a5-a3b1-3f83a29960db
# ╠═3ed55dae-3565-4a4f-ba1f-da0d7ad22456
# ╟─9b209eb9-bf23-4ac1-bfde-8e0eba7dea68
# ╠═891545b2-da00-4543-a800-e4265a3422a9
# ╠═9552116f-8556-42cd-b5cc-107821798f82
# ╟─9e310f18-3a83-4e3d-9ea8-5eed8140b958
# ╠═cd842114-59f9-455e-952b-ff026bc3aae4
# ╠═27beb0df-3312-4ce2-a64d-4e1ec6f617a2
# ╠═497dcdf8-982c-4f07-9f61-d2cf9fce0497
