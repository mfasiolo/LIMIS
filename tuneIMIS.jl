###########################################
# Choose the length of time-integration used by langevin IMIS
###########################################
### DESCRIPTION
# This function can be used to tune the pseudo-time t integration used
# to create the linearized Langevin Mixture. Specifically, the function
# calculated estimate the variance of the importance weights as a function
# of t, and then one can choose the t that minimizes it. The input is the
# output of an IMIS() run, and the tuning is done by exploiting the sample
# already drawn by IMIS().
### INPUT
# - obj: a dictionary which is the output of an IMIS() run.
# - tseq: an increasing sequence of times at which the variance of the weights has to be estimated.
# - frac: the fraction of samples in obj used to estimate the variance of the weights.
#         frac ∈ (0, 1]. Reducing frac speeds up the estimation, but reduces the accuracy.
#         Default is frac = 1.0.
# - self: if true the weights need to be normalized
# - crit: if crit == "kl" the kl divergence will be evaluated, if crit == "var" the variance will be evaluated
# - verbose: if true some information is printed as the algorithm runs. Default is verbose = true.
#### OUTPUT
# - expVar: for each value in tseq: the estimated kl divergence between target and importance density or
#           the estimated variance of the importance weights corresponding to each value in tseq.
#
@everywhere function tuneIMIS(obj, tseq; frac = 1.0, self = true, crit = "kl", verbose = false)

  if self && (crit=="var")   @printf("If self-normalized IS is used, it is better setting argument \"crit\" to \"kl\""); end

  # Total number of samples
  total = size(obj["X₀"], 2);

  # Select a fraction "frac" of samples to be used and store the indices in subInd
  np = round(Int, frac * total);
  subInd = sample(1:total, np, replace = false);

  # Extracting stuff from obj
  X₀ = obj["X₀"][:, subInd];
  μ₀ = obj["μOrig"];
  logw = obj["logw"][subInd];
  wmix = obj["wmix"];
  dLogTar = obj["dLogTar"][subInd];
  dLogPrior = obj["dLogPrior"][subInd];
  control = obj["control"];

  d, nmix = size( μ₀ );

  # The weight of the prior is proportional to the fraction of samples from the prior
  nk = (control["n₀"] + control["niter"] * control["n"])
  α = control["n₀"] / nk;
  nt = length( tseq );
  δt = diff( [0; tseq] );

  # If self-normalized IS used, calculate normalizing constant
  chat = 1.;
  if self    chat = meanExpTrick( obj["logw"] );   end

  # Output: expected variance of the importance sampling estimates
  out = Array(Float64, nt);

  dmix = Array(Float64, np);

  # Mean and covariance. Σ is "nothing" so createLangMix() does not use it in the first iteration
  μ = copy(μ₀);
  Σ = nothing;

  # Storage for mean and covariance, one for each value in tseq
  μStore = Array(Float64, d, nmix, nt);
  ΣStore = Array(Float64, d, d, nmix, nt);

  # Storage to be used by dGausMix()
  dTrans = Array(Float64, np, nmix);

  # Progressively increasing time t and estimating the importance sampling variance
  for ii = 1:nt

    if verbose @printf("%d ", ii); end

    # Create Langevin mixture importance density efficiently, because we start from Σ computed by previous iteration
    μ, Σ = createLangMix(μ, control["score"], control["hessian"], δt[ii]; targetESS = control["targetESS"], Σ₀ = Σ);

    # Store
    μStore[:, :, ii] = μ;
    ΣStore[:, :, :, ii] = Σ;

    # Evaluate log importance density with no underflow
    logdmix = dGausMix(X₀, μ, Σ; df = control["df"], w = wmix, Log = true, dTrans = dTrans)[1];
    minp = minimum( [dLogPrior; logdmix] );
    logdmix = minp + log(α * exp(dLogPrior-minp) + (1-α) * exp(logdmix-minp));

    # Calculate variance of ...
    if crit == "kl" # self-normalized IS estimator: compute negative cross-entropy OR...

     mxtmp = maximum(logw);
     out[ii] = - exp(mxtmp) * dot(logdmix, exp(logw-mxtmp)) / (chat*np);

    elseif crit == "var" # ... standard IS estimator: compute estimated variance of IS

        tmp = dLogTar + logw - logdmix ;
        mxtmp = maximum(tmp);
        out[ii] = log(sum(exp(tmp-mxtmp))) + mxtmp - log(np);

    else error("Argument crit should be either \"kl\" or \"var\".")  end

  end

  output = Dict{Any,Any}("expVar" => out, "μ" => μStore, "Σ" => ΣStore);

  return output

end
