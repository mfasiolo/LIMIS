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
# - verbose: if true some information is printed as the algorithm runs. Default is verbose = true.
#### OUTPUT
# - expVar: the estimated variance of the importance weights corresponding to each value in tseq.
#
function tuneIMIS(obj, tseq; frac = 1.0, verbose = false)

  # Total number of samples
  total = size(obj["X₀"], 2);

  # Select a fraction "frac" of samples to be used and store the indices in subInd
  np = round(Int, frac * total);
  subInd = sample(1:total, np, replace = false);

  # Extracting stuff from obj
  X₀ = obj["X₀"][:, subInd];
  μ₀ = obj["μOrig"];
  w = obj["w"][subInd];
  wmix = obj["wmix"];
  dLogTar = obj["dLogTar"][subInd];
  dLogPrior = obj["dLogPrior"][subInd];
  control = obj["control"];

  # Normalize weights
  w = w / sum(w)

  d, nmix = size( μ₀ );

  # The weight of the prior is proportional to the fraction of samples from the prior
  α = control["n₀"] / (control["n₀"] + control["niter"] * control["n"]);

  nt = length( tseq );
  δt = diff( [0; tseq] );

  # Output: expected variance of the importance sampling estimates
  expVar = Array(Float64, nt);

  dmix = Array(Float64, np);

  # Storage for means and covariance. Σ is "nothing" so createLangMix() does not use it in the first iteration
  μ = copy(μ₀);
  Σ = nothing;

  # Storage to be used by dGausMix()
  dTrans = Array(Float64, np, nmix);

  # Progressively increasing time t and estimating the importance sampling variance
  for ii = 1:nt

    if verbose @printf("%d ", ii); end

    # Create Langevin mixture importance density efficiently, because we start from Σ computed by previous iteration
    μ, Σ = createLangMix(μ, control["score"], control["hessian"], δt[ii]; targetESS = control["targetESS"], Σ₀ = Σ);

    # Evaluate weighted mixture
    dmix = α * exp(dLogPrior) .+ (1.-α) * dGausMix(X₀, μ, Σ; df = control["df"], w = wmix, Log = false, dTrans = dTrans)[1];

    # Cross-validate using existing samples
    expVar[ii] = dot(exp(dLogTar)./dmix, w);

  end

  return expVar

end
