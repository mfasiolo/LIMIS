##################
#### Incremental Mixture Importance Sampling
##################
### DESCRIPTION
# This is an improved version of the IMIS algorithm described in
# "Estimating and Projecting Trends in HIV/AIDS Generalized Epidemics Using Incremental Mixture Importance Sampling".
# Besides implementing the Nearest Neighbours approach of the above paper, it also allows the create
# the local Gaussian mixture by linearizing a Langevin diffusion.
# Another improvement is that, while the above paper always adds the sample with the highest weight to the mixture,
# we do it only if it does not fall to close to any existing mixture component. This is done by calculating the
# Mahalanobis distance of the sample wrt all mixture components, and comparing with a chosen quantile of a χ²(d)
# distribution. If any distance is lower than that quantile, we find the mixture component with the lowest distance
# and we increase its weight by one, rather than adding a new mixture component.
# The distance check is done after the weights of the new samples have been calculated. If Langevin is used, the
# check will be done again, after propagating the highest weighted sample using the Langevin linearization.
### INPUT
# - niter: total number of iterations. If quant == 0.0 then niter will also be
#          the number of components in the mixture
# - n: number of samples generated at each iteration by the latest density that have been added to the mixture.
# - n₀: number of initial samples from the prior
# - dTarget: function returning the (log) density of the target
# - dPrior: function returning the (log) density of the prior
# - rPrior: function that generates samples from the prior
# - df: degrees of freedom used by the Multivariante T mixture. Default is df = 3
# - trunc: should Truncated Importance Sampling be used. Default is trunc = true
# - quant: quantile of a χ²(d) variable, used to chose to increase the weight of an existing mixture component,
#          rather than creating a new component. quant ∈ [0, 1]. As quant increases, new mixture components will
#          be created less frequently. Default is quant = 0.25.
# - useLangevin: if true new densities will be created using a Langevin linearization, if false Nearest Neighbours
#                will be used instead. Default is useLangevin = true.
# - verbose: if true some information is printed as the algorithm runs. Default is verbose = true.
# - targetESS: the expected ESS to aim for at each step of the Langevin Linearization. targetESS ∈ (0, 1).
#              It determines the step size during linearization, as targetESS increases the steps become smaller.
#              Needed only if useLangevin == true. Default is targetESS = 0.999.
# - t₀: pseudo-time at which the linearization stops. t₀ > 0. Needed only if useLangevin == true.
# - Q: optional scaling matrix for the Langeving diffusion. Needed only if useLangevin == true. Currently not used.
# - score: gradient of the target density. Needed only if useLangevin == true.
# - hessian: hessian of the target density. Needed only if useLangevin == true.
# - B: number of nearest neighbours to be used to construct the covariance. Relevant is useLangevin == false.
#      By default B = n
# - maxMix: maximum number of mixture components. When it is reached the mixture growth will slow down drastically,
#           but it will not stop entirely. By default maxMix = Inf.
#
### OUTPUT
# The output is a dictionary with the following entries
# - ESS: the ESS achieved at each iteration.
# - X₀: all the samples generated
# - μ₀: the means of the mixture importance density at the final iteration
# - Σ₀: the covariance matrices of the mixture importance density at the final iteration[:, :, 1:nmix]
# - μOrig: the samples that had the highest weights at each iteration. If useLangevin == true than μOrig = μ₀.
# - w: the sample weights at the final iteration.
# - wmix: the normalized mixture weights.
# - dLogTar: the value of the log-target density at each sample point in X₀.
# - dLogPrior: the value of the log-prior density at each sample point in X₀.
# - dimMix: the number of mixture component at each iteration.
# - control: dictionary of internal controls, to be used by other methods.
#
function IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
              df = 3, trunc = true, quant = 0.01, useLangevin = true, verbose = true,
              targetESS = 1-1e-3, t₀ = nothing, Q = nothing, score = nothing, hessian = nothing,
              maxMix = Inf, B = nothing)

  # Safety checks
  if useLangevin && (t₀ == nothing || score == nothing || hessian == nothing)

    error("If useLangevin == true you have to specify ϵ, lag, Q, score and hessian.")

  end

  if B == nothing   B = n;   end

  # Sample from prior
  x = rPrior( n₀ );
  d = size(x, 1);

  # Storage for mixture means and covariance matrices
  μ₀ = zeros(d, niter);
  μOrig = zeros(d, niter);

  Σ₀ = zeros(d, d, niter);

  # Storage for simulated variables
  totSim = n₀ + n * niter;
  X₀ = zeros(d, totSim);
  X₀[:, 1:n₀] = x;

  # Storage for all samples and their density wrt all mixture components
  dImpStore = zeros(totSim, niter);
  dImpAcc = zeros( totSim );

  # Storage for log-target and log-prior
  dLogTar = zeros( totSim );
  dLogPrior = zeros( totSim );

  # Evaluate target and prior
  dLogTar[1:n₀] = dTarget(x; Log = true)[:];
  dLogPrior[1:n₀] = dPrior(x; Log = true)[:];

  # Mixture weights and number of mixture components
  wmix = ones(niter);
  dimMix = zeros(niter);
  nmix = 0;
  neigh = NaN;

  # Calculate importance weights
  logw = dLogTar[1:n₀] .- dLogPrior[1:n₀];

  # Find sample with maximum weight and indicate that it must be added to the mixture
  μ = X₀[:, findfirst( logw .== maximum(logw) )];
  add = true;

  # (Optionally) Truncate importance weights
  if trunc

    logw = min(logw, log(length(w))/2);

  end

  # Calculate normalized weigths using sumExpTrick.
  tmp = maximum( logw );
  wn = exp(logw - tmp) / sum( exp(logw - tmp) );

  ESS = zeros( niter + 1 );
  ESS[1] = ( 1 / sum( wn.^2 ) ) / n₀;

  for ii = 1:niter ##### START main loop

    if verbose @printf("%d \n", ii); end

    # If the maximum number of mixture has been reached, set quant to almost 1. We don't set it to
    # 1 because we still want to add mixture components that are very different from the old ones.
    # If you set quant=1. then the ESS might decrease when maxMix is reached
    if nmix > maxMix   quant = 0.99   end

    # Check if μ is very close to an existing mixture component
    if quant > 0. && ii > 1

      dist = map(kk_ -> maha(μ, μ₀[:, kk_], Σ₀[:, :, kk_])[1], 1:nmix);
      add = minimum(dist) > quantile(Chisq(d), quant);

    end

    if add

      # Save it before it gets modified
      μ_saved = copy(μ);

      # Estimate mean and covariance of new mixture components using...
      if useLangevin # ... linearized Langevin or ...

        μ, Σ = createLangMix(μ, score, hessian, t₀; targetESS = targetESS);
        Σ = Σ[:, :, 1];

      else  # ... n nearest neighbours

        # Raftery in step 2.a of the IMIS paper says to use tmpw = 0.5*(wn + 1/length(wn))
        # but this does not quite work in high dimensions.
         tmpw = wn.*0 + 1/length(wn);
         X_old = X₀[:, 1:(n₀ + n*(ii-1))];
         Σ = nnCov(μ, tmpw, X_old, cov(X_old'), min(B, n₀ + n*(ii-1)))[:, :, 1];

      end

      # Second check on the propagated mixture component, to check if it's actually worth adding it
      if quant > 0. && ii > 1

        #dist = map(kk_ -> - expESS(μ[:], μ₀[:, kk_][:], Σ, Σ₀[:, :, kk_]), 1:nmix);
        #add = maximum( abs(dist) ) < 0.5;

        dist = map(kk_ -> maha(μ, μ₀[:, kk_], Σ₀[:, :, kk_])[1], 1:nmix);
        add = minimum(dist) > quantile(Chisq(d), quant);

      end

      if(add)

        # The mixture has grown by one component
        nmix += 1;

        # Store the unmodified μ
        μOrig[:, nmix] = μ_saved;

        # Add new component to the mixture
        μ₀[:, nmix] = μ;
        Σ₀[:, :, nmix] = Σ;

      end

    end

    if !add

      neigh = findfirst( dist .== minimum(dist) );

      μ =  μ₀[:, neigh];
      Σ = Σ₀[:, :, neigh];

      wmix[neigh] += 1;

    end

    # Simulate from new mixture component and add samples to storage
    x = rmvt(n, μ, Σ, df);

    start = n₀ + (ii-1) * n + 1;
    stop = n₀ + ii * n;
    X₀[:, start:stop] = x;

    # Evaluate target and prior at _new_ sample points
    dLogTar[start:stop] = dTarget(X₀[:, start:stop]; Log = true)[:];
    dLogPrior[start:stop] = dPrior(X₀[:, start:stop]; Log = true)[:];

    # If new component added, evaluate _new_ mixture component at _all_ sample points
    if add

      dImpStore[1:stop, nmix] = dmvt(X₀[:, 1:stop], μ, Σ, df);

    end

    if ii > 1

      oldInd =  1:(nmix-add);

      # Evaluate _old_ mixture components at _new_ sample points
      dImpStore[start:stop, oldInd] = dGausMix(X₀[:, start:stop], μ₀[:, oldInd], Σ₀[:, :, oldInd]; df = df)[2];

      # Accumulate densities of _new_ samples wrt _old_ mixture components ...
      tmpw = wmix[oldInd];
      if(!add) tmpw[neigh] -= 1; end
      dImpAcc[start:stop] += ( dImpStore[start:stop, oldInd] * tmpw );

    end

    # ... and add densities of _all_ samples wrt _new_ or _recycled_ mixture component
    dImpAcc[1:stop] += dImpStore[1:stop, ifelse(add, nmix, neigh)];

    # Calculate importance density: weighted average of prior and mixture (n₀ * Prior + ii * n * sum(Mix) / ii) / (n₀ + n * ii)
    dimp = (n₀ * exp(dLogPrior[1:stop]) + n * dImpAcc[1:stop]) / (n₀ + n * ii);

    # Checks
    # @printf("%f ", sum(abs(dImpAcc[1:stop] / ii - dImpStore[1:stop, 1:nmix]*(wmix[1:nmix]/sum(wmix[1:nmix])))) );
    # @printf("%f ", sum(abs( dimp - (n₀ * exp(dLogPrior[1:stop]) + (n * ii) * dImpStore[1:stop, 1:nmix]*(wmix[1:nmix]/sum(wmix[1:nmix]))) / (n₀ + n * ii))))
    # dimpCheck = dGausMix(X₀[:, 1:stop], μ₀[:, 1:(stop/n)], Σ₀[:, :, 1:(stop/n)]; df = df)[1][:];
    # @printf("%f \n" , mean(abs(dimp - dimpCheck)));

    # Calculate importance weights
    logw = dLogTar[1:stop] .- log(dimp);

    # Find sample with maximum weight
    μ = X₀[:, findfirst( logw .== maximum(logw) )];

    # (Optionally) Truncate importance weights
    if trunc

      logw = min(logw, log(length(w))/2);

    end

    # Calculate normalized weigths using sumExpTrick.
    tmp = maximum( logw );
    wn = exp(logw - tmp) / sum( exp(logw - tmp) );

    dimMix[ii] = nmix;

    ESS[ii+1] = ( 1 / sum( wn.^2 ) ) / stop;

  end  ##### END main loop

  # List of internal controls, useful to tuneIMIS, which the user probably will not touch
  control = Dict{Any,Any}("score" => score, "hessian" => hessian, "t₀" => t₀,
                          "Q" => Q, "df" => df, "trunc" => trunc,
                          "n₀" => n₀, "n" => n, "niter" => niter, "targetESS" => targetESS)

  # Dropping oversized containers
  output = Dict{Any,Any}("ESS" => ESS,
                         "X₀" => X₀,
                         "μ₀" => μ₀[:, 1:nmix], "Σ₀" => Σ₀[:, :, 1:nmix], "μOrig" => μOrig[:, 1:nmix],
                         "w" => w, "wmix" => wmix[1:nmix] / niter,
                         "dLogTar" => dLogTar, "dLogPrior" => dLogPrior,
                         "dimMix" => dimMix,
                         "control" => control)


  return output

end



















function IMIS2(niter, n, n₀, dTarget, dPrior, rPrior;
              df = 3, trunc = true, quant = 0.01, useLangevin = true, verbose = true,
              targetESS = 1-1e-3, t₀ = nothing, Q = nothing, score = nothing, hessian = nothing,
              maxMix = Inf, B = nothing
              )

  # Safety checks
  if useLangevin && (t₀ == nothing || score == nothing || hessian == nothing)

    error("If useLangevin == true you have to specify ϵ, lag, Q, score and hessian.")

  end

  if B == nothing   B = n;   end

  # Sample from prior
  x = rPrior( n₀ );
  d = size(x, 1);

  # Storage for mixture means and covariance matrices
  μ₀ = zeros(d, niter);
  μOrig = zeros(d, niter);

  Σ₀ = zeros(d, d, niter);

  # Storage for simulated variables
  totSim = n₀ + n * niter;
  X₀ = zeros(d, totSim);
  X₀[:, 1:n₀] = x;

  # Storage for all samples and their density wrt all mixture components
  dImpNewMix = zeros(totSim);
  dImpOldMix = zeros(n, niter);
  dImpAcc = zeros( totSim );

  # Storage for log-target and log-prior
  dLogTar = zeros( totSim );
  dLogPrior = zeros( totSim );

  # Evaluate target and prior
  dLogTar[1:n₀] = dTarget(x; Log = true)[:];
  dLogPrior[1:n₀] = dPrior(x; Log = true)[:];

  # Mixture weights and number of mixture components
  wmix = ones(niter);
  dimMix = zeros(niter);
  nmix = 0;
  neigh = NaN;

  # Calculate importance weights
  logw = dLogTar[1:n₀] .- dLogPrior[1:n₀];

  # Find sample with maximum weight and indicate that it must be added to the mixture
  μ = X₀[:, findfirst( logw .== maximum(logw) )];
  add = true;

  # (Optionally) Truncate importance weights
  if trunc

    logw = min(logw, log(length(w))/2);

  end

  # Calculate normalized weigths using sumExpTrick.
  tmp = maximum( logw );
  wn = exp(logw - tmp) / sum( exp(logw - tmp) );

  ESS = zeros( niter + 1 );
  ESS[1] = ( 1 / sum( wn.^2 ) ) / n₀;

  for ii = 1:niter ##### START main loop

    if verbose @printf("%d \n", ii); end

    # If the maximum number of mixture has been reached, set quant to almost 1. We don't set it to
    # 1 because we still want to add mixture components that are very different from the old ones.
    # If you set quant=1. then the ESS might decrease when maxMix is reached
    if nmix > maxMix   quant = 0.99   end

    # Check if μ is very close to an existing mixture component
    if quant > 0. && ii > 1

      dist = map(kk_ -> maha(μ, μ₀[:, kk_], Σ₀[:, :, kk_])[1], 1:nmix);
      add = minimum(dist) > quantile(Chisq(d), quant);

    end

    if add

      # Save it before it gets modified
      μ_saved = copy(μ);

      # Estimate mean and covariance of new mixture components using...
      if useLangevin # ... linearized Langevin or ...

        μ, Σ = createLangMix(μ, score, hessian, t₀; targetESS = targetESS);
        Σ = Σ[:, :, 1];

      else  # ... n nearest neighbours

       # Raftery in step 2.a of the IMIS paper says to use tmpw = 0.5*(wn + 1/length(wn))
       # but this does not quite work in high dimensions.
        tmpw = wn.*0 + 1/length(wn);
        X_old = X₀[:, 1:(n₀ + n*(ii-1))];
        Σ = nnCov(μ, tmpw, X_old, cov(X_old'), min(B, n₀ + n*(ii-1)))[:, :, 1];

      end

      # Second check on the propagated mixture component, to check if it's actually worth adding it
      if quant > 0. && ii > 1

        #dist = map(kk_ -> - expESS(μ[:], μ₀[:, kk_][:], Σ, Σ₀[:, :, kk_]), 1:nmix);
        #add = maximum( abs(dist) ) < 0.5;

        dist = map(kk_ -> maha(μ, μ₀[:, kk_], Σ₀[:, :, kk_])[1], 1:nmix);
        add = minimum(dist) > quantile(Chisq(d), quant);

      end

      if(add)

        # The mixture has grown by one component
        nmix += 1;

        # Store the unmodified μ
        μOrig[:, nmix] = μ_saved;

        # Add new component to the mixture
        μ₀[:, nmix] = μ;
        Σ₀[:, :, nmix] = Σ;

      end

    end

    if !add

      neigh = findfirst( dist .== minimum(dist) );

      μ =  μ₀[:, neigh];
      Σ = Σ₀[:, :, neigh];

      wmix[neigh] += 1;

    end

    # Simulate from new mixture component and add samples to storage
    x = rmvt(n, μ, Σ, df);

    start = n₀ + (ii-1) * n + 1;
    stop = n₀ + ii * n;
    X₀[:, start:stop] = x;

    # Evaluate target and prior at _new_ sample points
    dLogTar[start:stop] = dTarget(X₀[:, start:stop]; Log = true)[:];
    dLogPrior[start:stop] = dPrior(X₀[:, start:stop]; Log = true)[:];

    if ii > 1

      oldInd =  1:(nmix-add);

      # Evaluate _old_ mixture components at _new_ sample points
      dImpOldMix[:, oldInd] = dGausMix(X₀[:, start:stop], μ₀[:, oldInd], Σ₀[:, :, oldInd]; df = df)[2];

      # Accumulate densities of _new_ samples wrt _old_ mixture components ...
      tmpw = wmix[oldInd];
      if(!add) tmpw[neigh] -= 1; end
      dImpAcc[start:stop] += ( dImpOldMix[:, oldInd] * tmpw );

    end

    # If new component added, evaluate _new_ mixture component at _all_ sample points ...
    # ... and add densities of _all_ samples wrt _new_ or _recycled_ mixture component
    if add

      dImpAcc[1:stop] += dmvt(X₀[:, 1:stop], μ, Σ, df);

    else

      dImpAcc[1:stop] += dmvt(X₀[:, 1:stop], μ₀[:, neigh], Σ₀[:, :, neigh], df);

    end

    # Calculate importance density: weighted average of prior and mixture (n₀ * Prior + ii * n * sum(Mix) / ii) / (n₀ + n * ii)
    dimp = (n₀ * exp(dLogPrior[1:stop]) + n * dImpAcc[1:stop]) / (n₀ + n * ii);

    # Checks
    # @printf("%f ", sum(abs(dImpAcc[1:stop] / ii - dImpStore[1:stop, 1:nmix]*(wmix[1:nmix]/sum(wmix[1:nmix])))) );
    # @printf("%f ", sum(abs( dimp - (n₀ * exp(dLogPrior[1:stop]) + (n * ii) * dImpStore[1:stop, 1:nmix]*(wmix[1:nmix]/sum(wmix[1:nmix]))) / (n₀ + n * ii))))
    # dimpCheck = dGausMix(X₀[:, 1:stop], μ₀[:, 1:(stop/n)], Σ₀[:, :, 1:(stop/n)]; df = df)[1][:];
    # @printf("%f \n" , mean(abs(dimp - dimpCheck)));

    # Calculate importance weights
    logw = dLogTar[1:stop] .- log(dimp);

    # Find sample with maximum weight
    μ = X₀[:, findfirst( logw .== maximum(logw) )];

    # (Optionally) Truncate importance weights
    if trunc

      logw = min(logw, log(length(w))/2);

    end

    # Calculate normalized weigths using sumExpTrick.
    tmp = maximum( logw );
    wn = exp(logw - tmp) / sum( exp(logw - tmp) );

    dimMix[ii] = nmix;

    ESS[ii+1] = ( 1 / sum( wn.^2 ) ) / stop;

  end  ##### END main loop

  # List of internal controls, useful to tuneIMIS, which the user probably will not touch
  control = Dict{Any,Any}("score" => score, "hessian" => hessian,
                          "t₀" => t₀, "Q" => Q, "df" => df, "trunc" => trunc,
                          "n₀" => n₀, "n" => n, "niter" => niter, "targetESS" => targetESS)

  # Dropping oversized containers
  output = Dict{Any,Any}("ESS" => ESS,
                         "X₀" => X₀,
                         "μ₀" => μ₀[:, 1:nmix], "Σ₀" => Σ₀[:, :, 1:nmix], "μOrig" => μOrig[:, 1:nmix],
                         "w" => w, "wmix" => wmix[1:nmix] / niter,
                         "dLogTar" => dLogTar, "dLogPrior" => dLogPrior,
                         "dimMix" => dimMix,
                         "control" => control)


  return output

end
