
########### REAL OR ....
sonar = FileIO.load("sonar.RData")["sonar"];

sonar = convert(Matrix, sonar);

d = 61; #61 maximum

y = sonar[:, end][:];

nobs = length(y);

XL = sonar[:, 1:d];

λ = exp(-2) * nobs; # Chosen with glmnet in R

########### ... Simulated data
# Simulate parameters, covariates and data
#
#d = 5; error("this stuff does not work in parallel")
#nobs = 200;

#θ = rand(d);

#XL = hcat(rep(1, nobs), rand(MvNormal(zeros(d-1), eye(d-1)), nobs)');

#p = exp(XL*θ) ./ ( 1 + exp(XL*θ) );

#y = map(p_ -> rand(Bernoulli(p_), 1)[1], p);

#λ = 1;
############

##### Useful stuff
XXᵗ = map(ii -> XL[ii, :]* XL[ii, :]', 1:nobs);

# Get scatter matrix of X
Σₓ = eye(d); #reduce(+, map(ii -> XL[ii, 2:end]'*XL[ii, 2:end], 1:nobs) ) / nobs;
Σₓ[1, 1] = 0.;

##############

# Target function
function dTarget(θ; Log = false)

  if ( ndims(θ) < 2 ) θ = θ'' end

  (d, n) = size(θ)

  d = size(θ)[1];
  n = size(θ)[2];

  θᶻ = zeros(d);
  out = zeros(n);
  for zz = 1:n

    θᶻ = θ[:, zz];

    out[zz] = ( y'*XL*θᶻ - sum( log(1 + exp(XL*θᶻ)) ) - 0.5*λ*(θᶻ'*Σₓ*θᶻ) )[1];

  end

  if( !Log ) out = exp( out ) end

  #@printf("%f ", out[1]);

  return( out );

end

# Score (gradient) of log-target
function score(θ)

  if ( ndims(θ) < 2 ) θ = θ'' end

  (d, n) = size(θ)

  d = size(θ)[1];
  n = size(θ)[2];
  m = size(XL)[1];

  θᶻ = zeros(d);
  out = zeros(d, n);
  for zz = 1:n

    θᶻ = θ[:, zz];

    out[:, zz] = XL'*y - sum( broadcast(/, XL, 1. + exp(-XL*θᶻ)), 1)[:] - λ*Σₓ*θᶻ;

    #@printf("%f ", sum(abs(out[:, zz] - aaa)));

  end

  return( out );

end

# Hessian of log-target
function hessian(θ)

  m, d = size(XL);

  tmp = exp(XL*θ) ./ (1. + exp(XL*θ)).^2

  out = - λ * Σₓ;

  for ii = 1:m

    out -= tmp[ii] * XXᵗ[ii];

  end

   #@printf("%f ", sum(abs(out - out2)));

  return( out );

end


#######
# Tests
#######
dbg = true;

# Testing gradient with finite differences
if dbg
  nreps = 100;
  x = rmvt(nreps, rep(0, d), eye(d) );
  fdGrad = zeros(d, nreps);
  for kk = 1:(nreps)
    for ii = 1:d
      x1 = copy(x[:, kk][:]);
      x2 = copy(x[:, kk][:]);
      x1[ii] -= 1e-6;
      x2[ii] += 1e-6;
      fdGrad[ii, kk] = ( dTarget(x2; Log = true)[1] - dTarget(x1; Log = true)[1] ) ./ (2*1e-6);
    end
  end

  tmp = maximum( abs( score(x) - fdGrad ) ./ abs(fdGrad), 2 )
  if( maximum(tmp) > 0.01 ) error("score() disagrees with finite differences ") end
end

# Testing gradient with finite differences
if dbg
  nreps = 100;
  x = rmvt(nreps, rep(0, d), eye(d) );
  DHess = map(ii -> fdHessian(x[:, ii][:], score; h = 1e-6), 1:1:nreps);
  AHess = map(ii -> hessian(x[:, ii][:]), 1:1:nreps);

  tmp = map(ii -> maximum( abs(DHess[ii] - AHess[ii]) ), 1:1:nreps)
  if( maximum(tmp) > 0.00001 ) error("hessian() disagrees with finite differences ") end
end


########## Prior
μ_P = optimize(par -> -dTarget(par; Log = true)[1], rep(0., d), LBFGS()).minimum
Σ_P = - 2 * inv( hessian(μ_P) )
Σ_P = (Σ_P .+ Σ_P') ./ 2.;

# Prior density
function dPrior(x_; Log = false)

  out = dmvt(x_, μ_P, Σ_P, 3; Log = Log);

  if( !Log ) out = exp( out ); end

  return( out );

end

# Prior Generator
rPrior(n_) = rmvt(n_, μ_P, Σ_P, 3);



########################################
####### Plotting it
########################################

if false #d == 2

  L = 25;
  x1 = linspace(θ[1]-3., θ[1]+3., L);
  x2 = linspace(θ[2]-3., θ[2]+3., L);

  # Define Target
  # yₜ=[ (x₁² + x₂²)^0.5, atan( x₂ / x₁ )] + νₜ ,  νₜ ∼ N(0, R)

  d_lik = eye(L);

  for iRow = 1:L
    for iCol = 1:L
      d_lik[iRow, iCol] = dTarget([x1[iCol]; x2[iRow]]; Log = false)[1];
    end;
  end;

end

plotTarget = function()

  if(d != 2) throw("d != 2"); end

  contour(x1, x2, d_lik);
  #scatter(θ[1], θ[2]);

end
