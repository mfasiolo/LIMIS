d = 2;
n = 200;

# Simulate parameters, covariates and data
θ = rand(Normal(0., 1.), d);

XL = rand(MvNormal(zeros(d), eye(d)), n)';

XXᵗ = map(ii -> XL[ii, :]'* XL[ii, :], 1:n);

# This should be  exp(X*θ) ./ ( 1 + exp(X*θ) )
# but for some reason I need to flip the sign of θ (PROBLEM)
p = exp(-XL*θ) ./ ( 1 + exp(-XL*θ) );

y = map(p_ -> rand(Bernoulli(p_), 1)[1], p);

# Get scatter matrix of X
Σₓ = reduce(+, map(ii -> XL[ii, :]'*XL[ii, :], 1:n) ) / n;

# Complexity penalty
λ = 0.;

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

    out[zz] = ( -y'*XL*θᶻ - sum( log(1 + exp(-XL*θᶻ)) ) - 0.5*λ*(θᶻ'*Σₓ*θᶻ) )[1];

  end

  if( !Log ) out = exp( out ) end

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

    out[:, zz] = -XL'*y + sum( broadcast(/, XL, 1. + exp(XL*θᶻ)), 1)[:] - λ*Σₓ*θᶻ;

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

# Checks
score(θ)

hessian(θ)
#fdHessian(θ, score)


tmpθ = zeros(d);
score(tmpθ)[1] - (dTarget(tmpθ + 1e-6*[1.; zeros(d-1)]; Log = true) - dTarget(tmpθ; Log = true)) / 1e-6
score(tmpθ)[2] - (dTarget(tmpθ + 1e-6*[zeros(d-1), 1]; Log = true) - dTarget(tmpθ; Log = true)) / 1e-6

# optimize(par -> -dTarget(par; Log = true)[1],
#          # g!,
#          zeros(d), # θ - 1,
#          method = :nelder_mead)

# tmp = map(x_ -> dTarget([x_; θ[2]]; Log = true)[1], -2.:0.1:2.)[:];

#plot(-2.:0.1:2., tmp)


########## Prior
μ_P = optimize(par -> -dTarget(par; Log = true)[1], θ).minimum
Σ_P = -2 * inv( hessian(μ_P) )

# Prior density
function dPrior(x_; Log = false)

  out = logpdf(MvNormal(μ_P, Σ_P), x_);

  if( !Log ) out = exp( out ); end

  return( out );

end

# Prior Generator
rPrior(n_) = rand(MvNormal(μ_P, Σ_P), n_);




########################################
####### Plotting it
########################################

if d == 2

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
  scatter(θ[1], θ[2]);

end
