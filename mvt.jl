###########
# Simulate multivariate t variables
###########
function rmvt(n, μ, Σ, df = 3)

  if ( ndims(μ) > 1 ) μ = vec( μ ) end

  d = length(μ);

  out = rand(MvNormal(zeros(d), Σ), n);

  if( df != Inf )

    broadcast!(/, out, out, sqrt(rand(Chisq(df), n)/df)');

  end

  # N.B. if (df != Inf) you cannot add μ before dividing by the χ² variables!
  broadcast!(+, out, out, μ);

  return out;

end


##########
# Evaluate density of multivariate t density
##########
# INPUT
# - A: optional storage of dimension size(x)
# - out: optional storage of dimension size(x, 2)
function dmvt(x, μ, Σ, df = 3; Log = false, A = nothing, out = nothing)

  if ( ndims(μ) > 1 ) μ = vec(μ); end

  if ( ndims(x) < 2 ) x = x''; end

  d, n = size(x);

  if (A == nothing) A = Array(Float64, d, n); end
  if (out == nothing) out = Array(Float64, n); end

  # We want the lower triangle
  dec = factorize( chol( Σ )' );

  # A = x - μ
  broadcast!(-, A, x, μ);

  # A = Σ^{-1/2} (x - μ) Forward solve
  A_ldiv_B!(dec, A);

  # out = (x - μ)ᵀ Σ^{-1/2} Σ^{-1/2} (x - μ)
  broadcast!(*, A, A, A);
  out = vec( sum(A, 1) );

  if( df == Inf )

    out = -sum(log(diag(dec))) - 0.5 * d * log(2 * pi) - 0.5 * out;

  else

    out = lgamma((d + df)/2) - (lgamma(df/2) + sum(log(diag(dec)))  +
                                  d/2 * log(pi * df) + 0.5 * (df + d) * log1p(out/df));

  end

  if ( !Log ) out = exp(out); end

  return out;

end

##########
# Evaluate Squared Mahalanobis distance
##########
# Evaluates Squared Mahalanobis distance between each column of x and the centre μ
# INPUT
# - A: optional storage of dimension size(x)
# - out: optional storage of dimension size(x, 2)
function maha(x, μ, Σ; isChol = false, A = nothing, out = nothing)

  if ( ndims(μ) > 1 ) μ = vec(μ); end

  if ( ndims(x) < 2 ) x = x''; end

  d, n = size(x);

  if (A == nothing) A = Array(Float64, d, n); end
  if (out == nothing) out = Array(Float64, n); end

  # We want the lower triangle
  if ( !isChol )

    dec = factorize( chol( Σ )' );

  else

    dec = factorize( Σ );

  end

  # A = x - μ
  broadcast!(-, A, x, μ);

  # A = Σ^{-1/2} (x - μ) Forward solve
  A_ldiv_B!(dec, A);

  # out = (x - μ)ᵀ Σ^{-1/2} Σ^{-1/2} (x - μ)
  broadcast!(*, A, A, A);

  out = vec( sum(A, 1) );

  return out;

end

#####################
## Evaluate Density of Gaussian mixture
#####################
# INPUT
# - w: optional mixture weights. The function does not check whether they add to 1.
# - dTrans: additional storage of dimension [size(x, 2), size(μ, 2)].
# OUTPUT
# - dens: the density of the mixture, evaluated at each column of x
# - dTrans: matrix containing the density of each pair sample-mixture component pair p(xᵢ|μₖ, Σₖ)
function dGausMix(x, μ, Σ; Log = false, df = Inf, w = nothing, dTrans = nothing)

  d, n = size( x );

  # Number of mixture components
  m = size(μ, 2);

  # If the mixture weights are not given, they are assumed to be equal to 1/m
  if (w == nothing) w = ones(m) / m; end

  # Matrix containing density of each mixture-sample pairs
  if (dTrans == nothing) dTrans = Array(Float64, n, m); end

  # Output density
  dens = Array(Float64, n);

  # Pre-allocating storage to be used by dmvt
  storeM = Array(Float64, d, n);

  # Loop over m mixture components μⁱ, Σⁱ
  for ii = 1:m

    dTrans[:, ii] = dmvt(x, μ[:, ii], Σ[:, :, ii], df; A = storeM, out = dens);

  end

  # p(xᵢ) = Σₖ wₖ p(xᵢ|μₖ, Σₖ)
  dens = dTrans * w;

  if( Log ) dens = log(dens); end

  return dens, dTrans

end


####################
### Sample Weighted Gaussian Mixture
####################
function rGausMix(n, μ, Σ; df = Inf, w = nothing, A = nothing, labs = false)

  # Number of mixture components
  d, m = size( μ );

  # If the mixture weights are not given, they are assumed to be equal to 1/m
  if (w == nothing) w = ones(m) / m; end

  # Matrix containing density of each mixture-sample pairs
  if (A == nothing) A = Array(Float64, d, n); end

  # Number of samples from each mixture component
  ns = vec( rand(Multinomial(n, w), 1) );
  ind = (1:m)[ns .> 0];

  start = 1;
  for kk = ind

    A[:, start:(start+ns[kk]-1)] = rmvt(ns[kk], μ[:, kk], Σ[:, :, kk], df);

    start += ns[kk];

  end

  if labs

   return A, [ind ns[ind]];

  else

    return A;

  end

end
