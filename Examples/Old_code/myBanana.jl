###########
## Banana Density Example
###########

####### My Banana
d = 2
bananicity = 1.5;
tilt = 0;

# Simulator
function rBanana(n)

  x = rand(Normal(0., 1.), n);
  y = rand(Normal(0., 1.), n);

  out = hcat(x, y .+ bananicity .* (x - tilt).^2);

  if d > 2

    out = hcat( out, reshape(rand(Normal(0., 1.), n*(d-2)), n, d-2) );

  end

  return( out );

end

# Prior density
function dPrior(x_; Log = false)

  out = logpdf(MvNormal(μ_P, Σ_P), x_);

  if( !Log ) out = exp( out ); end

  return( out );

end

# Prior Generator
rPrior(n_) = rand(MvNormal(μ_P, Σ_P), n_);

# Target function
function dTarget(x; Log = false)

  if ( ndims(x) < 2 ) x = x''; end

  b = bananicity;
  a = tilt;

  d, n = size(x);

  out = logpdf(Normal(0., 1.), x[1, :]);

  tmp = b * (x[1, :]-a).^2.
  for kk = 2:d

  out .+= logpdf(Normal(0., 1.), x[kk, :] - tmp);

  end

  if( !Log ) out = exp( out ) end

  return( out[:] );

end

# Derivative of the log-likelihood function w.r.t. x
function score(x)

  if ( ndims(x) < 2 ) x = x''; end

  b = bananicity;
  a = tilt;

  d, n = size( x );

  out = zeros(d, n);

  for ii = 1:n

   out[1, ii] = -x[1, ii] + 2. * b * (x[1, ii]-a) * sum( x[2:d, ii] - b * (x[1, ii]-a)^2. );

   out[2:d, ii] = b * (x[1, ii]-a) ^ 2. - x[2:d, ii];

  end

  return( out )

end

# Derivative of the state transition pdf w.r.t. x
function hessian(x_)

  b = bananicity;
  a = tilt;

  d = size(x_)[1];

  out = - eye(d);

  out[1, 1] = -1 + 2. * b * sum( x_[2:d] - b * (x_[1]-a)^2. ) - 4*b^2.*(x_[1]-a)^2*(d-1);

  out[1, 2:d] = out[2:d, 1] = 2. * b * (x_[1]-a);

  return( out )

end

## Reference distributions
# Set parameters for the reference probability distributions
L = 100;
x1 = linspace(-4., 4., L);
x2 = linspace(-5., 10., L);

# Define target parameters
d_lik = eye(L);

if( d == 2)

 for iRow = 1:L
   for iCol = 1:L
     #@printf("%d", iCol);
     d_lik[iRow, iCol] = dTarget([x1[iCol] x2[iRow]]')[1];
  end;
 end;

 contour(x1, x2, d_lik);

end

# Define previous posterior - Gaussian
μ_P  = zeros(d);
Σ_P = 10 * eye(d);
d_prior = eye(L);

if( d == 2)

 for iRow = 1:L
   for iCol = 1:L
     d_prior[iRow, iCol] = pdf(MvNormal(μ_P, Σ_P), [x1[iCol], x2[iRow]]);
  end;
 end;

 contour(x1, x2, d_prior);

end






