###########
## Banana Density Example
###########

# Simulator
function rBanana(n)

  a = copy( sigmaBan );
  b = copy( bananicity );

  out = reshape(rand(Normal(0., 1.), d*n), n, d);

  out[:, 1] = out[:, 1] * a;

  out[:, 2] .-= b * (out[:, 1].^2 - a^2);

  out[:, 1] .+=  banSh[1];
  out[:, 2] .+=  banSh[2];

  return( out );

end

a = rBanana(1000);
scatter(a[:, 1], a[:, 2])

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

  b = copy(bananicity);
  a = copy(sigmaBan);

  d, n = size(x);

  out = logpdf(Normal(banSh[1], a), x[1, :]);
  out +=  logpdf(Normal(banSh[2], 1.), x[2, :] + b * ( (x[1, :]-banSh[1]).^2. - a.^2.));

  if d > 2
    for kk = 3:d

      out .+= logpdf(Normal(0., 1.), x[kk, :]);

    end
  end

  if( !Log ) out = exp( out ) end

  return( out[:] );

end

# Derivative of the log-likelihood function w.r.t. x
function score(x)

  if ( ndims(x) < 2 ) x = x''; end

  b = copy(bananicity);
  a = copy(sigmaBan);

  d, n = size( x );

  sx1 = x[1, :] - banSh[1];
  sx2 = x[2, :] - banSh[2];

  out = - x;

  out[1, :] = -sx1/a^2 - 2*b*sx1 * (sx2 + b*(sx1.^2 - a^2));

  out[2, :] = -sx2 - b*(sx1.^2 - a^2);

  return( out )

end

# Testing gradient with finite differences
x = rBanana( 1 )[:];
fdGrad = zeros(d);
for ii = 1:d
  x1 = copy(x);
  x2 = copy(x);
  x1[ii] -= 1e-6;
  x2[ii] += 1e-6;
  fdGrad[ii] = ( dTarget(x2; Log = true)[1] - dTarget(x1; Log = true)[1] ) ./ (2*1e-6);
end

hcat(score(x), fdGrad)

# Derivative of the state transition pdf w.r.t. x
function hessian(x)

  b = copy(bananicity);
  a = copy(sigmaBan);

  d = size(x)[1];

  sx1 = x[1] - banSh[1];
  sx2 = x[2] - banSh[2];

  out = - eye(d);

  out[1, 1] = - 1/a^2 - 2*b*( sx2 + b*(sx1^2 - a^2) ) - 4*b^2*sx1^2;

  out[1, 2] = out[2, 1] = -2 * b * sx1;

  return( out )

end

x = rBanana( 1 )[:];
fdHessian(x, score; h = 1e-6)
hessian(x)

## Reference distributions
# Set parameters for the reference probability distributions
L = 100;
x1 = linspace(-20., 20., L);
x2 = linspace(-10., 10., L);

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
Σ_P = 20 * eye(d);
d_prior = eye(L);

if( d == 2)

 for iRow = 1:L
   for iCol = 1:L
     d_prior[iRow, iCol] = pdf(MvNormal(μ_P, Σ_P), [x1[iCol], x2[iRow]]);
  end;
 end;

 contour(x1, x2, d_prior);

end






