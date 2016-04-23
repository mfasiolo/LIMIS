###########
## Flavio's Example
###########

####
# Define functions
####

# Output/measurement function
H_func(x_) = [norm(x_, 2), atan2(x_[2], x_[1])];

# Derivative of output function w.r.t. the state x
DH_func(x_) = [ [x_[1]/norm(x_,2)  x_[2]/norm(x_,2)],
                [-x_[2]/(dot(x_, x_))  x_[1]/(dot(x_, x_))] ];

# Derivative of target w.r.t. x
function score(x)

  if ( ndims(x) < 2 ) x = x'' end

  (d, n) = size(x);

  out = zeros(d, n);

  for ii = 1:n

   z = x[:, ii];

   out[:, ii] = DH_func(z)' * (R \ (yTarget - H_func(z))) -  z / dot(z, z);

  end

  return out;

end

# Hessian of the target
function hessian(x_)

 x_ = x_[:];

 h = 1e-3;

 out = zeros(2, 2);

 score0 = score(x_)[:];

 out[1, 1] = ( score(x_ + [h; 0.])[1] - score0[1] ) / h;

 out[2, 2] = ( score(x_ + [0.; h])[2] - score0[2] ) / h;

 out[1, 2] = out[2, 1] = ( score(x_ + [0.; h])[1] - score0[1] ) / h;

 return( out );

end

# Prior density
function dPrior(x_, log = false)

  if ( ndims(x_) < 2 ) x_ = x_'' end

  (d, n) = size(x_)

  if( d != size(Σ_P)[1] ) throw( "d != size(Σ_P)[1]" ) end

  out = logpdf(MvNormal(μ_P, Σ_P), x_);

  if( !log ) out = exp( out ); end

  return( out );

end

# Prior Generator
rPrior(n_) = rand(MvNormal(μ_P, Σ_P), n_);

# Target function
function dTarget(x_; Log = false)

 if ( ndims(x_) < 2 ) x_ = x_'' end

 (d, n) = size(x_)

 if( d != size(R)[1] ) throw( "d != size(R)[1]" ) end

 d = size(x_)[1];
 n = size(x_)[2];

 out = zeros(n);

 for zz = 1:n

  polar = H_func(x_[:, zz]);
  out[zz] = logpdf(MvNormal(polar, R), yTarget) - log( polar[1] );

 end

 if( !Log ) out = exp( out ) end

 return( out );

end

## Reference distributions
# Set parameters for the reference probability distributions
N = 2500;                           # grid of 50 x 50
L = 50;                             # number of abscissas / dimension
dx = L/sqrt(N);                     # state-space increment / dimension
d = 2;                              # state dimension
x1 = -L/2:dx:L/2;                   # abscissas (state x = [x1;x2])
x2 = copy(x1);

# Define previous posterior - Gaussian
μ_P  = [10., -10.];
Σ_P = 5 * eye(d);
d_prior = eye(L+1);

for iRow = 1:(L+1)
  for iCol = 1:(L+1)
    d_prior[iRow, iCol] = pdf(MvNormal(μ_P, Σ_P), [x1[iCol], x2[iRow]]);
 end;
end;

contour(x1, x2, d_prior);

# Define Target
# yₜ=[ (x₁² + x₂²)^0.5, atan( x₂ / x₁ )] + νₜ ,  νₜ ∼ N(0, R)
yTarget = [+20; deg2rad(45)];
R = [1 0; 0 0.16];

d_lik = eye(L+1);

for iRow = 1:(L+1)
  for iCol = 1:(L+1)
    d_lik[iRow, iCol] = dTarget([x1[iCol] x2[iRow]]'; Log = false)[1];
 end;
end;

contour(x1, x2, d_lik);

#μ_a = optimize(par_ -> -logpdf(MvNormal(H_func(par_), R), yTarget), [10. ,  10.]).minimum;
#H_a = tHess(μ_a);
#Σ_a = inv( -tHess(μ_a) );


