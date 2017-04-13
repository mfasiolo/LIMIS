###########################################################################################
## Mixture of Bananas Example
###########################################################################################

dbg = true;
if d > 10 dbg = false; end

###############
## Banana function
###############

# Simulator
function rBanana(n, a, b, shi1, shi2)

  out = reshape(rand(Normal(0., 1.), banDim*n), n, banDim);

  out[:, 1] = out[:, 1] * a;

  out[:, 2] .-= b * (out[:, 1].^2 - a^2);

  out[:, 1] .+=  shi1;
  out[:, 2] .+=  shi2;

  return( out );

end

# Target function
function dBanana(x, a, b, shi1, shi2; Log = false)

  if ( ndims(x) < 2 ) x = x''; end

  d, n = size(x);

  if d != banDim error("d != banDim") end

  out = logpdf(Normal(shi1, a), x[1, :]);
  out +=  logpdf(Normal(shi2, 1.), x[2, :] + b * ( (x[1, :]-shi1).^2. - a.^2.));

  if d > 2
    for kk = 3:d

      out .+= logpdf(Normal(0., 1.), x[kk, :]);

    end
  end

  if( !Log ) out = exp( out ) end

  return( out[:] );

end

# Vectorized version of gradient
#function banScore(x, a, b, shi1, shi2)

#  if ( ndims(x) < 2 ) x = x''; end

#  d, n = size( x );

#  if d != banDim error("d != banDim") end

#  sx1 = (x[1, :] - shi1)[:];
#  sx2 = (x[2, :] - shi2)[:];

#  out = - x;

#  out[1, :] = -sx1/a^2 - 2*b*sx1 .* (sx2 + b*(sx1.^2 - a^2));

#  out[2, :] = -sx2 - b*(sx1.^2 - a^2);

#  return( out )

#end

# Derivative of the log-likelihood function w.r.t. x
function banScore(x, a, b, shi1, shi2)

  sx1 = (x[1] - shi1);
  sx2 = (x[2] - shi2);

  out = - x;

  out[1] = -sx1/a^2 - 2*b*sx1 .* (sx2 + b*(sx1.^2 - a^2));
  out[2] = -sx2 - b*(sx1.^2 - a^2);

  return( out )

end

if dbg
  # Testing gradient with finite differences
  tmp = rBanana(1, 6, 0.03, 1, 2)[:];
  fdGrad = zeros(banDim);
  for ii = 1:banDim
    x1 = copy(tmp);
    x2 = copy(tmp);
    x1[ii] -= 1e-6;
    x2[ii] += 1e-6;
    fdGrad[ii] = ( dBanana(x2, 6, 0.03, 1, 2; Log = true)[1] - dBanana(x1, 6, 0.03, 1, 2; Log = true)[1] ) ./ (2*1e-6);
  end

  hcat(banScore(tmp, 6, 0.03, 1, 2), fdGrad)
end

# Derivative of the state transition pdf w.r.t. x
function banHessContructor(d)

  out = - eye(d);

  function banHess(x, a, b, shi1, shi2)

    sx1 = x[1] - shi1;
    sx2 = x[2] - shi2;

    out[1, 1] = - 1/a^2 - 2*b*( sx2 + b*(sx1^2 - a^2) ) - 4*b^2*sx1^2;

    out[1, 2] = out[2, 1] = -2 * b * sx1;

    return( out )

  end

end

banHess = banHessContructor(d)

nreps = 10000;
x =  rBanana(nreps, 0.1, 0.1, 1, 2);
@time AHess = map(ii -> banScore(x[ii, :][:], 0.1, 0.1, 1, 2), 1:1:nreps);
@time AHess = map(ii -> banHess(x[ii, :][:], 0.1, 0.1, 1, 2), 1:1:nreps);
#0.036715 seconds (63.26 k allocations: 3.856 MB)

#######################
# Mixture functions
#######################

# Simulator
function rBanMix(n)

  nmix = length( bananicity );

  m = floor( n * bananaW );
  m[1] += n - sum(m);
  m = round(Int, m);

  out = reduce(vcat, map(ii -> rBanana(m[ii], sigmaBan[ii], bananicity[ii], banShiftX[ii], banShiftY[ii]), 1:1:nmix) );

  return( out );

end

# Density function
function dTarget(x; Log = false)

  nmix = length( bananicity );

  out = reduce(hcat, map(ii -> dBanana(x, sigmaBan[ii], bananicity[ii],
  banShiftX[ii], banShiftY[ii]; Log=true), 1:1:nmix) );

  # Weighted sum exp trick
  z = maximum(out, 2)[:];
  out = z + log( exp(broadcast(-, out, z)) * bananaW );

  if( !Log ) out = exp( out ) end

  return( out );

end



# Score of Banana mixture
function scoreGenerator(d, nmix)

  logpᵢ = Array(Float64, nmix);
  u = Array(Float64, nmix);
  lbw = log(bananaW);

  function helper987(x1, x2, u, a, b, shi1, shi2)

    o1 = 0.;
    o2 = 0.;

    nmix = length(u);

    for ii = 1:1:nmix
      sx1 = (x1 - shi1[ii]);
      sx2 = (x2 - shi2[ii]);
      o1 += u[ii]*(-sx1/a[ii]^2 - 2*b[ii]*sx1 .* (sx2 + b[ii]*(sx1.^2 - a[ii]^2)));
      o2 += u[ii]*(-sx2 - b[ii]*(sx1.^2 - a[ii]^2));
    end

    return( [o1; o2] )

  end

 function fun(x; Log = false)

   # Densities and gradients, of each mixture density and for each column of x
   for ii in 1:nmix
     logpᵢ[ii] = dBanana(x, sigmaBan[ii], bananicity[ii], banShiftX[ii], banShiftY[ii]; Log=true)[1];
   end

   # Weighted sum exp trick
   z = maximum(logpᵢ);
   logΣwᵢpᵢ = z + log( dot(exp(logpᵢ - z), bananaW) );
   u .= exp( lbw + logpᵢ - logΣwᵢpᵢ );

   out = -x;
   out[1:2] = helper987(x[1], x[2], u, sigmaBan, bananicity, banShiftX, banShiftY);

   return( out );

  end

end

score = scoreGenerator(d, length(bananaW));

if dbg
  # Testing gradient with finite differences
  nreps = 1000;
  x = rBanMix( nreps );
  fdGrad = zeros(d, nreps);
  for kk = 1:(nreps)
    for ii = 1:d
      x1 = copy(x[kk, :][:]);
      x2 = copy(x[kk, :][:]);
      x1[ii] -= 1e-6;
      x2[ii] += 1e-6;
      fdGrad[ii, kk] = ( dTarget(x2; Log = true)[1] - dTarget(x1; Log = true)[1] ) ./ (2*1e-6);
    end
  end

  AGrad = reduce(hcat, map(ii -> score(x[ii, :][:]), 1:1:nreps))
  tmp = maximum( abs( AGrad - fdGrad ) ./ abs(fdGrad), 2 )
  if( maximum(tmp) > 0.01 ) error("score() disagrees with finite differences ") end
end

# Hessian of Banana mixture
function hessianConstructor(d, nmix)

  logpᵢ = Array(Float64, nmix);
  u = Array(Float64, nmix);
  gr = Array(Float64, d, nmix);
  grTot = Array(Float64, d);
  out = Array(Float64, d, d);

  function fun(x; Log = false)

    if ( ndims(x) < 2 ) x = x''; end
    d = size(x)[1];

    if d != banDim error("d != banDim") end

    nmix = length( bananicity );

    for ii in 1:nmix
      logpᵢ[ii] = dBanana(x, sigmaBan[ii], bananicity[ii], banShiftX[ii], banShiftY[ii]; Log=true)[1];
    end

    # Weighted sum exp trick
    z = maximum(logpᵢ);
    logΣwᵢpᵢ = z + log( dot(exp(logpᵢ - z), bananaW) );
    u .= exp( log(bananaW) + logpᵢ - logΣwᵢpᵢ );

    for ii = 1:nmix
      gr[:, ii] = banScore(x, sigmaBan[ii], bananicity[ii], banShiftX[ii], banShiftY[ii]);
    end

    # ∇²logp(x) = ∑uᵢ ( ∇²logpᵢ(x) + ∇logpᵢ(x)∇logpᵢ(x)ᵀ ) + ...
    # out = gr*(gr'.*u);
    A_mul_B!(out, gr, gr'.*u);
    for ii = 1:nmix
      out += u[ii] * ( banHess(x, sigmaBan[ii], bananicity[ii], banShiftX[ii], banShiftY[ii]) );
    end

    # + ∇logp(x) ∇logp(x)^T + ...
    grTot .= gr*u
    out -= grTot * grTot';

    out = (out+out')./2 # Symmetrize

    return( copy(out) );

  end

end

hessian = hessianConstructor(d, length(bananaW))

nreps = 10000;
x =  rBanMix( 10000 );
@time AHess = map(ii -> score(x[ii, :][:]), 1:1:nreps);
@time AHess = map(ii -> hessian(x[ii, :][:]), 1:1:nreps);
# 0.181034 seconds (3.78 M allocations: 190.755 MB, 16.14% gc time)
# 0.286289 seconds (4.17 M allocations: 254.990 MB, 17.91% gc time)

if dbg
  nreps = 1000;
  x =  rBanMix( 1000 );
  DHess = map(ii -> fdHessian(x[ii, :][:], score; h = 1e-6), 1:1:nreps);
  AHess = map(ii -> hessian(x[ii, :][:]), 1:1:nreps);

  tmp = map(ii -> maximum( abs(DHess[ii] - AHess[ii]) ), 1:1:nreps);
  if( maximum(tmp) > 0.00001 ) error("hessian() disagrees with finite differences ") end
end

###############
##### Prior functions
###############

# Prior density
function dPrior(x_; Log = false)

  out = dmvt(x_, μ_P, Σ_P, 3; Log = Log);

  if( !Log ) out = exp( out ); end

  return( out );

end

# Prior Generator
rPrior(n_) = rmvt(n_, μ_P, Σ_P, 3);

#################
### Contour plots
#################

## Reference distributions
# Set parameters for the reference probability distributions
L = 100;
x1 = linspace(-20., 20., L);
x2 = linspace(-15., 15., L);

# Define target parameters
d_lik = eye(L);

if( banDim == 2)

  for iRow = 1:L
    for iCol = 1:L
      #@printf("%d", iCol);
      d_lik[iRow, iCol] = dTarget([x1[iCol] x2[iRow]]')[1];
    end;
  end;

  contour(x1, x2, d_lik);

end

# Define previous posterior - Gaussian
μ_P  = zeros(banDim);
Σ_P = 100 * eye(banDim);

d_prior = eye(L);

if( banDim == 2)

  for iRow = 1:L
    for iCol = 1:L
      d_prior[iRow, iCol] = pdf(MvNormal(μ_P, Σ_P), [x1[iCol], x2[iRow]]);
    end;
  end;

  contour(x1, x2, d_prior);

end
