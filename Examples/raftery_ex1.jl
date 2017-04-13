
### Raftery example 1

# Parameters of priors and likelihood
αP = [6., 0.5, 5.5, 0.15, 3, 0.6];
βP = [1.3, 0.14, 0.289, 0.029, 0.04, 0.1];
μM = [7, 0.0525, 2, 4];
σM = [0.5, 0.00144, 0.01, 0.01];

##############

# Target function
function dTarget(θ; Log = false)

  if ( ndims(θ) < 2 ) θ = θ'' end

  (d, n) = size(θ);

  ϕ = zeros(4);
  out = zeros(n);

  for ff = 1:n
    θz = θ[:, ff];
    ϕ[1] = prod(θz);
    ϕ[2] = θz[2]*θz[4];
    ϕ[3] = θz[1]/θz[5];
    ϕ[4] = θz[3]*θz[6];

    tmp = 0.0;
    for ii=1:4
      tmp += logpdf(Normal(μM[ii], σM[ii]), ϕ[ii]);
    end

    for ii in 1:6
      tmp += logpdf(Normal(αP[ii], βP[ii]), θz[ii]);
    end

    out[ff] = tmp;

  end

  if( !Log ) out = exp( out ) end

  #@printf("%f ", out[1]);

  return( out );

end

# Score (gradient) of log-target
function scoreGenerator()

  dp = Array(Float64, 6);
  dl = Array(Float64, 4);
  ii = [3, 2, 4, 2, 3, 4];

 function fun(θ)
   pT = prod(θ);
   dp .= (αP-θ)./(βP.^2);
   dl .= (μM-[pT, θ[2]*θ[4], θ[1]/θ[5], θ[3]*θ[6]])./(σM.^2);

   out = copy( dp );

   out += dl[1]*pT./θ + dl[ii].*[1./θ[5]; θ[4]; θ[6]; θ[2]; -θ[1]/(θ[5]^2); θ[3]];

   return( out );
 end

end

# Hessian of log-target
function hessianGenerator()

  dph = zeros(4, 6);
  d2ph = zeros(6, 6, 4);
  ll2 = [3, 2, 4, 2, 3, 4];
  dp = Array(Float64, 6);
  d2p = Array(Float64, 6);
  dl = Array(Float64, 4);
  d2l = Array(Float64, 4);

 function fun(θ)

   #θ = θ[:];

   pT = prod(θ);
   dp .= (αP-θ)./(βP.^2);
   d2p .= -1./(βP.^2);
   dl .= (μM-[pT, θ[2]*θ[4], θ[1]/θ[5], θ[3]*θ[6]])./(σM.^2);
   d2l .= -1./(σM.^2);

   dph[1, :] = pT ./ θ';
   dph[2, 2] = θ[4];
   dph[2, 4] = θ[2];
   dph[3, 1] = 1./θ[5];
   dph[3, 5] = -θ[1]/(θ[5]^2);
   dph[4, 3] = θ[6];
   dph[4, 6] = θ[3];

   d2ph[:, :, 1] = pT ./ (θ*θ');
   for ii = 1:6    d2ph[ii, ii, 1] = 0.;   end
   d2ph[2, 4, 2] = d2ph[4, 2, 2] = 1.;
   d2ph[5, 5, 3] = 2*θ[1]/θ[5]^3;
   d2ph[1, 5, 3] = d2ph[5, 1, 3] = -1/θ[5]^2;
   d2ph[3, 6, 4] = d2ph[6, 3, 4] = 1.;

   out = Array(Float64, 6, 6);
   for ir = 1:6
     for ic = ir:6
       kk = ll2[ir];
       out[ir, ic] = out[ic, ir] = d2l[1]*dph[1,ir]*dph[1,ic] + dl[1]*d2ph[ir, ic, 1] +
                     d2l[kk]*dph[kk,ir]*dph[kk,ic] + dl[kk]*d2ph[ic, ir, kk] +
                     ifelse(ir==ic, d2p[ir], 0.);
     end
   end

   return( out );
  end

end

score = scoreGenerator()
hessian = hessianGenerator()

nreps = 10000;
x = rmvt(nreps, [6, -0.5, 5.5, -0.1, 3, 0.7], eye(6) );
@time AHess = map(ii -> score(x[:, ii][:]), 1:1:nreps);
@time AHess = map(ii -> hessian(x[:, ii][:]), 1:1:nreps);
#  0.057845 seconds (462.74 k allocations: 25.804 MB, 19.17% gc time)
#  0.072038 seconds (432.67 k allocations: 36.331 MB, 11.89% gc time)


#######
# Tests
#######
dbg = true;

# Testing gradient with finite differences
if dbg
  nreps = 100;
  x = rmvt(nreps, [6, -0.5, 5.5, -0.1, 3, 0.7], eye(6) );
  fdGrad = zeros(6, nreps);
  for kk = 1:(nreps)
    for ii = 1:6
      x1 = copy(x[:, kk][:]);
      x2 = copy(x[:, kk][:]);
      x1[ii] -= 1e-6;
      x2[ii] += 1e-6;
      fdGrad[ii, kk] = ( dTarget(x2; Log = true)[1] - dTarget(x1; Log = true)[1] ) ./ (2*1e-6);
    end
  end

  AGrad = reduce(hcat, map(ii -> score(x[:, ii][:]), 1:1:nreps))
  tmp = maximum( abs( AGrad - fdGrad ) ./ abs(fdGrad), 2 )
  if( maximum(tmp) > 0.01 ) error("score() disagrees with finite differences ") end
end

# Testing hessian with finite differences
if dbg
  nreps = 500;
  x = rmvt(nreps, [6, -0.5, 5.5, -0.1, 3, 0.7], 0.1*eye(6) );
  DHess = map(ii -> fdHessian(x[:, ii][:], score; h = 1e-6), 1:1:nreps);
  AHess = map(ii -> hessian(x[:, ii][:]), 1:1:nreps);
  tmp = map(ii -> maximum( abs(DHess[ii] - AHess[ii])./(abs(AHess[ii])+1e-5) ), 1:1:nreps);
  if( maximum(tmp) > 0.1 ) error("hessian() disagrees with finite differences ") end
end


########## Prior
μ_P = copy(αP);
Σ_P = eye(6);
for ii = 1:6    Σ_P[ii, ii] = βP[ii]^2;    end

# Prior density
function dPrior(x_; Log = false)

  out = dmvt(x_, μ_P, Σ_P, 3; Log = Log);

  if( !Log ) out = exp( out ); end

  return( out );

end

# Prior Generator
rPrior(n_) = rmvt(n_, μ_P, Σ_P, 3);


############ Importance sampler


########## Prior
μ_I = optimize(par -> -dTarget(par; Log = true)[1], μ_P, LBFGS()).minimizer
Σ_I = - 2 * inv( hessian(μ_I) )
Σ_I = (Σ_I .+ Σ_I') ./ 2.;

# Prior density
function dImp(x_; Log = false)

  out = dmvt(x_, μ_I, Σ_I, 3; Log = Log);

  if( !Log ) out = exp( out ); end

  return( out );

end

# Prior Generator
rImp(n_) = rmvt(n_, μ_I, Σ_I, 3);
