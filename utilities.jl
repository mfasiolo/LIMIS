
#################################################
# Repeat n times each element of v
#################################################
rep(v, n) = [v[div(i,n)+1] for i=0:n*length(v)-1]

### Univartiate kernel density estimator

function kde(y, x, h; logw = nothing);

  y = y[:];
  x = x[:];

  ny = length( y );
  nx = length( x );

  if logw == nothing    logw = zeros(nx)    end

  # Using sumExpTrick here to avoid underflow
  mx = maximum( logw );
  logw -= mx;

  out = map(_z -> exp(mx)*mean(exp(logw) .* pdf(Normal(_z, h), x)), y);

  return out;

end


######## A classic....
function sumExpTrick(x)

  z = maximum(x);

  out = sum( exp(x - z) ) * exp(z);

  return out;

end

function meanExpTrick(x) return sumExpTrick(x) / length(x) end

### This is the meanExpTrick, but when a weighted sum is needed.
### I assume that the weights might overflow, not x.
# Each row of x is a d-dimensional vector
function WsumExpTrick(x, logw)

  z = maximum( logw );
  logw -= z;

  out = (x * exp(logw)) * exp(z);

  return out;

end

function WmeanExpTrick(x, logw) return WsumExpTrick(x, logw) / length(logw) end

#################################################
# Root-finding by Bisection
#################################################

function rootBisect(f, l, r; tol = 1e-6, maxit = 100)

  # Evaluate function left and right and store signs
  vl = f(l);
  vr = f(r);
  sl = sign(vl);
  sr = sign(vr);

  if( sl==sr ) error("The root is not bracketed!") end

  # Estimate root position
  c = NaN;

  # Function value at estimated root
  cv = 2. * tol;

  # Main Loop
  kk = 1;
  while(abs(cv) > tol)

    # Bisect and evaluate function at newpoints
    c = (l + r)/2.
    # @printf("c = %f \n", c);
    cv = f(c);
    sc = sign(cv);

    # If sign of new point is the same as on (right) left, new point
    # is the new (right) left.
    if(sc == sl)

      sl = sc;
      l = copy(c);

    else

      sr = sc;
      r = copy(c);

    end

    kk += 1;
    if( kk > maxit ) error("Maximum number of iteration excedeed") end

  end

  return c;

end

#####################################################
# Finite differencing to get an Hessian
#####################################################
function fdHessian(x, score; h = 1e-6)

  x = vec( x );

  d = length(x);

  out = zeros(d, d);

  for ii = 1:d

    x1 = copy(x);
    x2 = copy(x);
    x1[ii] -= h;
    x2[ii] += h;

    out[ii, :] = out[:, ii] =  ( score(x2) - score(x1) ) / (2*h)

  end

  return( out );

end

########################################
######## Propagate μ and Σ using Runge-Kutta and do line-search with step halving
########################################
# Keeps halving the step sizes until the log-target is actually increased
# INPUT
# - μ₀: initial position
# - Σ₀: initial position
# - δt: step size
# - t₀: final time at which the propagation of μ(t) and Σ(t) stops
# - score: function returning the gradient of the log-target
# - hessian: function returning the hessian of the log-target
# - dTarget: function returning the (log) density of the target
# - maxit: maximum number of step-halving iterations
#
function propagate(μ₀, Σ₀, δt, t₀, score, hessian, dTarget; maxit = 20)

  d = length(μ₀);

  # Output mean and variance
  μ₁ = copy(μ₀);
  Σ₁ = Array(Float64, d, d);

  # Hessians
  H = Array(Float64, d, d);
  H₀ = Array(Float64, d, d);

  # Temporary storage
  μ = Array(Float64, d);

  # Current log-target value and gradient
  old = dTarget(μ₀; Log = true)[1];
  ∇logπ = score( μ₀ );

  t = 0.
  jj = 1;
  # Continue to propagate μ(t) and Σ(t) until the stopping time t₀
  while t < t₀ # START progagation

    kk = 1;
    Δ = -1.0;
    # Propagating μ
    # Halve the step size until the log-target increases in the chosen direction
    while( true )

      δt = minimum([δt; t₀ - t]);

      # Second order Runge-Kutta
      μ = μ₀ + (δt/2.) * ∇logπ/2.; # Half move
      μ₁ = μ₀ + δt * score(μ)/2.   # Whole move

      Δ =  try   dTarget(μ₁; Log = true)[1] - old    catch    -1.    end

      if( kk > maxit ) throw( "lineSearch exceded the max number of iterations" ); end

      if (Δ < 0.)   δt /= 2.   else    break   end

      kk += 1;

    end

    # Propagating Σ
    if(Σ₀ == nothing)

      Σ₁ = δt * eye(d);

    else

      H₀ = hessian(μ₀);
      H = hessian(μ);

      # Second order Runge-Kutta
      #Σ₁  = Σ₀ + (δt/4.) * (H₀*Σ₀ + Σ₀*H₀ + H₀*Σ₀*H₀*(δt/4.))  + (δt/2.)*eye(d);
      #Σ₁  = Σ₀ + (δt/2.) * (H*Σ₁ + Σ₁*H + H*Σ₁*H*(δt/2)) + δt*eye(d);

      # Forward Euler: for some reason it seems to work better than RK when δt is big.
      Σ₁ = Σ₀ + (δt/2.) * (H₀*Σ₀ + Σ₀*H₀ + H₀*Σ₀*H₀*(δt/2)) + δt*eye(d);

    end

    μ₀ = copy(μ₁);
    Σ₀ = copy(Σ₁);

    t += δt;

    jj += 1;

  end # END progagation

  #@printf("%d ", jj);

  return vec(μ₁), Σ₁;

end

#########################################################
# Nearest Neighbour Covariance
#########################################################
# Calculates a covariance for each column of x, using the sample
# covariance of the B nearest columns of x₀. This is very similar
# to what is described in "Estimating and Projecting Trends in HIV/AIDS Generalized
# Epidemics Using Incremental Mixture Importance Sampling".
## INPUT:
# - x: the positions at which the covariance is calculated
# - w: the weights of x₀, used to calculated weighted nearest neighbour covariance
# - x₀: a matrix contaning the vectors using to calculate the covariances
# - Σ₀: the covariance used to define the mahalanobis distances between elements of x and x₀
# - B: number of nearest neighbours used to calculate the covariance
## OUTPUT:
# - Σout a 3D array of covariances, where Σout[:, :, ii] is the covariance of x[:, ii]
#
function nnCov(x, w, x₀, Σ₀, B)

  if( ndims(x) < 2 ) x = x'' end

  nₓ = size(x, 2);
  d, n₀ = size( x₀ );

  dec = sparse( chol( Σ₀ )' );

  # Create storage to be used my maha()
  storeM = Array(Float64, d, n₀);
  storeV = Array(Float64, n₀);

  if( B >  n₀) throw("Neighbourhood size B bigger then total population size(x₀, 2)"); end

  Σout = zeros(d, d, nₓ)

  for ii = 1:nₓ

    dist = maha(x₀, x[:, ii], dec; isChol = true, A = storeM, out = storeV)
    neigh = sortperm(dist)[ 1:B ];

    Σout[:, :, ii] = cov(x₀[:, neigh]', WeightVec(w[neigh]));

  end

  return Σout;

end

##### Test nnCov
#tmpd = 3
#B = 100000;
#x = zeros(tmpd);
#x₀ = randn(tmpd, 100000);
#Σ₀ = eye(tmpd);
#w =  ones( size(x₀, 2) );
#nnCov(x, w, x₀, Σ₀, B) - eye(tmpd)


###########################################################
# Calculates Expected Effective Sample Size between a
# Gaussian target (t) and a Gaussian importance density (p)
###########################################################
function expESS(μt, μp, Σt, Σp)

  d = size(μt)[1];

  ΣDif = Array(Float64, d, d);
  try
    ΣDif = cholfact(2.*Σp .- Σt);
    Σp = cholfact(Σp);
  catch
    return 0.0;
  end

  try  Σt = cholfact(Σt);  catch  error("The covariance of the target is not positive definite")  end

  # Old Version
  #varW = (2.^d * det(inv(Σt) .- 0.5*inv(Σp)))^-0.5 * det(Σp)^0.5 * det(Σt)^-1 *
  #        exp( dot(μp .- μt, (2.*Σp .- Σt) \ (μp .- μt) ) );

  # Interpretable version as in the paper
  #varW =  π^(d/2) * det(sigP) / sqrt(det(Σt)) /  sqrt((-2*π)^d * det(Σt/2 - Σp))  *
  #                        exp( -0.5 * dot(μp .- μt, (Σt/2 - Σp) \ (μp .- μt) ) )

  # Computationally faster version
  varW =  det(Σp) / sqrt(det(Σt)) /  sqrt(det(ΣDif))  * exp( dot(μp .- μt, ΣDif \ (μp .- μt) ) )

  # @printf("ESS = %f \n", 1. / varW);

  return( 1. / varW )

end

# Test
muT = [0., 0.];
sigT = 9. * [1 -0.5; -0.5 1];
muP = [2., 1.]
sigP = 20. * [1 -0.3; -0.3 1];

N = 100000;
x = rand(MvNormal(muP, sigP), N);

w = pdf(MvNormal(muT, sigT), x) ./ pdf(MvNormal(muP, sigP), x);
1 / mean(w.^2.)
expESS(muT, muP, sigT, sigP)


#######
# Choosing Langevin step length so that the chosen ESS is achieved
#######
# This function chooses the step size δt so that the Guassians N(μ₁, Σ₁) and N(μ₂, Σ₂)
# costructed at the current μ₀ and at the proposed μ₁ position differ by a predetermined
# amount. This distance is quantified in terms of the expected ESS that would be achieved
# of we used N(μ₂, Σ₂) to get an importance sample from N(μ₁, Σ₁).
# INPUT
# - μ: current position
# - Σ: current covariance. This will be propagated forward using the Jacobian of the transformation.
# - score: gradient of the target density
# - hessian: hessian of the target density
# - targetESS: the expected ESS to aim for. targetESS ∈ (0, 1). Generally as targetESS increases the
#              step δt becomes smaller.
# OUTPUT
# - δt: the chosen step size.
function matchESS(μ₀, Σ₀, score, hessian, dTarget, targetESS; rel_tol = 1e-1)

  # Internal objective function, quantifying the distance between
  # the targetESS and the expected ESS
  function objFun( δt )

    if ( δt < 1e-18 ) return(targetESS - 1.0) end

    Δ = 1.0;

    # Two errors can occur, both indicate that δt is too large:
    # 1. propagate() needs to halve the step size.
    # 2. expESS give an error.
    try

      # Propagating using full step size
      μ₁, Σ₁ = propagate(μ₀, Σ₀, δt, δt, score, hessian, dTarget; maxit = 1);

      # Propagating using 10 steps with 1 / 10th of the step size
      μ₂, Σ₂ = propagate(μ₀, Σ₀, δt / 10., δt, score, hessian, dTarget; maxit = 1);

      Δ  =  targetESS - expESS(μ₂, μ₁, Σ₂, Σ₁);

    catch end

    # @printf("Δ  = %f", Δ)

    return Δ

  end

  minδt = 1e-6;
  maxδt = 1.0;
  δt = copy( maxδt );

  if( objFun( maxδt ) > 0 )
    if( objFun( minδt ) > rel_tol * (1.0 - targetESS) )

      error("We cannot achieve the chosen ESS, even using the smallest step size");

    else

      δt = rootBisect(objFun, 1e-6, 1.; tol = rel_tol * (1.0 - targetESS), maxit = 50);

    end
  end

  return δt

end

#Test matchESS
#include("Examples/Gaussian.jl")
#chooseStep(μ_T + rand(d), nothing, 0.1, score, hessian, 0.99)

#include("Examples/myBanana.jl");
#chooseStep([3.; 6.], 0.1*eye(2), score, hessian, 0.99999)
