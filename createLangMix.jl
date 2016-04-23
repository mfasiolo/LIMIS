####
# Creates Langeving mixture density
####
# Given a set of initial points it creates a Gaussian mixture, by linearizing a Langevin diffusion.
# INPUT
# - x₀: the initial points
# - score: gradient of the target density
# - hessian: hessian of the target density
# - t₀: pseudo-time at which the linearization stops
# - targetESS: the expected ESS to aim for at each step of the Langevin Linearization. targetESS ∈ (0, 1).
#              It determines the step size during linearization, as targetESS increases the steps become smaller.
#              Default is targetESS = 0.999.
# - Q: optional scaling matrix for the Langeving diffusion. Currently not used.
# - Σ₀: optional initial covariances of each component of the mixture. If it is provided, then the i-th output
#       covariance will start from Σ₀[:, :, i] at the first iteration.
# OUTPUT
# - μ_out, Σ_out: means and covariances of each mixture component
#
function createLangMix(x₀, score, hessian, t₀; targetESS = 1-1e-3, Q = nothing, Σ₀ = nothing)

  if ( ndims(x₀) < 2 ) x₀ = x₀''; end

  d, n = size( x₀ );

  # These will store means and covariances of the mixture
  μ_out = zeros(d, n);
  Σ_out = zeros(d, d, n);

  # Loop over starting points x₀ⁱ
  for ii = 1:n

    Σ  =  if(Σ₀ == nothing)    nothing    else    Σ₀[:, :, ii]    end

    # Select step-size δt
    δt = matchESS(x₀[:, ii], Σ, score, hessian, dTarget, targetESS);

    # Propagate μ(t) and Σ(t) until the stopping time t₀
    μ, Σ = propagate(x₀[:, ii], Σ, δt, t₀, score, hessian, dTarget);

    μ_out[:, ii] = μ;
    Σ_out[:, :, ii] = Σ;

  end

  return μ_out, Σ_out

end
