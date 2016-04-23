#################
## Example functions
#################

###### Gaussian
# Target dimension, mean and covariance
d = 2;
μ_T = zeros(d);
Σ_T = [4.  0; 0  0.2]#1. * eye(d);

μ_P = zeros(d);
Σ_P = [4.  0; 0  0.2]#eye(d);

# Prior density
function dPrior(x_; Log = false)

  out = logpdf(MvNormal(μ_P, Σ_P), x_);

  if( !Log ) out = exp( out ); end

  return( out );

end

# Prior Generator
rPrior(n_) = rand(MvNormal(μ_P, Σ_P), n_);

# Target density function
function dTarget(x_; Log = false)

 out = logpdf(MvNormal(μ_T, Σ_T), x_);

 if( !Log ) out = exp( out ); end

 return(out)

end

# Target score
function score(x)

 return Σ_T \ broadcast(+, -x, μ_T);

end

# Target Hessian
function hessian(x)

 return - inv(Σ_T);

end
