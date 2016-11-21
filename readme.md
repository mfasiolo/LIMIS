## LIMIS: Langevin Incremental Mixture Importance Sampling

This is the Julia code for the LIMIS algorithm of Fasiolo et al. (2016). 
The IMIS sampler of Raftery and Bao (2010) is also implemented, but we call it NIMIS. This is not yet a full-fledged 
Julia package, hence it cannot be installed using something like `Pkg.add("LIMIS")`.

Instead, assume that all the code here is in "~/Desktop/myfolder" then we need to change the working 
directory and load a number packages and functions. This can be done as follows:
```julia
# Change working directory
cd("myfolder");

# Load packages
using StatsBase, Distributions, PyPlot, Optim, Distances, Roots;

# Loading functions
include("utilities.jl");
include("mvt.jl");            # Multivariate student's t methods
include("createLangMix.jl");  # Linearization (Section 3 of Fasiolo et al. (2016))
include("IMIS.jl");           # LIMIS and NIMIS algorithm
include("tuneIMIS.jl");       # Determines the pseudo-time as in Section 7 of Fasiolo et al. (2016)
```

Having loaded the functions implementing LIMIS and NIMIS, we know load the functions for the mixture 
of warped Gaussians example:

```julia
# Parameters of banana mixture example (Global variables)
d = 5;                                            # five dimensional banana
banDim = copy(d);
bananicity = [0.2, -0.03, 0.1, 0.1, 0.1, 0.1];
sigmaBan = [1, 6, 4, 4, 1, 1];
banShiftX = [0, 0, 7, -7, 7, -7];
banShiftY = [0, -5, 7, 7, 7.5, 7.5];
nmix = length(bananicity);
bananaW = [1, 4, 2.5, 2.5, 0.5, 0.5];
bananaW = bananaW / sum(bananaW);

# Load functions for banana mixture
include("Examples/mixtureBanana.jl");
```
We can now sample the target using LIMIS (might take a minute or two):

```julia
res = IMIS(200, 100*d, 1000*d,         # Number of iterations, samples per iteration and samples from prior
            dTarget, dPrior, rPrior;
            df = 3,               # Degrees of freedom of the importance mixture 
            trunc = false,        # Should truncated importance sampling be used?
            useLangevin = true,   # If true use LIMIS otherwise NIMIS            
            t₀ = 2,               # Pseudo-time for LIMIS
            score = score, hessian = hessian, # Gradient and hessian of target
            targetESS = 1 - 1e-2); # Alpha parameter from Section 4
```

And now we compare the importance density and the target, across the first two dimensions:

```julia
# Create grid
L = 100;
x1 = linspace(-20., 20., L);
x2 = linspace(-15., 15., L);

# Evaluate target on grid
d_lik1 = eye(L);
for iRow = 1:L
  for iCol = 1:L
    d_lik1[iRow, iCol] = dTarget(vcat([x1[iCol] x2[iRow]][:], rep(0., d-2)))[1];
  end;
end;

# Evaluate LIMIS importance density on grid
d_lik2 = eye(L);
for iRow = 1:L
  for iCol = 1:L
    d_lik2[iRow, iCol] = dGausMix(vcat([x1[iCol] x2[iRow]][:], rep(0., d-2))'', res["μ₀"], res["Σ₀"]; df = 3)[1][1];
  end;
end;

# Plot 
fig = figure();
subplot(121);
grid("on");
title("Target density");
xlabel("x1")
ylabel("x2")
contour(x1, x2, d_lik1);

subplot(122);
grid("on");
title("NIMIS 200 iterations, 5 dimensions");
xlabel("x1")
ylabel("x2")
contour(x1, x2, d_lik2);
```

The result should look like this:
![alt tag](https://github.com/mfasiolo/LIMIS/blob/master/smile.png)
If it doesn't, let me know!

