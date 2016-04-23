##########################################################################################
#################### INCREMENTAL MIXTURE IMPORTANCE SAMPLING
##########################################################################################

##
# TODO
##
# - Write down how to choose the initial t\_0 using the smoothing approach from Silverman
# - Write in the Lyx file that now we are not choosing \epsilon at each step.

# Urgent
#
# - At the moment Q is used to indicate what in my paper is indicated with Q and D, which
#   are really different things. Sort it out.
# - Finish the lag selection procedure.
# - Sort out when to truncate, and when to use self-normalized IS. At the moment you
#   are truncating all the time, and probably using self-normalized even when both target and
#   importance density are normalized.
# - Use the meanExpTrick in several places.

#######
# Load functions and one of the examples
#######

cd("$(homedir())/Desktop/All/Dropbox/Work/Liverpool/IMIS/Julia_code")

using StatsBase;
using Distributions;
using PyPlot;
using Optim;
using Distances;
using Roots;
using HDF5, JLD;

# Loading necessary functions
include("utilities.jl");
include("mvt.jl");
include("createLangMix.jl");
include("IMIS.jl");
include("tuneIMIS.jl");
include("fastMix.jl");

#include("Examples/Gaussian.jl");
include("Examples/myBanana.jl");
#include("Examples/flavio_banana.jl");
#include("Examples/logistic_regression.jl");

###### Run Tests
include("Tests/mvtTest.jl");
#include("Tests/fastMixTest.jl");

#####
# Incremental Mixture Importance Sampling
#####

### Set up
srand(525);
niter = 100;
n = 100 * d;
n₀ = 1000 * d;
t₀ = ( (4/(d+1)) ^ (1/(d+4)) * niter ^ -(1/(d+4)) )^2 ;
t₀ = log( ( (4/(d+1)) ^ (1/(d+4)) * niter ^ -(1/(d+4)) )^2 + 1 );

### Langevin IMIS
resL = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0.25, useLangevin = true, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
            );

plot(1:1:(niter+1), resL["ESS"]);

plot(1:niter, resL["dimMix"]);

a = false;
plotTarget()
for ii = 1:length(resL["wmix"])

  a = !a;
  if(a) col = "red" else col = "blue" end
  scatter(resL["μOrig"][1, :][ii], resL["μOrig"][2, :][ii], color = col)
  chomp(readline())

end

### Nearest Neighbour IMIS
resN = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true,  quant = 0.25, useLangevin = false, verbose = true);

plot(1:1:(niter+1), resN["ESS"]);

plot(1:niter, resN["dimMix"]);

plot(1:1:(niter), resL["ESS"][1:niter]./resL["dimMix"]);
plot(1:1:(niter), resN["ESS"][1:niter]./resN["dimMix"]);


#### Selecting optimal number of steps
tseq = .02:0.02:.5;

expVar = tuneIMIS(resL, tseq; frac = 1., verbose = true);

plot(tseq, expVar);


#################
# Testing fast mixture procedure
#################

# Extract mixture from output of IMIS
μL = resL["μ₀"];
ΣL = resL["Σ₀"];
wL = resL["wmix"];

# Create Fast mixture
ϕ = createϕTree(μL, ΣL);
ϕMat = getϕMat(ϕ);
Q = createQTree(ϕ, μL, ΣL);
goodMix = selectMix(ϕ, Q, 0.01, w = wL);

# Sample from true mixture
df = Inf
n = 100000
x, labs = rGausMix(n, μL, ΣL; df = df, w = wL, labs = true);

# Put labels of each sample in a single vector
tmpLab = mapreduce(ii -> rep(labs[ii, 1], labs[ii, 2]), vcat, 1:size(labs, 1));

# Evaluate approximate and true mixture
@time ap = dFastGausMix(x, tmpLab, ϕMat, goodMix, μL, ΣL; Log = false, df = df, w = wL);

@time truth = dGausMix(x, μL, ΣL; Log = false, df = df, w = wL, dTrans = nothing)[1];

# Plot a subsample of the mixture evaluations
nsub = 10000;
index = round(rand(nsub) * n, 0);

scatter(ap[index][:], truth[index][:])
plot(minimum(ap[index]):1e-6:maximum(ap[index]), minimum(ap[index]):1e-6:maximum(ap[index]), "red")

[ap[1:20] truth[1:20]]

map(x_ -> length(goodMix[x_]), keys(goodMix))




################################################################
################################################################
############ MONTE CARLO EXAMPLES
################################################################
################################################################

###########################
####### BANANA
###########################
include("Examples/myBanana.jl");

### Set up
srand(525)
niter = 500;
n = 100 * d;
n₀ = 1000 * d;
t₀ = log( ( (4/(d+1)) ^ (1/(d+4)) * niter ^ -(1/(d+4)) )^2 + 1 );

### Langevin IMIS
resL = IMIS2(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0.01, useLangevin = true, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
            );

resL_R = IMIS2(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0.25, useLangevin = true, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
            );

### Nearest Neighbour IMIS
resN = IMIS2(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true,  quant = 0.01, useLangevin = false, verbose = true);

resN_R = IMIS2(niter, n, n₀, dTarget, dPrior, rPrior;
              df = 3, trunc = true,  quant = 0.25, useLangevin = false, verbose = true);


### Save results to file
ESS = { resL["ESS"], resL_R["ESS"], resN["ESS"], resN_R["ESS"] }
dimMix = { resL["dimMix"], resL_R["dimMix"], resN["dimMix"], resN_R["dimMix"] }
res = {ESS, dimMix}

#file = jldopen("Data/Banana_2d.jld", "w");
#@write file res;
#close(file);

### Plots
fig = figure();
subplot(121);
grid("on");
title("Effective Sample Size (ESS)");
xlabel("Iteration")
ylabel("ESS")
plot(0:1:(niter), resL["ESS"], label = "LIMIS");
plot(0:1:(niter), resL_R["ESS"], label = "LIMIS MR");
plot(0:1:(niter), resN["ESS"], label = "NIMIS");
plot(0:1:(niter), resN_R["ESS"], label = "NIMIS MR");
legend(loc="lower right",fancybox="true")

subplot(122);
grid("on");
title("Number of mixture components (NC)");
xlabel("Iteration")
ylabel("NC")
plot(1:niter, resL["dimMix"]);
plot(1:niter, resL_R["dimMix"]);
plot(1:niter, resN["dimMix"]);
plot(1:niter, resN_R["dimMix"]);


### Load results from file
file = jldopen("Data/Banana_2d.jld", "r")
res = read(file, "res")
close(file)

ESS = res[1];
dimMix = res[2];
niter = length(ESS[1])-1

### Plots
fig = figure();
subplot(121);
grid("on");
title("Normalized ESS (NESS)");
xlabel("Iteration")
ylabel("NESS")
plot(0:1:(niter), ESS[1], label = "LIMIS");
plot(0:1:(niter), ESS[2], label = "LIMIS MR");
plot(0:1:(niter), ESS[3], label = "NIMIS");
plot(0:1:(niter), ESS[4], label = "NIMIS MR");
legend(loc="lower right",fancybox="true")

subplot(122);
grid("on");
title("Number of mixture components (NC)");
xlabel("Iteration")
ylabel("NC")
plot(1:niter, dimMix[1]);
plot(1:niter, dimMix[2]);
plot(1:niter, dimMix[3]);
plot(1:niter, dimMix[4]);












###########################
####### Logistic Regression
###########################

srand(525)

include("Examples/logistic_regression.jl");

### Set up
niter = 100;
n = 100 * d;
n₀ = 1000 * d;
t₀ = log( ( (4/(d+1)) ^ (1/(d+4)) * niter ^ -(1/(d+4)) )^2 + 1 );

### Langevin IMIS
resL = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0.01, useLangevin = true, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
            );

resL_R = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0.25, useLangevin = true, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
            );

### Nearest Neighbour IMIS
resN = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true,  quant = 0.01, useLangevin = false, verbose = true);

resN_R = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
              df = 3, trunc = true,  quant = 0.25, useLangevin = false, verbose = true);


### Save results to file
ESS = { resL["ESS"], resL_R["ESS"], resN["ESS"], resN_R["ESS"] }
dimMix = { resL["dimMix"], resL_R["dimMix"], resN["dimMix"], resN_R["dimMix"] }
res = {ESS, dimMix}

#file = jldopen("Data/Logistic_15d.jld", "w");
#@write file res;
#close(file);

### Plots
fig = figure();
subplot(121);
grid("on");
title("Normalized Effective Sample Size (NESS)");
xlabel("Iteration")
ylabel("NESS")
plot(0:1:(niter), resL["ESS"], label = "LIMIS");
plot(0:1:(niter), resL_R["ESS"], label = "LIMIS MR");
plot(0:1:(niter), resN["ESS"], label = "NIMIS");
plot(0:1:(niter), resN_R["ESS"], label = "NIMIS MR");
legend(loc="lower right",fancybox="true")

subplot(122);
grid("on");
title("Number of mixture components (NC)");
xlabel("Iteration")
ylabel("NC")
plot(1:niter, resL["dimMix"]);
plot(1:niter, resL_R["dimMix"]);
plot(1:niter, resN["dimMix"]);
plot(1:niter, resN_R["dimMix"]);


### Load results from file
file = jldopen("Data/Logistic_15d.jld", "r")
res = read(file, "res")
close(file)

ESS = res[1];
dimMix = res[2];
niter = length(ESS[1])-1

### Plots
fig = figure();
subplot(121);
grid("on");
title("Effective Sample Size (NESS)");
xlabel("Iteration")
ylabel("NESS")
plot(0:1:(niter), ESS[1], label = "LIMIS");
plot(0:1:(niter), ESS[2], label = "LIMIS MR");
plot(0:1:(niter), ESS[3], label = "NIMIS");
plot(0:1:(niter), ESS[4], label = "NIMIS MR");
legend(loc="lower right",fancybox="true")

subplot(122);
grid("on");
title("Number of mixture components (NC)");
xlabel("Iteration")
ylabel("NC")
plot(1:niter, dimMix[1]);
plot(1:niter, dimMix[2]);
plot(1:niter, dimMix[3]);
plot(1:niter, dimMix[4]);


