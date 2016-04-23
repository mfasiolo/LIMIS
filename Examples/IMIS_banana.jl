##########################################################################################
#################### BANANA IMIS EXAMPLE
##########################################################################################

### TODO
# - In IMIS, find this line "add = minimum(dist) > - 0.75;". Where we are checking
# whether the expected ESS of old mixtures wrt the new components is > 0.75. We need
# to write in the text that we do two checks, one before propagating the mixture (which)
# is used also by NIMIS, and one after propagating the mixture (the one I am referring to
# above), which is done only in LIMIS.

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

###########################
####### BANANA
###########################

### Banana
banSh = [10.; 5];
bananicity = 0.03;
sigmaBan = 6;
d = 2;

#include("Examples/newBanana.jl");

# Mixture of Bananas
d = 2
banDim = copy(d);
bananicity = [0.2, -0.03, 0.1, 0.1, 0.1, 0.1];
sigmaBan = [1, 6, 4, 4, 1, 1];
banShiftX = [0, 0, 7, -7, 7, -7];
banShiftY = [0, -5, 7, 7, 7.5, 7.5];
nmix = length(bananicity);
bananaW = [1, 4, 2.5, 2.5, 0.5, 0.5]; #ones( nmix ) / nmix #[0.2, 0.6, 0.2]; #;
bananaW = bananaW / sum(bananaW)

include("Examples/mixtureBanana.jl");

### Set up
srand(525)
niter = 200;
n = 100 * d;
n₀ = 1000 * d;
t₀ = log( ( (4/(d+2)) ^ (1/(d+4)) * niter ^ -(1/(d+4)) )^2 + 1 );

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
ESS = { resL["ESS"], resL_R["ESS"], resN["ESS"], resN_R["ESS"] };
dimMix = { resL["dimMix"], resL_R["dimMix"], resN["dimMix"], resN_R["dimMix"] };
res = {ESS, dimMix};

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

########################
##### Getting some estimates
########################

resL = IMIS2(400, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0.25, useLangevin = true, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
            );

ns = size( resL["X₀"] )[2];

trueSam = ( rBanana(10^7) )';
iSam = resL["X₀"];

# Mean
iMean = iSam * resL["w"] / ns
mean(trueSam, 2)

# Variance
((iSam .- iMean).^2) * resL["w"] / ns
var(trueSam, 2)

###############
trueSam = ( rBanana(10^7) )';

a = hcat( rPrior(n₀), rGausMix(ns-n₀, resL["μ₀"], resL["Σ₀"]; df = 3, w = resL["wmix"]));

# a = resL["X₀"];
dimp = ( n₀ * dPrior(a)  + (ns - n₀) * dGausMix(a, resL["μ₀"], resL["Σ₀"]; df = 3, w = resL["wmix"])[1] ) / ns;
w = dTarget(a) ./ dimp ;

aMean = a * w / ns
((a .- aMean).^2) * w / ns
var(trueSam, 2)



##################################################################################
##############   Plotting the importance density
##################################################################################

# Mixture of Bananas
d = 2
banDim = copy(d);
bananicity = [0.2, -0.03, 0.1, 0.1, 0.1, 0.1];
sigmaBan = [1, 6, 4, 4, 1, 1];
banShiftX = [0, 0, 7, -7, 7, -7];
banShiftY = [0, -5, 7, 7, 7.5, 7.5];
nmix = length(bananicity);
bananaW = [1, 4, 2.5, 2.5, 0.5, 0.5]; #ones( nmix ) / nmix #[0.2, 0.6, 0.2]; #;
bananaW = bananaW / sum(bananaW);

# d = 2
# banDim = copy(d);
# bananicity = [0.03, 0.03];
# sigmaBan = [6, 6];
# banShiftX = [0, 0];
# banShiftY = [0, 0];
# nmix = length(bananicity);
# bananaW = [1, 4]; #ones( nmix ) / nmix #[0.2, 0.6, 0.2]; #;
# bananaW = bananaW / sum(bananaW)

include("Examples/mixtureBanana.jl");
# include("Examples/Gaussian.jl");

n = 100 * d;
n₀ = 1000 * d;

### 1 LIMIS
srand(525);
niter = 50;
t₀ = 1.; #log( ( (4/(d+2)) ^ (1/(d+4)) * niter ^ -(1/(d+4)) )^2 + 1 );

# Langevin IMIS
resL1 = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0., useLangevin = true, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
            );

#### 3 Tune Imis
tseq = .5:0.5:6;
expVar = tuneIMIS(resL1, tseq; frac = 1., verbose = true);
plot(tseq, expVar);

# plot(0:1:(niter), resL1["ESS"], label = "LIMIS");
plot(1:niter, resL1["dimMix"]);

#### 2
niter = 200;
t₀ = 1.; #log( ( (4/(d+2)) ^ (1/(d+4)) * niter ^ -(1/(d+4)) )^2 + 1 );

# Langevin IMIS
resL2 = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0., useLangevin = true, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
            );

plot(0:1:(niter), resL2["ESS"], label = "LIMIS");

#### 3 NIMIS
niter = 200;

# NN IMIS
resL3 = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0., useLangevin = false, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
            );

plot(0:1:(niter), resL3["ESS"], label = "LIMIS");

##### 4 LIMIS 20 Dimensions
# Mixture of Bananas
d = 20
banDim = copy(d);
bananicity = [0.2, -0.03, 0.1, 0.1, 0.1, 0.1];
sigmaBan = [1, 6, 4, 4, 1, 1];
banShiftX = [0, 0, 7, -7, 7, -7];
banShiftY = [0, -5, 7, 7, 7.5, 7.5];
nmix = length(bananicity);
bananaW = [1, 4, 2.5, 2.5, 0.5, 0.5]; #ones( nmix ) / nmix #[0.2, 0.6, 0.2]; #;
bananaW = bananaW / sum(bananaW);

include("Examples/mixtureBanana.jl");
# include("Examples/Gaussian.jl");

n = 100 * d;
n₀ = 1000 * d;

### 1 LIMIS
srand(525);
niter = 400;
t₀ = 4.; #log( ( (4/(d+2)) ^ (1/(d+4)) * niter ^ -(1/(d+4)) )^2 + 1 );

# Langevin IMIS
resL4 = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0., useLangevin = true, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
            );

### Plots
L = 100;
x1 = linspace(-20., 20., L);
x2 = linspace(-15., 15., L);

#x1 = linspace(-10., 10., L);
#x2 = linspace(-4., 4., L);

# Define target parameters
d_lik1 = eye(L);
d_lik2 = eye(L);
d_lik3 = eye(L);
d_lik4 = eye(L);

for iRow = 1:L
  for iCol = 1:L
    #@printf("%d", iCol);dGausMix(x, μ, Σ; Log = false, df = Inf, w = nothing, dTrans = nothing)
    d_lik1[iRow, iCol] = dTarget(vcat([x1[iCol] x2[iRow]][:], rep(0., d-2)))[1];
  end;
end;

for iRow = 1:L
  for iCol = 1:L
    #@printf("%d", iCol);dGausMix(x, μ, Σ; Log = false, df = Inf, w = nothing, dTrans = nothing)
    d_lik2[iRow, iCol] = dGausMix(vcat([x1[iCol] x2[iRow]][:], rep(0., d-2))'', resL3["μ₀"], resL3["Σ₀"]; df = 3)[1][1];
  end;
end;

for iRow = 1:L
  for iCol = 1:L
    #@printf("%d", iCol);dGausMix(x, μ, Σ; Log = false, df = Inf, w = nothing, dTrans = nothing)
    d_lik3[iRow, iCol] = dGausMix(vcat([x1[iCol] x2[iRow]][:], rep(0., d-2))'', resL2["μ₀"], resL2["Σ₀"]; df = 3)[1][1];
  end;
end;

for iRow = 1:L
  for iCol = 1:L
    #@printf("%d", iCol);dGausMix(x, μ, Σ; Log = false, df = Inf, w = nothing, dTrans = nothing)
    d_lik4[iRow, iCol] = dGausMix(vcat([x1[iCol] x2[iRow]][:], rep(0., d-2))'', resL4["μ₀"], resL4["Σ₀"]; df = 3)[1][1];
  end;
end;

fig = figure();
subplot(221);
grid("on");
title("Target density");
xlabel("x1")
ylabel("x2")

contour(x1, x2, d_lik1);

subplot(222);
grid("on");
title("NIMIS 200 iterations, 2 dimensions");
xlabel("x1")
ylabel("x2")

contour(x1, x2, d_lik2);
# tmp = resL3["μOrig"]';
# good = (tmp[:, 1] .> x1[1])[:] & (tmp[:, 1] .< x1[end])[:] & (tmp[:, 2] .> x2[1])[:] & (tmp[:, 2] .< x2[end])[:];
# scatter(tmp[good, 1][:], tmp[good, 2][:]);
# tmp = resL3["μ₀"]';
# good = (tmp[:, 1] .> x1[1])[:] & (tmp[:, 1] .< x1[end])[:] & (tmp[:, 2] .> x2[1])[:] & (tmp[:, 2] .< x2[end])[:];
# scatter(tmp[good, 1][:], tmp[good, 2][:], c = "red");

subplot(223);
grid("on");
title("LIMIS 200 iterations, 2 dimensions");
xlabel("x1")
ylabel("x2")

contour(x1, x2, d_lik3);
# tmp = resL2["μOrig"]';
# good = (tmp[:, 1] .> x1[1])[:] & (tmp[:, 1] .< x1[end])[:] & (tmp[:, 2] .> x2[1])[:] & (tmp[:, 2] .< x2[end])[:];
# scatter(tmp[good, 1][:], tmp[good, 2][:]);
# tmp = resL2["μ₀"]';
# good = (tmp[:, 1] .> x1[1])[:] & (tmp[:, 1] .< x1[end])[:] & (tmp[:, 2] .> x2[1])[:] & (tmp[:, 2] .< x2[end])[:];
# scatter(tmp[good, 1][:], tmp[good, 2][:], c = "red");


subplot(224);
grid("on");
title("LIMIS 400 iterations, 20 dimensions");
xlabel("x1")
ylabel("x2")

contour(x1, x2, d_lik4);
# tmp = resL4["μOrig"]';
# good = (tmp[:, 1] .> x1[1])[:] & (tmp[:, 1] .< x1[end])[:] & (tmp[:, 2] .> x2[1])[:] & (tmp[:, 2] .< x2[end])[:];
# scatter(tmp[good, 1][:], tmp[good, 2][:]);
# tmp = resL4["μ₀"]';
# good = (tmp[:, 1] .> x1[1])[:] & (tmp[:, 1] .< x1[end])[:] & (tmp[:, 2] .> x2[1])[:] & (tmp[:, 2] .< x2[end])[:];
# scatter(tmp[good, 1][:], tmp[good, 2][:], c = "red");

##################################################################################
######################## Monte Carlo Experiment
##################################################################################

idim = 1;

d = [2; 5; 10; 20][idim];
niter = [200, 400, 800, 1000][idim];
quL = [0.25, 0.25, 0.25, 0.99][idim];
quN = [0.05, 0.05, 0.15, 0.99][idim];
t₀ = [1, 1, 2, 4][idim];

# Mixture of Bananas
banDim = copy(d);
bananicity = [0.2, -0.03, 0.1, 0.1, 0.1, 0.1];
sigmaBan = [1, 6, 4, 4, 1, 1];
banShiftX = [0, 0, 7, -7, 7, -7];
banShiftY = [0, -5, 7, 7, 7.5, 7.5];
nmix = length(bananicity);
bananaW = [1, 4, 2.5, 2.5, 0.5, 0.5]; #ones( nmix ) / nmix #[0.2, 0.6, 0.2]; #;
bananaW = bananaW / sum(bananaW);

include("Examples/mixtureBanana.jl");

# Create importance mixture
μMix = vcat([0. 0. 7 -7; -6. 0 8.2 8.2], zeros(d-2, 4));
ΣMix = zeros(d, d, 4);
for ii = 1:4   ΣMix[:, :, ii] = -2*inv(hessian(μMix[:, ii][:]));   end
wMix = bananaW[1:4] / sum(bananaW[1:4]);

### Set up
#srand(542625);
n = 100 * d;
n₀ = 1000 * d;

nrep = 2;
resL = [];
resL_R = [];
resN = [];
resN_R = [];
resMix = [];

for ii in 1:nrep

  ### Langevin IMIS
  tmp = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
              df = 3, trunc = true, quant = 0, useLangevin = true, verbose = true,
              t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
              );

  resL = [resL; tmp];

  tmp = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
              df = 3, trunc = true, quant = quL, useLangevin = true, verbose = true,
              t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
              );

  resL_R = [resL_R; tmp];

  ### Nearest Neighbour IMIS
  tmp = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
              df = 3, trunc = true,  quant = 0, useLangevin = false, verbose = true);

  resN = [resN; tmp];

  # tmp = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
  #             df = 3, trunc = true,  quant = quN, useLangevin = false, verbose = true);
  #
  # resN_R = [resN_R; tmp];

  ### Gaussian Mixture IS
  rtmp = rGausMix(n₀ + n*niter, μMix, ΣMix; df = 3, w = wMix);
  wtmp = dTarget(rtmp)./dGausMix(rtmp, μMix, ΣMix; df = 3, w = wMix)[1][:];
  esstmp = ( 1 / sum( (wtmp/sum(wtmp)).^2 ) ) / (n₀ + n*niter);
  resMix = [resMix; Dict{Any,Any}("X₀" => rtmp, "w" => wtmp, "ESS" => esstmp)];

  ### MALA
  function wrap1(x) dTarget(x; Log = true)[1] end
  function wrap2(x) score(x)[:]; end
  p = BasicContMuvParameter(:p, logtarget = wrap1,
                                gradlogtarget = wrap2)
  model = likelihood_model(p, false);
  sampler = MALA(0.9);
  mcrange = BasicMCRange(nsteps=(n₀ + n*niter), burnin=round(Int, (n₀ + n*niter)/10.));
  v0 = Dict(:p=>rep(0., d));
  ### Save grad-log-target along with the chain (value and log-target)
  outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget, :gradlogtarget],
                              :diagnostics=>[:accept]);
  job = BasicMCJob(model, sampler, mcrange, v0,
                   tuner=VanillaMCTuner(verbose=true), outopts=outopts);
  run(job);
  chain = output(job);
  scatter(chain.value[1, :][:], chain.value[2, :][:])

  Lora.ess(chain.value)

end

ESSL = reduce(hcat, map(x_ -> x_["ESS"][:], resL) )';
ESSL_R = reduce(hcat, map(x_ -> x_["ESS"][:], resL_R) )';
ESSN = reduce(hcat, map(x_ -> x_["ESS"][:], resN) )';
ESSMix = reduce(hcat, map(x_ -> x_["ESS"], resMix) )';
# ESSN_R = reduce(hcat, map(x_ -> x_["ESS"][:], resN_R) )';

sizeL = reduce(hcat, map(x_ -> x_["dimMix"][:], resL) )';
sizeL_R = reduce(hcat, map(x_ -> x_["dimMix"][:], resL_R) )';
sizeN = reduce(hcat, map(x_ -> x_["dimMix"][:], resN) )';
# sizeN_R = reduce(hcat, map(x_ -> x_["dimMix"][:], resN_R) )';

fig = figure();
subplot(121);
grid("on");
title("Effective Sample Size (ESS)");
xlabel("Iteration")
ylabel("ESS")
plot(0:1:(niter), mean(ESSL, 1)[:], label = "LIMIS")
plot(0:1:(niter), mean(ESSL_R, 1)[:], label = "LIMIS_R")
plot(0:1:(niter), mean(ESSN, 1)[:], label = "NIMIS")
plot(0:1:(niter), rep(mean(ESSMix, 1)[:], niter+1), label = "GausMix")
# plot(0:1:(niter), mean(ESSN_R, 1)[:], label = "NIMIS_R")
legend(loc="lower right",fancybox="true")

subplot(122);
grid("on");
title("Number of mixture components (NC)");
xlabel("Iteration")
ylabel("NC")
plot(1:(niter), mean(sizeL, 1)[:], label = "LIMIS")
plot(1:(niter), mean(sizeL_R, 1)[:], label = "LIMIS_R")
plot(1:(niter), mean(sizeN, 1)[:], label = "NIMIS")
# plot(1:(niter), mean(sizeN_R, 1)[:], label = "NIMIS_R")


#############################
# Checking estimates
#############################

######
## Marginal accuracies
######

truX = rBanMix(1e7);

### Dimension 1
δ = 0.1;
ySeq = -20:δ:20;
densTRUE = kde(ySeq, truX[:, 1][:], 0.1);

h = 0.2
densL = map(O -> kde(ySeq, O["X₀"][1, :][:], h; w = O["w"]), resL);
densL_R = map(O -> kde(ySeq, O["X₀"][1, :][:], h; w = O["w"]), resL_R);
densN = map(O -> kde(ySeq, O["X₀"][1, :][:], h; w = O["w"]), resN);
densMix = map(O -> kde(ySeq, O["X₀"][1, :][:], h; w = O["w"]), resMix);

ii = 1
plot(ySeq, densL[ii], label = "LIMIS");
plot(ySeq, densL_R[ii], label = "LIMIS_R");
plot(ySeq, densN[ii], label = "NIMIS");
plot(ySeq, densMix[ii], label = "GausMix");
plot(ySeq, densTRUE, label = "Truth");
legend(loc="lower center",fancybox="true")

maccL1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densL);
maccL_R1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densL_R);
maccN1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densN);
maccMix1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densMix);
hcat(maccL1, maccL_R1, maccN1, maccMix1)

### Dimension 2
δ = 0.1;
ySeq = -11:δ:15;
densTRUE = kde(ySeq, truX[:, 2][:], 0.05);

h = 0.1;
densL = map(O -> kde(ySeq, O["X₀"][2, :][:], h; w = O["w"]), resL);
densL_R = map(O -> kde(ySeq, O["X₀"][2, :][:], h; w = O["w"]), resL_R);
densN = map(O -> kde(ySeq, O["X₀"][2, :][:], h; w = O["w"]), resN);
densMix = map(O -> kde(ySeq, O["X₀"][2, :][:], h; w = O["w"]), resMix);

ii = 1
plot(ySeq, densL[ii], label = "LIMIS");
plot(ySeq, densL_R[ii], label = "LIMIS_R");
plot(ySeq, densN[ii], label = "NIMIS");
plot(ySeq, densMix[ii], label = "GausMix");
plot(ySeq, densTRUE, label = "Truth");
legend(loc="lower center",fancybox="true")

maccL1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densL);
maccL_R1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densL_R);
maccN1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densN);
maccMix1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densMix);
hcat(maccL1, maccL_R1, maccN1, maccMix1)
