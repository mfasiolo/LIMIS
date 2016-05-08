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
using Lora;

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

@everywhere cd("$(homedir())/Desktop/All/Dropbox/Work/Liverpool/IMIS/Julia_code");

include("paralSetUp.jl");
@everywhere include("paralSetUp.jl");
@everywhere blas_set_num_threads(1);

nrep = 2;

### Langevin IMIS
resL = pmap(useless -> IMIS2(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0, useLangevin = true, verbose = false,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2),
            1:1:nrep);

# + mixture reduction
resL_R = pmap(useless -> IMIS2(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = quL, useLangevin = true, verbose = false,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2),
            1:1:nrep);

### NIMIS
resN = pmap(useless -> IMIS2(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true,  quant = 0, useLangevin = false, verbose = false),
            1:1:nrep);

### Gaussian Mixture importance sampling
resMix = [];
for ii = 1:nrep
  rtmp = rGausMix(n₀ + n*niter, μMix, ΣMix; df = 3, w = wMix);
  wtmp = dTarget(rtmp)./dGausMix(rtmp, μMix, ΣMix; df = 3, w = wMix)[1][:];
  esstmp = ( 1 / sum( (wtmp/sum(wtmp)).^2 ) ) / (n₀ + n*niter);
  resMix = [resMix; Dict{Any,Any}("X₀" => rtmp, "w" => wtmp, "ESS" => esstmp)];
end

### MALA
resMALA = pmap(launchMALAjob, 1:1:nrep);

####
# Diagnostics
####

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

truX = rBanMix(1e6);

### Dimension 1
δ = 0.1;
ySeq = -20:δ:20;
densTRUE = kde(ySeq, truX[:, 1][:], 0.1);

h = 0.2
densL = map(O -> kde(ySeq, O["X₀"][1, :][:], h; w = O["w"]), resL);
densL_R = map(O -> kde(ySeq, O["X₀"][1, :][:], h; w = O["w"]), resL_R);
densN = map(O -> kde(ySeq, O["X₀"][1, :][:], h; w = O["w"]), resN);
densMix = map(O -> kde(ySeq, O["X₀"][1, :][:], h; w = O["w"]), resMix);
densMala = map(O -> kde(ySeq, O.value[1, :][:], h), resMALA);

ii = 1
plot(ySeq, densL[ii], label = "LIMIS");
plot(ySeq, densL_R[ii], label = "LIMIS_R");
plot(ySeq, densN[ii], label = "NIMIS");
plot(ySeq, densMix[ii], label = "GausMix");
plot(ySeq, densMala[ii], label = "MALA");
plot(ySeq, densTRUE, label = "Truth");
legend(loc="lower center",fancybox="true");

maccL1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densL);
maccL_R1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densL_R);
maccN1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densN);
maccMix1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densMix);
maccMala1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densMala);
hcat(maccL1, maccL_R1, maccN1, maccMix1, maccMala1)

### Dimension 2
δ = 0.1;
ySeq = -11:δ:15;
densTRUE = kde(ySeq, truX[:, 2][:], 0.05);

h = 0.1;
densL = map(O -> kde(ySeq, O["X₀"][2, :][:], h; w = O["w"]), resL);
densL_R = map(O -> kde(ySeq, O["X₀"][2, :][:], h; w = O["w"]), resL_R);
densN = map(O -> kde(ySeq, O["X₀"][2, :][:], h; w = O["w"]), resN);
densMix = map(O -> kde(ySeq, O["X₀"][2, :][:], h; w = O["w"]), resMix);
densMala = map(O -> kde(ySeq, O.value[2, :][:], h), resMALA);

ii = 1
plot(ySeq, densL[ii], label = "LIMIS");
plot(ySeq, densL_R[ii], label = "LIMIS_R");
plot(ySeq, densN[ii], label = "NIMIS");
plot(ySeq, densMix[ii], label = "GausMix");
plot(ySeq, densMala[ii], label = "MALA");
plot(ySeq, densTRUE, label = "Truth");
legend(loc="top left",fancybox="true");

maccL1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densL);
maccL_R1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densL_R);
maccN1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densN);
maccMix1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densMix);
maccMala1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - densTRUE)), densMala);
hcat(maccL1, maccL_R1, maccN1, maccMix1, maccMala1)

######
## Mean and variance of "Gaussian" dimensions
######

### Means summed across dimensions
ns = length(resL[1]["w"]);
muL = reduce(hcat, map(O -> O["X₀"][3:d, :] * O["w"] / ns, resL));
muL_R = reduce(hcat, map(O -> O["X₀"][3:d, :] * O["w"] / ns, resL_R));
muN = reduce(hcat, map(O -> O["X₀"][3:d, :] * O["w"] / ns, resN));
muMix = reduce(hcat, map(O -> O["X₀"][3:d, :] * O["w"] / ns, resMix));
muMala = reduce(hcat, map(O -> mean(O.value[3:d, :], 2), resMALA));

tmp = hcat(sum(muL, 1)', sum(muL_R, 1)', sum(muN, 1)',
           sum(muMix, 1)', sum(muMala, 1)');

mean(tmp, 1) # Mean estimates
mean(tmp.^2, 1) # MSE
std(tmp, 1) # Standard deviation

### Variances summed across dimensions
function uglyFun(x, w);

  w = w[:];

  d, n = size(x);

  imu = x * w / n;
  ivar = zeros(d);

  for ii = 1:d
   ivar[ii] = dot( (x[ii, :][:] - imu[ii]).^2, w ) / n;
  end

  return ivar;

end

varL = reduce(hcat, map(O -> uglyFun(O["X₀"][3:d, :], O["w"]), resL));
varL_R = reduce(hcat, map(O -> uglyFun(O["X₀"][3:d, :], O["w"]), resL_R));
varN = reduce(hcat, map(O -> uglyFun(O["X₀"][3:d, :], O["w"]), resN));
varMix = reduce(hcat, map(O -> uglyFun(O["X₀"][3:d, :], O["w"]), resMix));
varMala = reduce(hcat, map(O -> var(O.value[3:d, :], 2), resMALA));

tmp = hcat(sum(varL, 1)', sum(varL_R, 1)', sum(varN, 1)',
           sum(varMix, 1)', sum(varMala, 1)');

mean(tmp, 1) # Mean estimates
mean((tmp - (d-2)).^2, 1) # MSE
std(tmp, 1) # Standard deviation
