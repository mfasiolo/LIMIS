##########################################################################################
#################### BANANA IMIS EXAMPLE
##########################################################################################

### TODO
# 1) In IMIS, find this line "add = minimum(dist) > - 0.75;". Where we are checking
# whether the expected ESS of old mixtures wrt the new components is > 0.75. We need
# to write in the text that we do two checks, one before propagating the mixture (which)
# is used also by NIMIS, and one after propagating the mixture (the one I am referring to
# above), which is done only in LIMIS. EDIT: this is not a good idea, because I have to checks
# to determine where to add the mixture component. In the first I cannot use the expESS because
# I have not created the mixture component yet. Hence I don't want to end up with two criteria.

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

##################################################################################
##############   Plotting the importance density
##################################################################################

blas_set_num_threads(1);

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

include("Examples/mixtureBanana.jl");

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

#### 2
niter = 200;
t₀ = 1.; #log( ( (4/(d+2)) ^ (1/(d+4)) * niter ^ -(1/(d+4)) )^2 + 1 );

# Langevin IMIS
resL2 = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0., useLangevin = true, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
            );

#### 3 NIMIS
niter = 200;

# NN IMIS
resL3 = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = true, quant = 0., useLangevin = false, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2
            );

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

n = 100 * d;
n₀ = 1000 * d;

### 1 LIMIS
srand(525);
niter = 400;
t₀ = 3.; #log( ( (4/(d+2)) ^ (1/(d+4)) * niter ^ -(1/(d+4)) )^2 + 1 );

# Langevin IMIS
resL4 = IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = false, quant = 0., useLangevin = true, verbose = true,
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
    d_lik1[iRow, iCol] = dTarget(vcat([x1[iCol] x2[iRow]][:], rep(0., 2-2)))[1];
  end;
end;

for iRow = 1:L
  for iCol = 1:L
    #@printf("%d", iCol);dGausMix(x, μ, Σ; Log = false, df = Inf, w = nothing, dTrans = nothing)
    d_lik2[iRow, iCol] = dGausMix(vcat([x1[iCol] x2[iRow]][:], rep(0., 2-2))'', resL3["μ₀"], resL3["Σ₀"]; df = 3)[1][1];
  end;
end;

for iRow = 1:L
  for iCol = 1:L
    #@printf("%d", iCol);dGausMix(x, μ, Σ; Log = false, df = Inf, w = nothing, dTrans = nothing)
    d_lik3[iRow, iCol] = dGausMix(vcat([x1[iCol] x2[iRow]][:], rep(0., 2-2))'', resL2["μ₀"], resL2["Σ₀"]; df = 3)[1][1];
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
tmp = resL4["μOrig"]';
good = (tmp[:, 1] .> x1[1])[:] & (tmp[:, 1] .< x1[end])[:] & (tmp[:, 2] .> x2[1])[:] & (tmp[:, 2] .< x2[end])[:];
scatter(tmp[good, 1][:], tmp[good, 2][:]);
tmp = resL4["μ₀"]';
good = (tmp[:, 1] .> x1[1])[:] & (tmp[:, 1] .< x1[end])[:] & (tmp[:, 2] .> x2[1])[:] & (tmp[:, 2] .< x2[end])[:];
scatter(tmp[good, 1][:], tmp[good, 2][:], c = "red");

##################################################################################
######################## Monte Carlo Experiment
##################################################################################

#@everywhere cd("$(homedir())/Desktop/All/Dropbox/Work/Liverpool/IMIS/Julia_code");
@everywhere cd("$(homedir())/Dropbox/Work/Liverpool/IMIS/Julia_code");

include("paralSetUp.jl");
@everywhere include("paralSetUp.jl");
@everywhere blas_set_num_threads(1);

nrep = 16;

# Setting seed in parallel
srand(525);
RND = rand(1:1:1000000, nrep);
pmap(x_ -> srand(x_), RND);

### Langevin IMIS
resL = pmap(useless -> IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = false, quant = 0., useLangevin = true, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 1 - 1e-2),
            1:1:nrep);

### NIMIS
resN = pmap(useless -> IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = false,  quant = 0., useLangevin = false, verbose = true, B = Bmult*n),
            1:1:nrep);

### Gaussian Mixture importance sampling
resMix = [];
for ii = 1:nrep
  rtmp = rGausMix(n₀ + n*niter, μMix, ΣMix; df = 3, w = wMix);
  wtmp = dTarget(rtmp; Log = true) - dGausMix(rtmp, μMix, ΣMix; df = 3, w = wMix, Log = true)[1][:];
  ctmp = sumExpTrick(wtmp);
  wtmpD = exp(wtmp - log(ctmp))
  esstmp = ( 1 / sum( wtmpD.^2 ) ) / (n₀ + n*niter);
  resMix = [resMix; Dict{Any,Any}("X₀" => rtmp, "logw" => wtmp, "ESS" => esstmp)];
end

### MALA
resMALA = pmap(nouse -> launchMALAjob(nouse), 1:1:nrep);

################
## Saving results
################

####
# A) Saving all samples
####

### Saving results#
#res = Dict{Any,Any}( "L" => resL, "N" => resN, "Mix" => resMix, "MALA" => resMALA );


#for ii = 1:nrep # Need to do this, otherwise JLD.save crashes
#  res["L"][ii]["control"]["score"] = nothing;
#  res["L"][ii]["control"]["hessian"] = nothing;
#end

# JLD.save("Data/Banana_Mix_5d.jld", "data", res);
#res = load("Data/Banana_Mix_20d.jld")["data"];

#plot(resMALA[1].value[1, :][:])

####
# B) Saving only summaries
####
truX = rBanMix(2e6);

# Marginal density of x₁
@everywhere h = 0.2
@everywhere δ = 0.1;
@everywhere ySeq = -20:δ:20;
densTRUE1 = kde(ySeq, truX[:, 1][:], h);

densL1 = pmap(O -> kde(ySeq, O["X₀"][1, :][:], h; logw = O["logw"]), resL);
densN1 = pmap(O -> kde(ySeq, O["X₀"][1, :][:], h; logw = O["logw"]), resN);
densMix1 = pmap(O -> kde(ySeq, O["X₀"][1, :][:], h; logw = O["logw"]), resMix);
densMala1 = pmap(O -> kde(ySeq, O["value"][1, :][:], h), resMALA);

dens1_res = Dict{Any,Any}( "L" => densL1, "N" => densN1,
                           "Mix" => densMix1, "MALA" => densMala1, "Truth" => densTRUE1);

# Marginal density of x₂
@everywhere h = 0.1;
@everywhere δ = 0.1;
@everywhere ySeq = -11:δ:15;
densTRUE2 = kde(ySeq, truX[:, 2][:], h);

densL2 = pmap(O -> kde(ySeq, O["X₀"][2, :][:], h; logw = O["logw"]), resL);
densN2 = pmap(O -> kde(ySeq, O["X₀"][2, :][:], h; logw = O["logw"]), resN);
densMix2 = pmap(O -> kde(ySeq, O["X₀"][2, :][:], h; logw = O["logw"]), resMix);
densMala2 = pmap(O -> kde(ySeq, O["value"][2, :][:], h), resMALA);

dens2_res = Dict{Any,Any}( "L" => densL2, "N" => densN2,
                           "Mix" => densMix2, "MALA" => densMala2, "Truth" => densTRUE2);

### Means of Gaussian dimensions
muL = reduce(hcat, pmap(O -> WmeanExpTrick(O["X₀"][3:end, :], O["logw"]), resL));
muN = reduce(hcat, pmap(O -> WmeanExpTrick(O["X₀"][3:end, :], O["logw"]), resN));
muMix = reduce(hcat, pmap(O -> WmeanExpTrick(O["X₀"][3:end, :], O["logw"]), resMix));
muMala = reduce(hcat, pmap(O -> mean(O["value"][3:end, :], 2), resMALA));

mean_res = Dict{Any,Any}( "L" => muL, "N" => muN, "Mix" => muMix, "MALA" => muMala);

### Variances of Gaussian dimensions
function uglyFun(x, logw);

  logw = logw[:];

  d, n = size(x);

  imu = WmeanExpTrick(x, logw);
  ivar = zeros(d);

  for ii = 1:d
   ivar[ii] = WmeanExpTrick((x[ii, :][:]' - imu[ii]).^2, logw)[1];
  end

  return ivar;

end

varL = reduce(hcat, map(O -> uglyFun(O["X₀"][3:end, :], O["logw"]), resL));
varN = reduce(hcat, map(O -> uglyFun(O["X₀"][3:end, :], O["logw"]), resN));
varMix = reduce(hcat, map(O -> uglyFun(O["X₀"][3:end, :], O["logw"]), resMix));
varMala = reduce(hcat, map(O -> var(O["value"][3:end, :], 2), resMALA));

var_res = Dict{Any,Any}( "L" => varL, "N" => varN, "Mix" => varMix, "MALA" => varMala);

### Normalizing constant
conL = reduce(hcat, pmap(O -> meanExpTrick(O["logw"]), resL));
conN = reduce(hcat, pmap(O -> meanExpTrick(O["logw"]), resN));
conMix = reduce(hcat, pmap(O -> meanExpTrick(O["logw"]), resMix));

con_res = Dict{Any,Any}( "L" => conL, "N" => conN, "Mix" => conMix);

### Effective Sample Size
ESSL = reduce(hcat, map(O -> O["ESS"][:], resL));
ESSN = reduce(hcat, map(O -> O["ESS"][:], resN));
ESSMix = reduce(hcat, map(O -> O["ESS"], resMix))[:];

ESS_res = Dict{Any,Any}( "L" => ESSL, "N" => ESSN, "Mix" => ESSMix);

## Save all results
all_summaries = Dict{Any,Any}( "d1" => dens1_res, "d2" => dens2_res,
                               "mu" => mean_res, "varG" => var_res, "con" => con_res,
                               "ESS" => ESS_res );

JLD.save("Data/Banana_Mix_5D.jld", "all_summaries", all_summaries);


#############################
# Checking summaries
#############################

# Load data
all_data = load("Data/Banana_Mix_5D.jld")["all_summaries"];

# 1) First marginal
tmp = all_data["d1"];
δ = 0.1;
ySeq = -20:δ:20;

ii = 1
plot(ySeq, tmp["L"][ii], label = "LIMIS");
plot(ySeq, tmp["N"][ii], label = "NIMIS");
plot(ySeq, tmp["Mix"][ii], label = "GausMix");
plot(ySeq, tmp["MALA"][ii], label = "MALA");
plot(ySeq, tmp["Truth"], label = "Truth");
legend(loc="lower center",fancybox="true");

maccL1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - tmp["Truth"])), tmp["L"]);
maccN1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - tmp["Truth"])), tmp["N"]);
maccMix1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - tmp["Truth"])), tmp["Mix"]);
maccMala1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - tmp["Truth"])), tmp["MALA"]);
tmp = Array{Float32}( hcat(maccL1, maccN1, maccMix1, maccMala1) )

mean(tmp, 1)
minimum(tmp, 1)

# 2) Second marginal
tmp = all_data["d2"];
δ = 0.1;
ySeq = -11:δ:15;

ii = 1
plot(ySeq, tmp["L"][ii], label = "LIMIS");
plot(ySeq, tmp["N"][ii], label = "NIMIS");
plot(ySeq, tmp["Mix"][ii], label = "GausMix");
plot(ySeq, tmp["MALA"][ii], label = "MALA");
plot(ySeq, tmp["Truth"], label = "Truth");
legend(loc="lower center",fancybox="true");

maccL1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - tmp["Truth"])), tmp["L"]);
maccN1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - tmp["Truth"])), tmp["N"]);
maccMix1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - tmp["Truth"])), tmp["Mix"]);
maccMala1 = map(_d -> 1 - 0.5 * δ * sum(abs(_d - tmp["Truth"])), tmp["MALA"]);
tmp = hcat(maccL1, maccN1, maccMix1, maccMala1)

tmp = Array{Float32}( hcat(maccL1, maccN1, maccMix1, maccMala1) )

mean(tmp, 1)
minimum(tmp, 1)

# 3) Means summed across Gaussian dimensions
tmp = all_data["mu"];

tmp = hcat(sum(tmp["L"], 1)', sum(tmp["N"], 1)', sum(tmp["Mix"], 1)', sum(tmp["MALA"], 1)')

nrep = size(tmp)[1];
mean(tmp, 1) # Mean estimates
sqrt( mean(tmp.^2, 1) ) # RMSE
diag(cov(tmp) * (nrep-1) / nrep)[:] ./ mean(tmp.^2, 1)[:] # Ratio VAR / MSE

# 4) Variances summed across Gaussian dimensions
tmp = all_data["varG"];

tmp = hcat(sum(tmp["L"], 1)', sum(tmp["N"], 1)', sum(tmp["Mix"], 1)', sum(tmp["MALA"], 1)')

dG = size(all_data["varG"]["L"])[1];
mean(tmp, 1) # Mean estimates
sqrt( mean((tmp - dG).^2, 1) ) # RMSE
diag(cov(tmp) * (nrep-1) / nrep)[:] ./ mean((tmp - dG).^2, 1)[:] # Ratio VAR / MSE

# 5) Normalizing constant estimates
tmp = all_data["con"];

tmp = hcat(tmp["L"]', tmp["N"]', tmp["Mix"]')

mean(tmp, 1) # Mean estimates
sqrt( mean((tmp - 1.).^2, 1) ) # RMSE
(var(tmp, 1) * (nrep-1) / nrep)[:] ./ mean((tmp - 1.).^2, 1)[:] # Ratio VAR / MSE

# 6) Effective Sample Size
ESS = all_data["ESS"];

tmp = hcat(ESS["L"][end, :][:], ESS["N"][end, :][:], ESS["Mix"])

mean(tmp, 1)
minimum(tmp, 1)

fig = figure();
subplot(121);
grid("on");
title("Effective Sample Size (ESS)");
xlabel("Iteration")
ylabel("ESS")
niter = size(ESS["L"])[1];
plot(1:1:(niter), mean(ESS["L"], 2)[:], label = "LIMIS")
plot(1:1:(niter), mean(ESS["N"], 2)[:], label = "NIMIS")
plot(1:1:(niter), rep(mean(ESSMix, 1)[:], niter), label = "GausMix")
legend(loc="lower right",fancybox="true")


#############
## Cost plot
#############

all_data = load("Data/Banana_Mix_5D.jld")["all_summaries"];
ESS1 = all_data["ESS"];
all_data = load("Data/Banana_Mix_20D.jld")["all_summaries"];
ESS2 = all_data["ESS"];
all_data = load("Data/Banana_Mix_80D.jld")["all_summaries"];
ESS3 = all_data["ESS"];

grid("on");
#title("Cost per sample");
xlabel("Iteration number j")
ylabel("log{c(j)}")
plot(0:1:(200), log( (6:1:206) ./ mean(ESS1["L"], 2)[:] ), label = "5D")
plot(0:1:(200), log( (6:1:206) ./ mean(ESS2["L"], 2)[:] ), label = "20D")
plot(0:1:(200), log( (6:1:206) ./ mean(ESS3["L"], 2)[:] ), label = "80D")
legend(loc="top right",fancybox="true")



############################################################################################
############################################################################################
############################################################################################
#####                TUNING T
############################################################################################
############################################################################################
############################################################################################

#@everywhere cd("$(homedir())/Desktop/All/Dropbox/Work/Liverpool/IMIS/Julia_code")
@everywhere cd("$(homedir())/Dropbox/Work/Liverpool/IMIS/Julia_code");

include("paralSetUp.jl");
@everywhere include("paralSetUp.jl");
@everywhere BLAS_set_num_threads(1);

ncores = 4;
nrep = 60;

# Setting seed in parallel
srand(525);
RND = rand(1:1:1000000, ncores);
pmap(x_ -> srand(x_), RND);

@everywhere niter = 50;

nlevs = 40;
timeLevs = linspace(.1, 5, nlevs);

time_out = zeros(nrep, nlevs);
ESSTUNED_out = zeros(nrep, nlevs);
ESSL_out = zeros(nrep, nlevs);

for kk = 1:1:nlevs

  # Langevin IMIS
  resL = pmap(t_ -> IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
              df = 3, trunc = false, quant = 0., useLangevin = true, verbose = false,
              t₀ = t_, score = score, hessian = hessian, targetESS = 1 - 1e-2),
              rep(timeLevs[kk], nrep));

  # Tune t₀
  @everywhere tseq = linspace(2., 5., 20);
  resTune = pmap(x_ -> tuneIMIS(x_, tseq; frac = 0.1, crit = "var", self = false, verbose = true), resL);

  # Extract optimal t₀ and mixture mean and covariances
  tOpt = zeros(nrep);
  μOpt = Array(Float64, d, niter, nrep);
  ΣOpt = Array(Float64, d, d, niter, nrep);
  for ii = 1:1:nrep

    kmin = findfirst( resTune[ii]["expVar"] .== minimum(resTune[ii]["expVar"]) );
    tOpt[ii] = tseq[kmin];
    μOpt[:, :, ii] = resTune[ii]["μ"][:, :, kmin];
    ΣOpt[:, :, :, ii] = resTune[ii]["Σ"][:, :, :, kmin];

  end

  # Try optimized importance mixture
  resMix = [];
  for ii = 1:nrep
    μtmp = hcat(μMix, μOpt[:, :, ii]);
    Σtmp = cat(3, ΣMix, ΣOpt[:, :, :, ii]);
    wMixTmp = wMix * 4 * 10;
    wMixTmp = [wMixTmp; ones(niter)];
    wMixTmp /= sum(wMixTmp);
    rtmp = rGausMix(n₀ + n*niter, μtmp, Σtmp; df = 3, w = wMixTmp);
    wtmp = dTarget(rtmp; Log = true) - dGausMix(rtmp, μtmp, Σtmp; df = 3, w = wMixTmp, Log = true)[1][:];
    ctmp = sumExpTrick(wtmp);
    wtmpD = exp(wtmp - log(ctmp))
    esstmp = ( 1 / sum( wtmpD.^2 ) ) / (n₀ + n*niter);
    resMix = [resMix; Dict{Any,Any}("X₀" => rtmp, "logw" => wtmp, "ESS" => esstmp)];
  end

  # Try INITIAL importance mixture
  resMix0 = [];
  for ii = 1:nrep
    μtmp = hcat(μMix, resL[ii]["μ₀"]);
    Σtmp = cat(3, ΣMix, resL[ii]["Σ₀"]);
    wMixTmp = wMix * 4 * 10;
    wMixTmp = [wMixTmp; ones(niter)];
    wMixTmp /= sum(wMixTmp);
    rtmp = rGausMix(n₀ + n*niter, μtmp, Σtmp; df = 3, w = wMixTmp);
    wtmp = dTarget(rtmp; Log = true) - dGausMix(rtmp, μtmp, Σtmp; df = 3, w = wMixTmp, Log = true)[1][:];
    ctmp = sumExpTrick(wtmp);
    wtmpD = exp(wtmp - log(ctmp))
    esstmp = ( 1 / sum( wtmpD.^2 ) ) / (n₀ + n*niter);
    resMix0 = [resMix0; Dict{Any,Any}("X₀" => rtmp, "logw" => wtmp, "ESS" => esstmp)];
  end

  time_out[:, kk] = tOpt;
  ESSL_out[:, kk] = map(x_ -> x_["ESS"][end], resMix0);
  ESSTUNED_out[:, kk] = map(x_ -> x_["ESS"][end], resMix);

end

tuned = Dict{Any,Any}( "timeOut" => time_out, "timeIn" => timeLevs,
                       "ESSOut" => ESSTUNED_out, "ESSIn" => ESSL_out);

# JLD.save("Data/tuned_t.jld", "tuned", tuned);

tuned = load("Data/tuned_t.jld")["tuned"];

fig = figure();
subplot(121);
grid("on");
#title("T* VS T_0");
xlabel("Initial t")
ylabel("Optimal t")
plot(tuned["timeIn"], mean(tuned["timeOut"], 1)[:]);
plot(tuned["timeIn"], mean(tuned["timeOut"], 1)[:] + std(tuned["timeOut"], 1)[:], color = "red");
plot(tuned["timeIn"], mean(tuned["timeOut"], 1)[:] - std(tuned["timeOut"], 1)[:], color = "red");

subplot(122);
grid("on");
#title("Efficiency");
xlabel("Initial t")
ylabel("EF")
plot(tuned["timeIn"], median(tuned["ESSIn"], 1)[:], label = "Initial EF")
plot(tuned["timeIn"], median(tuned["ESSOut"], 1)[:], label = "Optimized EF")
legend(loc="lower right",fancybox="true")


for ii = 1:nrep plot(timeLevs, time_out[ii, :][:]); end



##################################################################################
##################################################################################
##############   Show mixture contruction process
##################################################################################
##################################################################################

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

######
#
######

blas_set_num_threads(1);

# Mixture of Bananas
d = 2
banDim = copy(d);
bananicity = [0.1, -0.1];
sigmaBan = [6., 6.];
banShiftX = [0., 0.];
banShiftY = [0, 0];
nmix = length(bananicity);
bananaW = [1., 1.]; #ones( nmix ) / nmix #[0.2, 0.6, 0.2]; #;
bananaW = bananaW / sum(bananaW);

include("Examples/mixtureBanana.jl");

x₀ = [14. 3. 6.; 5. 7 -5]
t₀ = 0.;
nt = 100;
δt = 4/nt;

nmix  = size(x₀)[2];

μStore = Array(Float64, d, nmix, nt);
ΣStore = Array(Float64, d, d, nmix, nt);

μM = copy(x₀);
ΣM = nothing;
for ii in 1:1:nt
  μM, ΣM = createLangMix(μM, score, hessian, δt; targetESS = 1-1e-2,
                         Q = nothing, Σ₀ = ΣM)
  μStore[:, :, ii] = μM;
  ΣStore[:, :, :, ii] = ΣM;
end

#### Plotting

L = 150;
x1 = linspace(-15., 15., L);
x2 = linspace(-10., 10., L);

# Define target parameters
d_lik = eye(L);

if( banDim == 2)

 for iRow = 1:L
   for iCol = 1:L
     #@printf("%d", iCol);
     d_lik[iRow, iCol] = dTarget([x1[iCol] x2[iRow]]')[1];
     #if x1[iCol] > 0.   d_lik[iRow, iCol] =  0.; end
  end;
 end;

 contour(x1, x2, d_lik);

end

# Define target parameters
d_lik = eye(L);

for kk = 1:nmix

  plot(μStore[1, kk, :][:], μStore[2, kk, :][:])
  scatter(μStore[1, kk, 1], μStore[2, kk, 1])

 for iRow = 1:L
   for iCol = 1:L
     #@printf("%d", iCol);
     d_lik[iRow, iCol] = dmvt([x1[iCol] x2[iRow]]', μM[:, kk, end], ΣM[:,:, kk, end])[1];
  end;
 end;
 contour(x1, x2, d_lik);
end

xlabel("x")
ylabel("y")
