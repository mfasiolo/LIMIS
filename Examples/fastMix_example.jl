##########################################################################################
#################### FAST MIXTURE IMPORTANCE SAMPLING
##########################################################################################


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
include("Tests/fastMixTest.jl");


############################################
### Simple example with Gaussian Mixture
############################################
σs = (1:1:10);
n = length(σs);
out = zeros(n, 3);

d = 5;
m = 200;

for ii = 1:1:n
  ## Create a mixture
  μs = Array(Float64, d, m);
  Σs = Array(Float64, d, d, m);
  ws = rand( m );
  ws = ws / sum( ws );

  for jj = 1:m
    μs[:, jj] = rand(Normal(0., σs[ii]), d);
    sigma1 = rand(MvNormal(zeros(d), eye(d)), d);
    Σs[:, :, jj] = sigma1 * sigma1';
  end

  ## Create ϕ and Q trees
  ϕ = createϕTree(μs, Σs);
  Q = createQTree(ϕ, μs, Σs);

  ## Selecting mixture component that need to evaluated at error level α
  α = 0.01;
  goodMix = selectMix(ϕ, Q, α; w = ws);

  nEval = map(x_ -> length(goodMix[x_]), 1:1:m);

  out[ii, :] = [mean(nEval); quantile(nEval, 0.25); quantile(nEval, 0.75)];
end

fig = figure();
grid("on");
xlabel("sigma")
ylabel("|S_in|")
plot(σs, out[:, 1], color = "red");
plot(σs, out[:, 2], color = "blue");
plot(σs, out[:, 3], color = "blue");

#################
# Extract mixture from output of IMIS
d = 5;
m = 100;
μs = Array(Float64, d, m);
Σs = Array(Float64, d, d, m);
ws = rand( m );
ws = ws / sum( ws );

for jj = 1:m
  μs[:, jj] = rand(Normal(0., 5), d);
  sigma1 = rand(MvNormal(zeros(d), eye(d)), d);
  Σs[:, :, jj] = sigma1 * sigma1';
end

## Create ϕ and Q trees
ϕ = createϕTree(μs, Σs);
ϕMat = getϕMat(ϕ);
Q = createQTree(ϕ, μs, Σs);

## Selecting mixture component that need to evaluated at error level α
α = 0.01;
goodMix = selectMix(ϕ, Q, α; w = ws);

# Sample from true mixture
df = Inf
n = 100000
x, labs = rGausMix(n, μs, Σs; df = df, w = ws, labs = true);

# Put labels of each sample in a single vector
tmpLab = mapreduce(ii -> rep(labs[ii, 1], labs[ii, 2]), vcat, 1:size(labs, 1));

# Evaluate approximate and true mixture
@time ap = dFastGausMix(x, tmpLab, ϕMat, goodMix, μs, Σs; Log = false, df = df, w = ws);

@time truth = dGausMix(x, μs, Σs; Log = false, df = df, w = ws, dTrans = nothing)[1];

# Plot a subsample of the mixture evaluations
index = 1:10:n;
fig = figure();
subplot(121);
grid("on");
# title("Normalized Effective Sample Size (NESS)");
xlabel("Approx mixture density")
ylabel("True mixture density")
scatter(ap[index][:], truth[index][:]);
plot(minimum(ap[index]):1e-6:maximum(ap[index]), minimum(ap[index]):1e-6:maximum(ap[index]), "red");

subplot(122);
grid("on");
#title("Normalized Effective Sample Size (NESS)");
xlabel("Approx mixture log-density")
ylabel("True mixture log-density")
scatter(log(ap[index][:]), log(truth[index][:]));
plot(log(minimum(ap[index]):1e-6:maximum(ap[index])), log(minimum(ap[index]):1e-6:maximum(ap[index])), "red");

fig = figure();
subplot(121);
grid("on");
# title("Normalized Effective Sample Size (NESS)");
xlabel("Approx mixture density")
ylabel("True mixture density")
scatter(tmpAp[:]*100000, tmpTruth[:]*100000);
plot(minimum(ap[index]):1e-6:maximum(ap[index]), minimum(ap[index]):1e-6:maximum(ap[index]), "red");

tmpAp = sort(ap[index]);
tmpTruth = sort(truth[index]);

scatter(log(tmpTruth), log(abs(tmpAp[:].-tmpTruth[:])./tmpTruth[:]));
xlabel("True mixture log-density")
ylabel("Log relative error")
