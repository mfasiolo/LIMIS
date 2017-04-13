
# cd("$(homedir())/Desktop/All/Dropbox/Work/Liverpool/IMIS/Julia_code")

using StatsBase;
using Distributions;
using PyPlot;
using Optim;
using Distances;
using Roots;
using HDF5, JLD;
using Klara;
using DataFrames;
using RData;

# Loading necessary functions
include("utilities.jl");
include("mvt.jl");
include("createLangMix.jl");
include("IMIS.jl");
include("tuneIMIS.jl");
include("fastMix.jl");

#idim = 1;

#d = [2; 5; 10; 20; 50][idim];
#niter = [50, 50, 50, 50, 50][idim];
#t₀ = [1, 1, 1, 1, 1][idim];
#Bmult = [1, 1, 1, 2, 5][idim];
#thins = [1, 1, 1, 1, 1][idim]; # thinning for MALA

niter = 100;
t₀ = 1;
Bmult = 20;
wCov = false;
thins = 1;

include("Examples/logistic_regression.jl");

### Set up
#srand(542625);
n = 100 * d;
n₀ = 1000 * d;

# MALA setup
function launchMALAjob(nouse)

  function wrap1(x) dTarget(x; Log = true)[1] end
  function wrap2(x) score(x)[:]; end
  p = BasicContMuvParameter(:p, logtarget = wrap1, gradlogtarget = wrap2)
  model = likelihood_model(p, false);
  sampler = MALA(0.9);
  mcrange = BasicMCRange(nsteps=(n₀ + n*niter), burnin=round(Int, (n₀ + n*niter)/10.), thinning = thins);
  v0 = Dict(:p=>rep(0., d));
  ### Save grad-log-target along with the chain (value and log-target)
  outopts = Dict{Symbol, Any}(:monitor=>[:value, :logtarget], # Adding :gradlogtarget take lots of memory
                              :diagnostics=>[:accept]);

  # Function that launches a MALA job
  job = BasicMCJob(model, sampler, mcrange, v0,
                    tuner=AcceptanceRateMCTuner(0.571, verbose=true),
                    outopts=outopts)

  run(job);

  out = output(job);

  out = Dict{Any,Any}( "value" => out.value,
                       "logtarget" => out.logtarget);

  return out;

end
