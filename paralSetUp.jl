
# cd("$(homedir())/Desktop/All/Dropbox/Work/Liverpool/IMIS/Julia_code")

using StatsBase;
using Distributions;
using PyPlot;
using Optim;
using Distances;
using Roots;
using HDF5, JLD;
using Klara;

# Loading necessary functions
include("utilities.jl");
include("mvt.jl");
include("createLangMix.jl");
include("IMIS.jl");
include("tuneIMIS.jl");
include("fastMix.jl");

idim = 5;

d = [2; 5; 10; 20; 80][idim];
niter = [200, 200, 200, 200, 200][idim];
# quL = [0.25, 0.4, 0.01, 0.000001, 0.0000000001][idim];
# quN = [0.05, 0.05, 0.15, 0.99, 0.99][idim];
t₀ = [1, 1, 2, 3, 5][idim];
Bmult = [1, 1, 1, 4, 10][idim];
wCov = false;
thins = [1, 1, 1, 1, 1][idim]; # thining for MALA

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

### Set up
#srand(542625);
n = 100 * d;
n₀ = 1000 * d;

# Create importance mixture
μMix = vcat([0. 0. 7 -7; -6. 0 8.2 8.2], zeros(d-2, 4));
ΣMix = zeros(d, d, 4);
for ii = 1:4     ΣMix[:, :, ii] = -2*inv(hessian(μMix[:, ii][:]));     end
wMix = bananaW[1:4] / sum(bananaW[1:4]);

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

  out = Dict{Any,Any}( "value" => out.value, "logtarget" => out.logtarget);

  return out;

end
