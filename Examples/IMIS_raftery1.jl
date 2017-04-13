
##################################################################################
######################## Monte Carlo Experiment
##################################################################################

#@everywhere cd("$(homedir())/Desktop/All/Dropbox/Work/Liverpool/IMIS/Julia_code");
@everywhere cd("$(homedir())/Dropbox/Work/Liverpool/IMIS/Julia_code");

include("paralSetUp_raftery1.jl");
@everywhere include("paralSetUp_raftery1.jl");
@everywhere BLAS.set_num_threads(1);

nrep = 16;

# Setting seed in parallel
srand(525);
RND = rand(1:1:1000000, nrep);
pmap(x_ -> srand(x_), RND);

times = zeros(4);
### Langevin IMIS
tic();
resL = pmap(useless -> IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = false, quant = 0., useLangevin = true, verbose = true,
            t₀ = t₀, score = score, hessian = hessian, targetESS = 0.99),
            1:1:nrep);
times[1] = toc();

### NIMIS
tic();
resN = pmap(useless -> IMIS(niter, n, n₀, dTarget, dPrior, rPrior;
            df = 3, trunc = false,  quant = 0., useLangevin = false, verbose = true,
            B = Bmult*n, wCov = true),
            1:1:nrep);
times[2] = toc();

### Gaussian importance sampling
tic()
resMix = [];
for ii = 1:nrep
  rtmp = rmvt(n₀ + n*niter, μ_I, Σ_I, 3);
  wtmp = dTarget(rtmp; Log = true) - dmvt(rtmp, μ_I, Σ_I, 3; Log = true)[:];
  ctmp = sumExpTrick(wtmp);
  wtmpD = exp(wtmp - log(ctmp))
  esstmp = ( 1 / sum( wtmpD.^2 ) ) / (n₀ + n*niter);
  resMix = [resMix; Dict{Any,Any}("X₀" => rtmp, "logw" => wtmp, "ESS" => esstmp)];
end
times[3] = toc();

### MALA
tic()
resMALA = pmap(nouse -> launchMALAjob(nouse), 1:1:nrep);
times[4] = toc()

### Benchmark: massive IS
mss = pmap(O -> mean(O["X₀"]', WeightVec(exp(O["logw"])), 1), resL);
mss = reduce(+, mss) / nrep;
cvs = map(O -> cov(O["X₀"]', WeightVec(exp(O["logw"]))), resL);
cvs = reduce(+, cvs) / nrep;
nBench = 100000;
nTOT = nBench*nrep;
mySetUp = map(a-> Dict{Any,Any}("nBench" => nBench, "m" => mss, "vr" => cvs, "X" => 0 ), 1:1:nrep);
rtmp = pmap(O -> rmvt(O["nBench"], O["m"], O["vr"], 3), mySetUp);
for ii = 1:nrep    mySetUp[ii]["X"] = rtmp[ii]     end;
wtmp = pmap(O -> dTarget(O["X"]; Log = true) - dmvt(O["X"], O["m"], O["vr"], 3; Log = true)[:],
            mySetUp);
rtmp = reduce(hcat, rtmp);
wtmp = reduce(vcat, wtmp)[:];
ctmp = sumExpTrick(wtmp);
wtmp = exp(wtmp - log(ctmp));
esstmp = ( 1 / sum( wtmp.^2 ) ) / nTOT;
resBench = Dict{Any,Any}("X₀" => rtmp, "logw" => log(wtmp), "ESS" => esstmp, "c" => ctmp);

ESSL = reduce(hcat, map(x_ -> x_["ESS"][:], resL) )';
ESSN = reduce(hcat, map(x_ -> x_["ESS"][:], resN) )';
ESSMix = reduce(hcat, map(x_ -> x_["ESS"], resMix) )';
ESSMALA = reduce(hcat, pmap(_x -> mean(ess(_x["value"], 2)), resMALA) )'/ size(resMALA[1]["value"])[2];

plot((1:1:(niter+1))[:], mean(ESSL, 1)[:])
plot((1:1:(niter+1))[:], mean(ESSN, 1)[:])
plot((1:1:(niter+1))[:], zeros(niter+1)+mean(ESSMALA))
plot((1:1:(niter+1))[:], zeros(niter+1)+mean(ESSMix))

####################################################################################################
####
# B) Saving only summaries
####
####################################################################################################
### 1) Marginal Means
muBench = WsumExpTrick( resBench["X₀"], resBench["logw"] );
muL = reduce(hcat, map(O -> WsumExpTrick(O["X₀"],
                       O["logw"] - log(sumExpTrick(O["logw"]))), resL));
muN = reduce(hcat, map(O -> WsumExpTrick(O["X₀"],
                       O["logw"] - log(sumExpTrick(O["logw"]))), resN));
muMix = reduce(hcat, map(O -> WsumExpTrick(O["X₀"],
                         O["logw"] - log(sumExpTrick(O["logw"]))), resMix));
muMala = reduce(hcat, map(O -> mean(O["value"], 2), resMALA));

mu_res = Dict{Any,Any}( "B" => muBench, "L" => muL, "N" => muN,
                        "Mix" => muMix, "MALA" => muMala);

### 2) Marginal variances
function uglyFun(x, logw);

  logw = logw[:];

  d, n = size(x);

  imu = WsumExpTrick(x, logw);
  ivar = zeros(d);

  for ii = 1:d
   ivar[ii] = WsumExpTrick((x[ii, :][:]' - imu[ii]).^2, logw)[1];
  end

  return ivar;

end

varBench = uglyFun(resBench["X₀"], resBench["logw"]);
varL = reduce(hcat, map(O -> uglyFun(O["X₀"], O["logw"] - log(sumExpTrick(O["logw"]))), resL));
varN = reduce(hcat, map(O -> uglyFun(O["X₀"], O["logw"] - log(sumExpTrick(O["logw"]))), resN));
varMix = reduce(hcat, map(O -> uglyFun(O["X₀"], O["logw"] - log(sumExpTrick(O["logw"]))), resMix));
varMala = reduce(hcat, map(O -> var(O["value"], 2), resMALA));

var_res = Dict{Any,Any}( "B" => varBench, "L" => varL, "N" => varN,
                        "Mix" => varMix, "MALA" => varMala);

### 3) Log Normalizing constant
conBench =  log( resBench["c"] ) - log(nTOT);
conL = reduce(hcat, map(O -> log(meanExpTrick(O["logw"])), resL));
conN = reduce(hcat, map(O -> log(meanExpTrick(O["logw"])), resN));
conMix = reduce(hcat, map(O -> log(meanExpTrick(O["logw"])), resMix));

con_res = Dict{Any,Any}( "B" => conBench, "L" => conL, "N" => conN, "Mix" => conMix);


### 4) Efficiencies
ESS_res = Dict{Any,Any}("L" => ESSL, "N" => ESSN, "Mix" => ESSMix, "MALA" => ESSMALA);

all_summaries = Dict{Any,Any}( "mu" => mu_res, "var" => var_res,
                               "con" => con_res, "ESS" => ESS_res );

JLD.save("Data/raftery1_summaries_1.jld", "all_summaries", all_summaries);


#############################
# Checking summaries
#############################

# Load data
all_data = load("Data/raftery1_summaries_1.jld")["all_summaries"];

#### 1) Marginal posterior means

res = all_data["mu"];

tmp = res["L"] .- res["B"];
muMSE_L = mean(tmp.^2, 1);
muSD_L = sqrt(diag( cov(tmp) * (nrep-1)/nrep ) );

tmp = res["N"] .- res["B"];
muMSE_N = mean(tmp.^2, 1);
muSD_N = sqrt(diag( cov(tmp) * (nrep-1)/nrep ) );

tmp = res["Mix"] .- res["B"];
muMSE_Mix = mean(tmp.^2, 1);
muSD_Mix = sqrt(diag( cov(tmp) * (nrep-1)/nrep ) );

tmp = res["MALA"] .- res["B"];
muMSE_MALA = mean(tmp.^2, 1);
muSD_MALA = sqrt(diag( cov(tmp) * (nrep-1)/nrep ) );

mu_MSE_ALL = hcat(muMSE_L', muMSE_N', muMSE_Mix', muMSE_MALA')
mu_SD_ALL = hcat(muSD_L, muSD_N, muSD_Mix, muSD_MALA)

sqrt( mean(mu_MSE_ALL, 1) ) # RMSE
mean(mu_SD_ALL, 1)          # Standard Deviation
mean(mu_SD_ALL.^2, 1) ./ mean(mu_MSE_ALL, 1) # Ratio VAR / MSE

#### 2) Marginal posterior standard deviations

res = all_data["var"];

tmp = sqrt(res["L"]) .- sqrt(res["B"]);
sdMSE_L = mean(tmp.^2, 1);
sdSD_L = sqrt(diag( cov(tmp) * (nrep-1)/nrep ) );

tmp = sqrt(res["N"]) .- sqrt(res["B"]);
sdMSE_N = mean(tmp.^2, 1);
sdSD_N = sqrt(diag( cov(tmp) * (nrep-1)/nrep ) );

tmp = sqrt(res["Mix"]) .- sqrt(res["B"]);
sdMSE_Mix = mean(tmp.^2, 1);
sdSD_Mix = sqrt(diag( cov(tmp) * (nrep-1)/nrep ) );

tmp = sqrt(res["MALA"]) .- sqrt(res["B"]);
sdMSE_MALA = mean(tmp.^2, 1);
sdSD_MALA = sqrt(diag( cov(tmp) * (nrep-1)/nrep ) );

sd_MSE_ALL = hcat(sdMSE_L', sdMSE_N', sdMSE_Mix', sdMSE_MALA')
sd_SD_ALL = hcat(sdSD_L, sdSD_N, sdSD_Mix, sdSD_MALA)

sqrt( mean(sd_MSE_ALL, 1) ) # RMSE
mean(sd_SD_ALL, 1)          # Standard Deviation
mean(sd_SD_ALL.^2, 1) ./ mean(sd_MSE_ALL, 1) # Ratio VAR / MSE


#### 3) Normalizing constants
res = all_data["con"];

tmp = res["L"] .- res["B"];
conMSE_L = mean(tmp.^2, 1);
conSD_L = sqrt(var(tmp) * (nrep-1)/nrep);

tmp = res["N"] .- res["B"];
conMSE_N = mean(tmp.^2, 1);
conSD_N = sqrt(var(tmp) * (nrep-1)/nrep);

tmp = res["Mix"] .- res["B"];
conMSE_Mix = mean(tmp.^2, 1);
conSD_Mix = sqrt(var(tmp) * (nrep-1)/nrep);

con_MSE_ALL = hcat(conMSE_L', conMSE_N', conMSE_Mix')
con_SD_ALL = hcat(conSD_L, conSD_N, conSD_Mix)

sqrt( mean(con_MSE_ALL, 1) ) # RMSE
con_SD_ALL                   # Standard Deviation
con_SD_ALL.^2 ./ mean(con_MSE_ALL, 1) # Ratio VAR / MSE

#### 4) Efficiencies
ESS = all_data["ESS"];

tmp = hcat(ESS["L"][:, end][:], ESS["N"][:, end][:], ESS["Mix"], ESS["MALA"])

mean(tmp, 1)
std(tmp, 1)
minimum(tmp, 1)

fig = figure();
#subplot(122);
grid("on");
title("Effective Sample Size (ESS)");
xlabel("Iteration")
ylabel("ESS")
plot(0:1:(niter), mean(ESS["L"], 1)[:], label = "LIMIS")
plot(0:1:(niter), mean(ESS["N"], 1)[:], label = "NIMIS")
plot(0:1:(niter), rep(mean(ESS["Mix"], 1)[:], niter+1), label = "GausMix")
plot(0:1:(niter), rep(mean(ESS["MALA"][:]), niter+1) , label = "MALA")
legend(loc="lower right",fancybox="true")
