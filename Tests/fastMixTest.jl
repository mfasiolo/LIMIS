################################################################################
# TESTING prodGaus(μ₁, Σ₁, μ₂, Σ₂)
################################################################################
d = 5;

# 100 random tests
for ii = 1:100

  mu1 = rand(Normal(0., 1.), d);
  sigma1 = rand(MvNormal(zeros(d), eye(d)), d);
  sigma1 = sigma1 * sigma1';

  mu2 = rand(Normal(0., 1.), d);
  sigma2 = rand(MvNormal(zeros(d), eye(d)), d);
  sigma2 = sigma2 * sigma2';

  x = rmvt(100, mu1, sigma1, Inf);
  tmp = prodGaus(mu1, sigma1, mu2, sigma2);

  truth = dmvt(x, mu1, sigma1, Inf; Log = true) + dmvt(x, mu2, sigma2, Inf; Log = true);
  est = tmp["logc"] + dmvt(x, tmp["μ"], tmp["Σ"], Inf; Log = true);
  differ =  mean( abs(truth - est) );

  if( differ > (1e-9 * mean(abs(truth))) ) error("prodGaus() seems to have a bug"); end

end



################################################################################
# TESTING createϕTree(μ, Σ; ϕ = nothing); createQTree(ϕ, μ, Σ);
################################################################################
d = 5;
m = 10;

# Create a mixture
μs = Array(Float64, d, m);
Σs = Array(Float64, d, d, m);

for jj = 1:m
  μs[:, jj] = rand(Normal(0., 1.), d);
  sigma1 = rand(MvNormal(zeros(d), eye(d)), d);
  Σs[:, :, jj] = sigma1 * sigma1';
end

# Create ϕ tree
ϕ = createϕTree(μs[:, 1:div(m, 2)], Σs[:, :, 1:div(m, 2)]);
ϕ = createϕTree(μs[:, (div(m, 2)+1):m], Σs[:, :, (div(m, 2)+1):m]; ϕ = ϕ);

# Create Q Tree
Q = createQTree(ϕ, μs, Σs);

n = 100000;
auxMem = Array(Float64, d, n);
for i0 = 1:m

  # Simulate from one density component
  x = rmvt(n, μs[:, i0], Σs[:, :, i0], Inf);

  # Evaluates all densities over the simulation x
  trueDens = zeros(n, m);
  for ii = 1:m
    trueDens[:, ii] = dmvt(x, μs[:, ii], Σs[:, :, ii], Inf; Log = false, A = auxMem)
  end

  # Testing if the estimated expected densities provided of getϕVec are close to the average densities
  truth = log( mean(trueDens, 1) )[:];
  if mean( abs( truth - getϕVec(ϕ, i0) ) ./ abs(truth) ) > 0.01; error("growϕTree() is not accurate enough"); end

  # Comparing true covariances with estimated covariances in Q
  trueCov = cov( (trueDens' .- getϕVec(ϕ, i0))' )
  relErr = abs( trueCov - getQMat(Q, i0) ) ./ abs(cov(trueDens))
  if median(relErr) > 0.05 error("createQTree() is not accurate enough") end

  # Testing if the leafs of ϕ contain the right μ, Σ and logc
  for jj = i0:m

    truth = prodGaus(μs[:, i0], Σs[:, :, i0], μs[:, jj], Σs[:, :, jj]);

    if( mean( abs(ϕ[i0][jj]["μ"] - truth["μ"]) / abs(truth["μ"])) > 1e-6 ) error("growϕTree() seems to have a bug"); end
    if( mean( abs(ϕ[i0][jj]["Σ"] - truth["Σ"]) ./ abs(truth["Σ"])) > 1e-6 ) error("growϕTree() seems to have a bug"); end
    if( abs(ϕ[i0][jj]["logc"] - truth["logc"]) / abs(truth["logc"]) > 1e-6 ) error("growϕTree() seems to have a bug"); end

    if( abs( getϕ(ϕ, "logc", i0, jj) - truth["logc"] ) / abs(truth["logc"]) > 1e-6 ) error("getϕ() seems to have a bug"); end

  end

end




################################################################################
# TESTING selectMix(ϕTree, QTree, α; w = nothing)
################################################################################
d = 10;
m = 15;

μs = Array(Float64, d, m);
Σs = Array(Float64, d, d, m);

for jj = 1:m
  μs[:, jj] = rand(Normal(0., 1.), d);
  sigma1 = rand(MvNormal(zeros(d), eye(d)), d);
  Σs[:, :, jj] = sigma1 * sigma1';
end

μs[:, 6:10] += 20.;
μs[:, 11:15] += 40.;

ϕ = createϕTree(μs, Σs);

Q = createQTree(ϕ, μs, Σs);

goodMix = selectMix(ϕ, Q, 0.00001);

for jj = 0:5:10

  trueSet = Set(jj + (1:5));

  for ii = 1:5

    if Set(goodMix[jj+ii]) != trueSet  error("selectMix() is not selecting the right subset"); end

  end

end


################################################################################
# TESTING dFastGausMix()
################################################################################
## Create a mixture
d = 5;
m = 50;

μs = Array(Float64, d, m);
Σs = Array(Float64, d, d, m);
ws = rand( m );
ws = ws / sum( ws );

for jj = 1:m
  μs[:, jj] = rand(Normal(0., 1.), d);
  sigma1 = rand(MvNormal(zeros(d), eye(d)), d);
  Σs[:, :, jj] = sigma1 * sigma1';
end

## Create ϕ and Q trees
ϕ = createϕTree(μs, Σs);
Q = createQTree(ϕ, μs, Σs);

## Selecting mixture component that need to evaluated at error level α
α = 0.1;
goodMix = selectMix(ϕ, Q, α; w = ws);

ϕMat = getϕMat(ϕ);

## Simulating n random vectors from each component of the mixture and label them
n = 10000;
x = Array(Float64, d, m * n);
xlab = Array(Float64, m * n);

for ii = 1:m

  x[:, ((ii-1)*n+1):(ii*n)] = rmvt(n, μs[:, ii], Σs[:, :, ii], Inf);

  xlab[ ((ii-1)*n+1):(ii*n) ] = ii;

end

## Estimating approximate and true mixture density
@time ap = dFastGausMix(x, xlab, ϕMat, goodMix, μs, Σs; Log = false, df = Inf, w = ws);

@time truth = dGausMix(x, μs, Σs; Log = false, df = Inf, w = ws)[1];

dif = ap - truth;

# Calculating empirical MSE, expected MSE and relative error of the approximate mixture density
relErr = ones( m );
expMSE = ones( m );
empMSE = ones( m );
for ii = 1:m

  # Indices Mixture components that will be approximated, not evaluated, when sampling from the ii-th component
  excluded = setdiff(1:m, goodMix[ii]);

  # Empirical MSE
  empMSE[ii] = mean( dif[((ii-1)*n+1):(ii*n)].^2 );

  # Expected MSE
  expMSE[ii] = sum( (getQMat(Q, ii) .* (ws * ws') )[excluded, excluded] )

  # Root-MSE( p(x) ) / E( p(x) )
  relErr[ii] = sqrt( empMSE[ii] ) / sum(ws .* exp(ϕMat[:, ii]));

end

if (mean( abs(empMSE - expMSE) ./ empMSE ) > 0.05) error("the empirical MSE is much different form the expected MSE") end

if any(relErr .> 1.2 * α) error("dFastGausMix() is making a relative error bigger that α") end
if (mean(relErr) > 1.1 * α) error("dFastGausMix() is making very big relative errors given α") end
if (mean(relErr) < 0.8 * α) error("dFastGausMix() is making very small relative errors given α") end
