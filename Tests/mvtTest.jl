################################################################################
# TESTING dmvt( ), maha( )
################################################################################
n = 10000;
df = Inf;
d = 10;
μ = rand(d);
Σ = rand(MvNormal(zeros(d), eye(d)), d);
Σ = Σ * Σ';
x = rmvt(n, μ, Σ, df)

# Auxiliary storage
A = copy(x);
out = Array(Float64, n);

if( mean( abs(dmvt(x, μ, Σ, df; Log = true) - logpdf(MvNormal(μ, Σ), x)) ) > 1e-10 ) error("dmvt not working") else @printf("ok \n") end
if( mean( abs(dmvt(x, μ, Σ, df; Log = true, A = A) - logpdf(MvNormal(μ, Σ), x)) )  > 1e-10 ) error("dmvt not working")  else @printf("ok \n") end
if( mean( abs(dmvt(x, μ, Σ, df; Log = true, out = out) - logpdf(MvNormal(μ, Σ), x)) )  > 1e-10 ) error("dmvt not working")  else @printf("ok \n") end
if( mean( abs(dmvt(x, μ, Σ, df; Log = true, A = A, out = out) - logpdf(MvNormal(μ, Σ), x)) )  > 1e-10 ) error("dmvt not working") else @printf("ok \n") end

@time for ii = 1:100 logpdf!(out, MvNormal(μ, Σ), x); end
@time for ii = 1:100 dmvt(x, μ, Σ, df; Log = false, out = out); end
@time for ii = 1:100 dmvt(x, μ, Σ, df; Log = false, A = A); end
@time for ii = 1:100 dmvt(x, μ, Σ, df; Log = false, A = A, out = out); end

if( mean( abs( maha(x, μ, Σ; isChol = false, A = nothing, out = nothing) - colwise(Mahalanobis(inv(Σ)), x, μ).^2 )) > 1e-10 ) error("maha not working") else @printf("ok \n") end
if( mean( abs(maha(x, μ, Σ; isChol = false, out = out) - colwise(Mahalanobis(inv(Σ)), x, μ).^2 )) > 1e-10 ) error("maha not working") else @printf("ok \n") end
if( mean( abs(maha(x, μ, Σ; isChol = false, A = A) - colwise(Mahalanobis(inv(Σ)), x, μ).^2 )) > 1e-10 ) error("maha not working") else @printf("ok \n") end
if( mean( abs(maha(x, μ, Σ; isChol = false, A = A, out = out) - colwise(Mahalanobis(inv(Σ)), x, μ).^2 )) > 1e-10 ) error("maha not working") else @printf("ok \n") end

@time for ii = 1:100 colwise(Mahalanobis(inv(Σ)), x, μ).^2; end
@time for ii = 1:100 maha(x, μ, Σ; isChol = false, A = nothing, out = nothing); end
@time for ii = 1:100 maha(x, μ, Σ; isChol = false, out = out); end
@time for ii = 1:100 maha(x, μ, Σ; isChol = false, A = A); end
@time for ii = 1:100 maha(x, μ, Σ; isChol = false, A = A, out = out); end

######
# Test dmvt
######
x = [[0.; 0.] [1.; 1.]];
mu = zeros(2);
sigma = [[1; 0.5] [0.5; 2]];

dmvt(x, mu, sigma, 1; Log = true)
#-2.11768 -3.2609

dmvt(x, mu, sigma, 1000; Log = true)
dmvt(x, mu, sigma, Inf; Log = true)



################################################################################
# TESTING rmvt( )
################################################################################

n = 1000000;
d = 5;
mu = rand(Normal(0., 100.), d);
sigma = rand(MvNormal(zeros(d), eye(d)), d);
sigma = sigma * sigma';
#sigma = [[1.; 0.5] [0.5; 2.]];
df = 1000;

x = rmvt(n, mu, sigma, df);

mean(x, 2) - mu
cov(x') - sigma * (df / (df - 2))

# scatter(x[1, 1:1:10000][:], x[2, 1:1:10000][:])

srand(41241);
x = rmvt(n, mu, sigma, Inf);

srand(41241);
y = rmvt(n, mu, sigma, 1000000);

mean( abs(x-y) )

# scatter(x[1, 1:1:10000][:], x[2, 1:1:10000][:])


##############################################################################
## Testing rGausMix(n, μ, Σ; df = Inf, w = nothing, A = nothing)
##############################################################################
## Create a mixture
d = 2;
m = 5;

μs = Array(Float64, d, m);
Σs = Array(Float64, d, d, m);
ws = rand( m );
ws = ws / sum( ws )

for jj = 1:m
  μs[:, jj] = rand(Normal(0., 1.), d);
  sigma1 = rand(MvNormal(zeros(d), eye(d)), d);
  Σs[:, :, jj] = sigma1 * sigma1';
end

n = 1000000
x = rGausMix(n, μs, Σs; df = Inf, w = ws, labs = true);

eprop = x[2][:, 2] / sum(x[2][:, 2]);
if any( (abs(eprop - ws) ./ ws) .> 0.05 )
  error("rGausMix() is not sampling the mixture components with the right weights")
end
