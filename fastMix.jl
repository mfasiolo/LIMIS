
##############################################################################################
# Calculates product of Gaussian densities
##############################################################################################
# Given the means (μ₁, μ₂) and covariance matrices (Σ₁, Σ₂) of two Gaussians,
# the product of their densities is proportional to a Gaussian density
# (with mean μ = f(μ₁, Σ₁, μ₂, Σ₂) and covariance Σ = g(Σ₁, Σ₂) for some f() and g()) times a normalizing constant
# N(μ₁, μ₂, Σ), where N() is a Guassian pdf.
#
## INPUT
# μ₁, Σ₁, μ₂, Σ₂
## OUTPUT
# A dictionary containing μ, Σ, logc (the log constant)
function prodGaus(μ₁, Σ₁, μ₂, Σ₂)

  Σ = Σ₁ + Σ₂;

  μ = Σ₂ * (Σ \ μ₁) + Σ₁ * (Σ \ μ₂);

  logc = dmvt(μ₁, μ₂, Σ, Inf; Log = true)[1];

  Σ = Σ₁ * (Σ \ Σ₂);

  output = Dict{Any,Any}("μ" => μ, "Σ" => Σ, "logc" => logc)

  return( output );

end

#https://books.google.co.uk/books?id=WneJJEHYHLYC&pg=PA141&lpg=PA141&dq=product+of+multivariate+t+densities&source=bl&ots=5qls1j_E5a&sig=hPmTsAp86hsN9vQywV3ZJaDLuKM&hl=en&sa=X&ved=0CF0Q6AEwCWoVChMIzMCO8IXdxwIVxm0UCh1gggXt#v=onepage&q=product%20of%20multivariate%20t%20densities&f=false
#https://books.google.co.uk/books?id=dmxtU-TxTi4C&pg=PA213&lpg=PA213&dq=multiple+t+distribution&source=bl&ots=kwyGK4nsZG&sig=24SJwfL1TGOBae_WyOjUbIXZPoU&hl=en&sa=X&ved=0CGoQ6AEwCWoVChMI4Zi1sIfdxwIVAtMUCh1siAzB#v=onepage&q=product%20of%20t%20densities&f=false


#################################################################################################################
## Create tree with ϕᵢⱼ (means), μᵢⱼ, Σᵢⱼ
#################################################################################################################
# INPUT
# - μ: the mean vectors of the new mixture components to be added to the tree
# - Σ: the covariance matrices of the new mixture components to be added to the tree
# - ϕ: (optional) a pre-existing tree which will be expanded using the new mixture components
#      with mean in μ and covariances in Σ
# OUTPUT
# - ϕ: a tree (a dictionary) with one entry for each mixture component. Each entry (eg. ϕ[i])
#      is itself a dictionary where the leaf ϕ[i][0] contains the mean and covariance of the i-th mixture
#      component, while ϕ[i][j] with j >= i contains the output of prodGaus(μᵢ, Σᵢ, μⱼ, Σⱼ).
#      Hence the tree is unbalanced, because ϕ[1] has m leafs, ϕ[2] has m-1 leafs...ϕ[2] has 1 leafs, where
#      m is the total number of mixture components
function createϕTree(μ, Σ; ϕ = nothing)

  if(ϕ == nothing) ϕ = Dict{Int, Dict{Int, Dict{Any, Any}}}() end

  pos = length(ϕ) + 1;

  nnew = size(μ, 2);

  # Loop each new mixture components
  for kk = 1:nnew

    # Loop over mix components already in ϕ and calculate ϕᵢₖ and Σᵢⱼₖ
    for ii = keys(ϕ)

      ϕ[ii][pos] = prodGaus(ϕ[ii][0]["μ"], ϕ[ii][0]["Σ"], μ[:, kk], Σ[:, :, kk]);

    end

    # Add "branch" for new mixture component, with leaf index 0 containing μₖ and Σₖ,
    # and leaf index kk containing the output of prodGaus(μₖ, Σₖ, μₖ, Σₖ)
    ϕ[pos] = Dict( 0 => Dict("μ" => μ[:, kk], "Σ" => Σ[:, :, kk]),
                   pos => prodGaus(μ[:, kk], Σ[:, :, kk], μ[:, kk], Σ[:, :, kk]) );

    pos += 1;

  end

  return ϕ;

end


#################################################################################################################
## Create dictionary of arrays with Qᵢⱼ  (covariances)
#################################################################################################################
# INPUT
# - ϕ: a tree (a dictionary) with one entry for each mixture component. Each entry (eg. ϕ[i])
#      is itself a dictionary where the leaf ϕ[i][0] contains the mean and covariance of the i-th mixture
#      component, while ϕ[i][j] with j >= i contains the output of prodGaus(μᵢ, Σᵢ, μⱼ, Σⱼ).
#      Hence the tree is unbalanced, because ϕ[1] has m leafs, ϕ[2] has m-1 leafs...ϕ[2] has 1 leafs, where
#      m is the total number of mixture components
# - μ: the mean vectors of the new mixture components to be added to the tree
# - Σ: the covariance matrices of the new mixture components to be added to the tree
# OUTPUT
# - Q: a list of m upper triagular matrices of dimenstion m x m, where m is the number of mixture components.
#      Entry Q[k][i, j] contains Qᵢⱼₖ = ϕⱼₖ N(μᵢ | μⱼₖ, Σᵢ + Σⱼₖ) - ϕᵢₖϕⱼₖ.
function createQTree(ϕ, μ, Σ)

  m = length(ϕ);

  # Create upper-triangular matrix
  M = Array(Float64, m, m);
  for ir = 1:m
    for ic = ir:m
      M[ir, ic] = 1.0;
    end
  end
  M = sparse( M );

  # Create list of m triangular matrices
  Q = Dict{Int64, SparseMatrixCSC{Float64,Int64}}();
  for ii = 1:m
    Q[ ii ] = copy(M)
  end

  if m > 1

    for kk = 1:m

      for ir = kk:m

        for ic = ir:m

            # Pᵢⱼₖ = ϕⱼₖ N(μᵢ | μⱼₖ, Σᵢ + Σⱼₖ)
            P = exp( getϕ(ϕ, "logc", ic, kk) ) * dmvt(μ[:, ir],
                                                      getϕ(ϕ, "μ", ic, kk),
                                                      Σ[:, :, ir] + getϕ(ϕ, "Σ", ic, kk), Inf)[1];

            # Qᵢⱼₖ = Pᵢⱼₖ - ϕᵢₖϕⱼₖ
            Q[kk][ir, ic] =  P - exp( getϕ(ϕ, "logc", ir, kk) ) * exp( getϕ(ϕ, "logc", ic, kk) );
            Q[ir][kk, ic] =  P - exp( getϕ(ϕ, "logc", kk, ir) ) * exp( getϕ(ϕ, "logc", ic, ir) );
            Q[ic][kk, ir] =  P - exp( getϕ(ϕ, "logc", ir, ic) ) * exp( getϕ(ϕ, "logc", kk, ic) )

          end
      end

    end

  end

  return Q;

end


##############################################################
######## Utilities
##############################################################
# These are functions used to extract particular items from the output
# of createϕTree() or of createQTree()

# Extract μᵢⱼ, Σᵢⱼ or ϕᵢⱼ = logcᵢⱼ (depending of "what") from ϕTree
function getϕ(ϕTree, what, ii, jj)

 return ϕTree[ min(ii, jj) ][ max(ii, jj) ][ what ];

end

# Extract ϕᵢₖ, for k = 1:m, from ϕTree
function getϕVec(ϕTree, kk)

 m = length(ϕTree)

 ϕVec = Array(Float64, m);

 for ii = 1:m

    ϕVec[ii] = getϕ(ϕTree, "logc", ii, kk);

  end

 return ϕVec;

end

# Extract ϕᵢⱼ, for i,j = 1:m, from ϕTree and put them in an m x m matrix
function getϕMat(ϕTree)

  m = length(ϕTree)

  ϕMat = Array(Float64, m, m);

  for ir = 1:m
    for ic = ir:m

      ϕMat[ir, ic] = getϕ(ϕTree, "logc", ic, ir);
      ϕMat[ic, ir] = ϕMat[ir, ic];

    end
  end

  return ϕMat;

end

# Extract Qᵢⱼₖ = Qⱼᵢₖ from Q list of upper-triagular matrices
function getQ(Q, ii, jj, kk)

  return Q[ kk ][min(ii, jj), max(ii, jj)];

end

# Extract Qᵢⱼₖ = Qⱼᵢₖ, for i,j = 1:m, and put them in an m x m matrix
function getQMat(Qtree, kk)

 m = length(Qtree)

 QMat = Array(Float64, m, m);

 for ir = 1:m
    for ic = ir:m

      QMat[ir, ic] = getQ(Qtree, ir, ic, kk);
      QMat[ic, ir] = QMat[ir, ic];

    end
  end

  return QMat;

end



##############################################################################
######### Selects which densities need to be evaluated in a mixture
##############################################################################
# INPUT
# - ϕTree: a tree (a nested Dictionary) which is the output of createϕTree().
#          This contains ϕᵢⱼ, for i, j = 1, ..., m.
# - QTree: a list (dictionary) which is the output of createQTree().
#          This contains Qᵢⱼₖ, for i, j, k = 1, ..., m.
# - α: a tuning parameter in [0, 1], determining the stopping criterion.
# - w: vector of size m, containing the mixture weights
# OUTPUT
# - mixList: a list (dictionary) of m vectors. The i-th vector contains Sⁱᵢₙ, the list of the
#            mixture components that need to be evaluated (not approximated), when a random
#            variable is simulate from the i-th mixture component

function selectMix(ϕTree, QTree, α; w = nothing)

  m = length( ϕTree );

  # If the mixture weights are not given, they are assumed to be equal to 1/m
  if (w == nothing) w = ones(m) / m; end

  # Kronecker product of weights, needed to weight each element of the estimated covariance matrix Qkk
  wM = w * w';

  mixList = Dict{Int64, Array{Int64, 1}}();

  ϕkk = Array(Float64, m);
  Qkk = Array(Float64, m, m);

  # Loop over element of mixList
  for kk = 1:m

    # Extract ϕᵢₖ and Qᵢⱼₖ for i, j = 1, ..., m, (k is fixed) and weight their elements
    ϕkk = exp(getϕVec(ϕTree, kk)) .* w;
    Qkk = getQMat(QTree, kk) .* wM;

    # hat{p}ₖ = ∑ᵢ ϕᵢₖ and MSE(hat{p}ₖ) = var(pₖ) = ∑ᵢ∑ⱼwᵢwⱼQᵢⱼₖ
    denHat = sum( ϕkk );
    varHat = sum( Qkk );

    # iIn = Sᵏᵢₙ is initially empty
    iIn = [];
    rSum = sum(Qkk, 2);

    # Until MSE(hat{p}ₖ) = var(pₖ) is not lower than α * hat{p}ₖ we keep adding
    # the mixture density the leads to the biggest variance reduction to Sᵏᵢₙ
    while sqrt(varHat) > α * denHat

      mx = findfirst( rSum .== maximum(rSum) );
      iIn = [iIn; mx];

      rSum -= Qkk[:, mx];
      rSum[ iIn ] = 0.0;

      varHat = sum( rSum );

      if varHat < 0.0

        @printf("We might be having numerical problems ")
        break;

      end

    end

    mixList[kk] = iIn;

    # @printf("% d \n", length(iIn));

  end

  return mixList;

end

#################################################################################################################
################ Approximately evaluates density of Gaussian mixture
#################################################################################################################
##########
## INPUT
# - x: a (d x n) matrix, each colums is a vector at which the mixture is evaluated
# - xlab: a integer vector of length n, each element indicating to which mixture components each column of x belongs
# - ϕMat: an (m x m) symmetric matrix, containing ϕᵢⱼ = E[ p(x|μᵢ, Σᵢ) | μⱼ, Σⱼ ] = ∫p(x|μᵢ, Σᵢ)p(x|μⱼ, Σⱼ)dx
# - mix: a dictionary of length (m), where the i-th element is an integer vector containg the indexes of the mixture
#        components that need to be evaluated if a random vector has been generated from the i-th mixture component
# - μ: the mean vectors of the mixture components
# - Σ: the covariance matrices of the mixture components

function dFastGausMix(x, xlab, ϕMat, mix, μ, Σ; Log = false, df = Inf, w = nothing)

  d, m = size( μ );

  n = size(x, 2);

  # If the mixture weights are not given, they are assumed to be equal to 1/m
  if (w == nothing) w = ones(m) / m; end

  # Unique labels (corresponding to mixture components) of x
  ul = unique( xlab );

  out = Array(Float64, n);

  # Loop over labels of X (i-th loop evaluates x vectors that were generated form the i-th mixture component)
  for ii = ul

    # Indexes of x vectors belonging to the i-th mixture component
    index = ( xlab .== ii );

    # iI: indexes of mixture components that _NEED_ to be evaluated if x belongs to the i-th mix component
    # iO: ... _DON'T NEED_ ...
    iI = mix[ii];
    iO = setdiff(1:m, mix[ii]);

    # p(x) = ∑ⱼᴼwⱼϕᵢⱼ + ∑ⱼᴵwⱼp(x|μⱼ, Σⱼ)
    out[ index ] = sum( exp( ϕMat[ii, iO] ) * w[iO] ) +
                   dGausMix(x[:, index], μ[:, iI], Σ[:, :, iI]; Log = false, df = df, w = w[iI])[1];

  end

  if Log out = log(out); end

  return out

end


















###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
##################### TRASH
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################

##############################
## Create tree with Qᵢⱼ  (covariances)
##############################

# function growQTree(Q, ϕ)

#   pos = length(Q) + 1;

#   nnew = length(ϕ) - length(Q);

#   if (pos > 1) || (nnew > 1)

#     for kk = 1:nnew

#       Q[pos] = [ pos => Dict{UInt16, Dict{Array{UInt16,1}, Float64}}() ]

#       for i1 = keys(Q)

#         #if pos > i1

#           Q[i1][pos] = [ pos => Dict{Array{UInt16,1}, Float64}() ];

#           for i2 = keys(Q[i1])

#             P = exp(ϕ[i2][pos]["logc"]) * dmvt(ϕ[i1][0]["μ"], ϕ[i2][pos]["μ"], ϕ[i1][0]["Σ"] + ϕ[i2][pos]["Σ"], Inf)[1];

#             # Qᵢⱼₖ = ϕⱼₖ N(μᵢ | μⱼₖ, Σᵢ + Σⱼₖ) - ϕᵢₖϕⱼₖ
#             Q[i1][i2][pos] =  [ [i2; pos; i1] => P - exp(ϕ[i1][i2]["logc"]) * exp(ϕ[i1][pos]["logc"]),
#                                 [i1; pos; i2] => P - exp(ϕ[i1][i2]["logc"]) * exp(ϕ[i2][pos]["logc"]),
#                                 [i1; i2; pos] => P - exp(ϕ[i1][pos]["logc"]) * exp(ϕ[i2][pos]["logc"]) ]

#           end

#         #end

#       end

#       pos += 1;

#     end

#   end

#   return Q;

# end






##############################################################################################
## Estimates mean and covariance of Gaussian mixture evaluations
##############################################################################################
# function covMixEst(μ₀, Σ₀, μ, Σ)

#   d, m = size( μ );

#   # ϕⱼ = E( p(X|μⱼ,Σⱼ) )
#   ϕ = Array(Float64, m);
#   μp = Array(Float64, d, m);
#   Σp = Array(Float64, d, d, m);

#   for jj = 1:m

#     tmp = prodGaus(μ[:, jj], Σ[:, :, jj], μ₀, Σ₀);
#     ϕ[jj] = exp( tmp["logc"] );
#     μp[:, jj] = tmp["μ"];
#     Σp[:, :, jj] = tmp["Σ"];

#   end

#   # Qᵢⱼ = cov( p(X|μᵢ,Σᵢ), p(X|μⱼ,Σⱼ) )
#   Q = Array(Float64, m, m);
#   for ii = 1:m
#     for jj = ii:m

#       Q[ii, jj] = ϕ[jj] * dmvt(μ[:, ii], μp[:, jj], Σ[:, :, ii] + Σp[:, :, jj], Inf)[1] - ϕ[jj] * ϕ[ii];
#       Q[jj, ii] = Q[ii, jj];

#     end
#   end

#   return ϕ, Q;

# end



##############################################################################################
## Approximate Gaussian Mixture density
##############################################################################################
# function dApproxGausMix(x, μ₀, Σ₀, μ, Σ; α = 0.1, Log = false, df = Inf, w = nothing, dTrans = nothing)

#   d, m = size( μ );

#   n = size(x, 2);

#   ϕ, Q = covMixEst(μ₀, Σ₀, μ, Σ);

#   estDens = mean( ϕ );
#   estVar = sum( Q ) / m^2;

#   iIn = [];
#   rSum = sum(Q, 2);

#   while sqrt(estVar) > α * estDens

#    mx = findfirst( rSum .== maximum(rSum) );
#    iIn = [iIn, mx];

#    rSum -= Q[:, mx];
#    rSum[ iIn ] = 0.0;

#    estVar = sum( rSum ) / m^2;

#    #@printf("hey")

#   end

#   for ii = 1:length(iIn) @printf("%d \n", iIn[ii]); end

#   truth = dGausMix(x, μ[:, iIn], Σ[:, :, iIn]; Log = Log, df = df, w = w, dTrans = dTrans)[1];

#   out = ( truth * length(iIn) + sum(ϕ[setdiff(1:m, iIn)]) ) / m;

#   return out;

# end
