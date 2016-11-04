function [X_hat ] = compress(X, Mu, V, Percent)
%COMPRESS Summary of this function goes here
%   
%   
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%       o V      : (N x N), Eigenvector Matrix from PCA.
%       o Mu     : (N x 1), Mean Vector of Dataset.
%       o Percent      : scalar, Compression percentage (<1).
%
%   output ----------------------------------------------------------------
%
%       o X_hat  : (N x M), reconstructed data set with M samples each being of dimension N.
%  

N = size(X);

% Project Data to Choosen Principal Components
p = ceil((1-Percent)*N(1));
[A_p, Y] = project_pca(X, Mu, V, p);

[X_hat]  = reconstruct_pca(Y, A_p, Mu);

end

