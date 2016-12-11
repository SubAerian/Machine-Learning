function [prob] = my_gaussPDF(X, Mu, Sigma)
%MY_GAUSSPDF computes the Probability Density Function (PDF) of a
% multivariate Gaussian represented by a mean and covariance matrix.
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o Mu    : (N x 1), an Nx1 matrix corresponding to the centroid mu_k \in R^Nn
%       o Sigma : (N x N), an NxN matric representing the covariance matrix of the 
%                          Gaussian function
% Outputs ----------------------------------------------------------------
%       o prob  : (1 x M),  a 1xM vector representing the probabilities for each 
%                           M datapoints given Mu and Sigma    
%%
% Initialisation
[N, M] = size(X);
prob = zeros(1,M);

% Compute Probabilities
for i = 1:M
    dist = bsxfun(@minus, X(:,i), Mu);
    prob(i) = exp(-0.5*((dist.')/Sigma)*(dist))/(((2*pi)^(N/2))*sqrt(det(Sigma)));
end


