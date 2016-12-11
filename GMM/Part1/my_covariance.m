function [ Sigma ] = my_covariance( X, Mu, type )
%MY_COVARIANCE computes the covariance matrix of X given a covariance type.
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o Mu    : (N x 1), an Nx1 matrix corresponding to the centroid mu_k \in R^Nn
%       o type  : string , type={'full', 'diag', 'iso'} of Covariance matrix
%
% Outputs ----------------------------------------------------------------
%       o Sigma : (N x N), an NxN matrix representing the covariance matrix of the 
%                          Gaussian function
%%
% Initialisation
[N,M] = size(X);
Sigma = zeros(N,N);

% Demean X
Xd = bsxfun(@minus,X,Mu);

switch(type)
    case 'full' % initialize a full covariance Matrix
        Sigma = (1/(M-1))*Xd*(Xd');
        
    case 'diag' % initialize a diagonal covariance Matrix
        Sig = (1/(M-1))*Xd*(Xd');
        Sigma = diag(diag(Sig));
       
    % If it's not diag, not full, or the string is wrong (!= 'iso'),
    % we consider as an iso covariance Matrix initialisation
    otherwise 
        Sig = (1/(N*M))*sum(my_distX2Mu(X, Mu, 'L2').^2);
        Sigma = diag(repmat(Sig, 1, N));
        
end
        
        
end

