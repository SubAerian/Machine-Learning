function [Mu] =  kmeans_init(X, k, init)
%KMEANS_INIT This function computes the initial values of the centroids
%   for k-means algorithm, depending on the chosen method.
%
%   input -----------------------------------------------------------------
%   
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o k     : (double), chosen k clusters
%       o init  : (string), type of initialization {'random','uniform'}
%
%   output ----------------------------------------------------------------
%
%       o Mu    : (N x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N                   
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variable
[N, M] = size(X);

% Output Variable
Mu = zeros(N, k);

switch init
    case 'random'  % Select K datapoints at random from X        
         Mu = datasample(X.', k, 'Replace', false).';
        
    case 'uniform' % Select k datapoints uniformly at random from the range of X
        Mu = bsxfun(@plus, min(X.').', bsxfun(@times, rand(N, k), range(X.').'));
       
    otherwise
        warning('Unexpected initialization type. No centroids computed.')
end

end