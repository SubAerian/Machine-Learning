function [labels] =  gmm_cluster(X, Priors, Mu, Sigma, type, softThresholds)
%GMM_CLUSTER Computes the cluster labels for the data points given the GMM
%
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                           Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o type   : string ,{'hard', 'soft'} type of clustering
%
%       o softThresholds: (2 x 1), a vecor for the minimum and maximum of
%                           the threshold for soft clustering in that order
%
%   output ----------------------------------------------------------------
%
%       o labels   : (1 x M), a M dimensional vector with the label of the
%                             cluster for each datapoint
%                             - For hard clustering, the label is the 
%                             cluster number.
%                             - For soft clustering, the label is 0 for 
%                             data points which do not have high confidnce 
%                             in cluster assignment
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[N,M] = size(X);
[~, K] = size(Mu);
labels = zeros(1,M);

% Find the a posteriori probability for each data point for each cluster
% 1) Compute probabilities p(x^i|k)
for k = 1:K
    p_xi_k(k, :) = Priors(k) .* my_gaussPDF(X, Mu(:,k), Sigma(:,:,k));
end
% 2) Compute posterior probabilities p(k|x)
prob = sum(p_xi_k,1);
for k = 1:K
    p_xi_k(k, :) = p_xi_k(k, :) ./ prob;
end

% Use posterior probabilities to assign points to clusters based on
% clustering method 'hard' or 'soft'
switch (type)
    case 'hard'
        % Find the cluster with highest probability
        [~,labels] = max(p_xi_k);
        
    case 'soft'
        % Find the cluster with highest probabilty. Unless, the highest
        % and another cluster are in the same range specified by
        % threshold
        
        for ii = 1:M  
            a = 0;
            maxi = max(p_xi_k(:, ii));
            if softThresholds(1) < maxi && maxi < softThresholds(2)
                for k = 1:K
                    if softThresholds(1) < p_xi_k(k,ii) && p_xi_k(k,ii) < softThresholds(2)
                        a = a+1;
                    end
                end
            end
            if a <= 1
                [~,labels(ii)] = max(p_xi_k(:, ii));
            end
        end
          
    otherwise
        fprintf('Invalid type for clustering\n');
        
end
end

