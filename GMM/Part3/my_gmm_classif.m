function [y_est] = my_gmm_classif(X_test, models, labels, K, P_class)
%MY_GMM_CLASSIF Classifies datapoints of X_test using ML Discriminant Rule
%   input------------------------------------------------------------------
%
%       o X_test    : (N x M_test), a data set with M_test samples each being of
%                           dimension N, each column corresponds to a datapoint.
%       o models    : (1 x N_classes) struct array with fields:
%                   | o Priors : (1 x K), the set of priors (or mixing weights) for each
%                   |            k-th Gaussian component
%                   | o Mu     : (N x K), an NxK matrix corresponding to the centroids
%                   |            mu = {mu^1,...mu^K}
%                   | o Sigma  : (N x N x K), an NxNxK matrix corresponding to the
%                   |            Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o labels    : (1 x N_classes) unique labels of X_test.
%       o K         : (1 x 1) number K of GMM components.
%   optional---------------------------------------------------------------
%       o P_class   : (1 x N_classes), the vector of prior probabilities
%                      for each class i, p(y=i). If provided, equal class
%                      distribution assumption is no longer made.
%
%   output ----------------------------------------------------------------
%       o y_est  :  (1 x M_test), a vector with estimated labels y \in {0,...,N_classes}
%                   corresponding to X_test.
%%
[~,M_test] = size(X_test);
[~,N_class] = size(models);
loglik_ = zeros(N_class, M_test);

for c = 1:N_class
    % Initialization
    prob = zeros(K,M_test);
    
    %Compute the likelihood of each datapoint
    for i = 1:K
        prob(i,:) = models(c).Priors(i)*my_gaussPDF(X_test, models(c).Mu(:,i), models(c).Sigma(:,:,i));
    end
    
    %Compute the total log likelihood
     if exist('P_class', 'var') == 0
        % Uniform distribution
        loglik_(c,:) = -log(sum(prob,1));
     else
         loglik_(c,:) = -log(sum(prob,1)*P_class(c));        
     end
end

% We take only the indice of the minimum 
[~, y_est] = min(loglik_);

% Modify y_est to match with the labels
y_est = y_est - 1;

end