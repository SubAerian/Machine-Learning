function [RSS, AIC, BIC] =  my_metrics(X, labels, Mu)
%MY_METRICS Computes the metrics for clustering evaluation
%
%   input -----------------------------------------------------------------
%   
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Mu    : (N x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N
%
%   output ----------------------------------------------------------------
%
%       o d      : distance between x_1 and x_2 depending on distance
%                  type {'L1','L2','LInf'}
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variables
[D, M] = size(X);
[~, K] = size(Mu);
RSS = 0;


% Compute RSS (Equation 8)
for jj=1:M
    RSS = RSS + my_distance(X(:,jj),Mu(:,labels(jj)), 'L2')^2;
end
  
% Compute AIC (Equation 9)
AIC = RSS + 2*K*D;

% Compute AIC (Equation 10)
BIC = RSS + log(M)*K*D;


end