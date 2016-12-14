function [y_est, var_est] = my_gmr(Priors, Mu, Sigma, X, in, out)
% MY_GMR This function performs Gaussian Mixture Regression (GMR), using the 
% parameters of a Gaussian Mixture Model (GMM) for a D-dimensional dataset,
% for D= N+P, where N is the dimensionality of the inputs and P the 
% dimensionality of the outputs.
%
% Inputs -----------------------------------------------------------------
%   o Priors:  1 x K array representing the prior probabilities of the K GMM 
%              components.
%   o Mu:      D x K array representing the centers of the K GMM components.
%   o Sigma:   D x D x K array representing the covariance matrices of the 
%              K GMM components.
%   o x:       N x M array representing M datapoints of N dimensions.
%   o in:      1 x N array representing the dimensions of the GMM parameters
%                to consider as inputs.
%   o out:     1 x P array representing the dimensions of the GMM parameters
%                to consider as outputs. 
% Outputs ----------------------------------------------------------------
%   o y_est:     P x M array representing the retrieved M datapoints of 
%                P dimensions, i.e. expected means.
%   o var_est:   P x P x M array representing the M expected covariance 
%                matrices retrieved. 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
[N,M] = size(X);
[D,k] = size(Mu);
[~,P] = size(out);
y_est = zeros(P,M);
var_est = zeros(P,P,M);

% Compute bêta (k*M)
num = zeros(k,M);
for i = 1:k
    num(i,:) = Priors(i)*my_gaussPDF(X, Mu([1:N],i), Sigma([1:N],[1:N],i));
end
den = sum(num,1);
beta = bsxfun(@rdivide,num,den);


% Compute Sigma_tilde
Sigma_tilde = zeros(P,P,k);
for i = 1:k
    Sigma_tilde(:,:,i) = Sigma([N+1:D],[N+1:D],i) - Sigma([N+1:D],[1:N],i)/Sigma([1:N],[1:N],i)*Sigma([1:N],[N+1:D],i);
end

% Compute y_est (PxM) and var_est (PxPxk)
Mu_tilde = zeros(P,k);
for j = 1:M
    esp = zeros(P,k);
    vari = zeros(P,P,k);
    for i = 1:k
        % Compute Mu_tilde (Pxk)
        Mu_tilde(:,i) = Mu([N+1:D],i) + Sigma([N+1:D],[1:N],i)/Sigma([1:N],[1:N],i)*(bsxfun(@minus, X(:,j), Mu([1:N],i)));
        
        % Intermediate variable for y_est
        esp(:,i) = beta(i,j)*Mu_tilde(:,i);
        
        % Intermediate variable for var_est
        vari(:,:,i) = beta(i,j)*(Mu_tilde(:,i)*Mu_tilde(:,i)' + Sigma_tilde(:,:,i));
    end
    
    % Equation 10
    y_est (:,j) = sum(esp, 2);
    
    % Eqaution 12
    var_est (:,:,j) = sum(vari,3) - y_est(:,j)*y_est (:,j)';
end









