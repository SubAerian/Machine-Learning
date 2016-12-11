function [  Priors, Mu, Sigma ] = my_gmmEM(X, K, cov_type, plot_iter)
%MY_GMMEM Computes maximum likelihood estimate of the parameters for the 
% given GMM using the EM algorithm and initial parameters
%   input------------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of 
%                           dimension N, each column corresponds to a datapoint.
%       o K         : (1 x 1) number K of GMM components.
%       o cov_type  : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%       o plot_iter : (bool)  set to 1 of want to visalize initual Mu's and
%                          Sigma's, works only for N=2
%       o verb      : (bool)  set to 1 of want to see the convergence output
%   output ----------------------------------------------------------------
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                       Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%%
% Parameters
[N,M] = size(X);

%%%%%% STEP 1: Initialization of Priors, Means and Covariances %%%%%%
[Priors_i, Mu_i, Sigma_i] = my_gmmInit(X, K, cov_type, plot_iter);
log_i = my_gmmLogLik(X, Priors_i, Mu_i, Sigma_i);

% This line is only use to initialise dimensions of Mu and Sigma
Mu = Mu_i;
Sigma = Sigma_i;

while true
   
    %%%%%% STEP 2: Expectation Step: Membership probabilities %%%%%%
    
    % 1) Compute probabilities p(x^i|k)
    for k = 1:K
        p_xi_k(k, :) = Priors_i(k) .* my_gaussPDF(X, Mu_i(:,k), Sigma_i(:,:,k));
    end
    % 2) Compute posterior probabilities p(k|x)
    prob = sum(p_xi_k,1);
    for k = 1:K
        p_xi_k(k, :) = p_xi_k(k, :) ./ prob;
    end
    
    %%%%%% STEP 3: Maximization Step: Update Priors, Means and Sigmas %%%%%%    

    % 1) Update Priors
    Priors = (sum(p_xi_k,2)/M)';
    
    % 2) Update Means and Covariance Matrix
    for k=1:K        
        % Update Means
        prob1 = sum(p_xi_k(k,:));
        if prob1 == 0
            Mu(:,k) = Mu_i(:,k);
        else
            Mu(:,k) = sum(bsxfun(@times,p_xi_k(k,:),X),2)/prob1;
        end
        % Update Covariance Matrices 
        switch cov_type
            case 'full'
                num = bsxfun(@times, p_xi_k(k,:), bsxfun(@minus, X, Mu(:,k)))*bsxfun(@minus, X, Mu(:,k))';
                Sigma(:,:,k) = num/sum(p_xi_k(k,:));
                
            case 'diag'
                num = bsxfun(@times, p_xi_k(k,:), bsxfun(@minus, X, Mu(:,k)))*bsxfun(@minus, X, Mu(:,k))';
                Sigma(:,:,k) = num/sum(p_xi_k(k,:));
                Sigma(:,:,k) = diag(diag(Sigma(:,:,k)));
                
                % If it's not diag, not full, or the string is wrong (!= 'iso'),
                % we consider as an iso covariance Matrix initialisation
            otherwise
                num = sum(p_xi_k(k,:).*(my_distX2Mu(X, Mu(:,k), 'L2').^2));
                Sigma(:,:,k) = (num/(N*sum(p_xi_k(k,:)))) * eye(N); 
        end  
        
        % Add a tiny variance to avoid numerical instability
        epsilon = 1e-5;
        Sigma(:,:,k) = Sigma(:,:,k) + diag(repmat(epsilon, 1, N));
        
    end    
    
    %%%%%% Stopping criterion %%%%%%
    threshold = 1e-10;
    logar = my_gmmLogLik(X, Priors, Mu, Sigma);
    if abs(log_i - logar) < threshold
        break;
    else
        Mu_i = Mu; Sigma_i = Sigma; Priors_i = Priors; log_i = logar;
    end
        
    
end

%%%%%% Visualize Final Estimates %%%%%%
if (N==2 && plot_iter==1)
options.labels      = [];
options.class_names = {};
options.plot_figure = false;

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
colors     = hsv(K);
ml_plot_centroid(Mu',colors);hold on; 
plot_gmm_contour(gca,Priors,Mu,Sigma,colors);
title('Final GMM Parameters');
grid on; box on;

end

