function [] =  gmm_eval(X, K_range, repeats, cov_type)
%GMM_EVAL Implementation of the GMM Model Fitting with AIC/BIC metrics.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o repeats  : (1 X 1), # times to repeat k-means
%       o K_range  : (1 X K), Range of k-values to evaluate
%       o cov_type : string ,{'full', 'diag', 'iso'} type of Covariance matrix
%
%   output ----------------------------------------------------------------
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_iter = 0;
AIC_curve = zeros(1, length(K_range));
BIC_curve = zeros(1, length(K_range));

% Populate Curves
for i=1:length(K_range)
    % Select K from K_range
    K = K_range(i);
    
    % First repeats
    [Priors, Mu, Sigma] = my_gmmEM(X, K, cov_type, plot_iter);
    loglik = my_gmmLogLik(X, Priors, Mu, Sigma);
        
    for ii = 2:repeats
        [Priors_i, Mu_i, Sigma_i] = my_gmmEM(X, K, cov_type, plot_iter);
        loglik_i = my_gmmLogLik(X, Priors_i, Mu_i, Sigma_i);
        if loglik_i > loglik
            Priors = Priors_i; Mu = Mu_i; Sigma = Sigma_i;
            loglik = loglik_i;
        end
    end
    
    [AIC_curve(i),BIC_curve(i)]  = gmm_metrics(X, Priors, Mu, Sigma, cov_type);
end


% Plot Metric Curves
figure;
plot(AIC_curve,'--o', 'LineWidth', 1); hold on;
plot(BIC_curve,'--o', 'LineWidth', 1); hold on;
xlabel('K')
legend('AIC', 'BIC')
title(sprintf('GMM (%s) Model Fitting Evaluation metrics',cov_type))
grid on


end