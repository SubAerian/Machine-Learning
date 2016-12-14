function [ MSE_F_fold, NMSE_F_fold, R2_F_fold, AIC_F_fold, BIC_F_fold, std_MSE_F_fold, ...,
    std_NMSE_F_fold, std_R2_F_fold, std_AIC_F_fold, std_BIC_F_fold] = cross_validation_gmr( X, y, ...,
    cov_type, plot_iter, F_fold, tt_ratio, k_range )
%CROSS_VALIDATION_REGRESSION Implementation of F-fold cross-validation for regression algorithm.
%
%   input -----------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y         : (P x M) array representing the y vector assigned to
%                           each datapoints
%       o F_fold    : (int), the number of folds of cross-validation to compute.
%       o tt_ratio  : (double), Training/Testing Ratio.
%       o k_range   : (1 x K), Range of k-values to evaluate
%
%   output ----------------------------------------------------------------
%
%       o MSE_F_fold      : (1 x K), Mean Squared Error computed for each value of k averaged over the number of folds.
%       o NMSE_F_fold     : (1 x K), Normalized Mean Squared Error computed for each value of k averaged over the number of folds.
%       o R2_F_fold       : (1 x K), Coefficient of Determination computed for each value of k averaged over the number of folds.
%       o AIC_F_fold      : (1 x K), Mean AIC Scores computed for each value of k averaged over the number of folds.
%       o BIC_F_fold      : (1 x K), Mean BIC Scores computed for each value of k averaged over the number of folds.
%       o std_MSE_F_fold  : (1 x K), Standard Deviation of Mean Squared Error computed for each value of k.
%       o std_NMSE_F_fold : (1 x K), Standard Deviation of Normalized Mean Squared Error computed for each value of k.
%       o std_R2_F_fold   : (1 x K), Standard Deviation of Coefficient of Determination computed for each value of k averaged over the number of folds.
%       o std_AIC_F_fold  : (1 x K), Standard Deviation of AIC Scores computed for each value of k averaged over the number of folds.
%       o std_BIC_F_fold  : (1 x K), Standard Deviation of BIC Scores computed for each value of k averaged over the number of folds.
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
AIC_F_fold = zeros(1, length(k_range));
BIC_F_fold = zeros(1, length(k_range));
MSE_F_fold = zeros(1, length(k_range));
NMSE_F_fold = zeros(1, length(k_range));
R2_F_fold = zeros(1, length(k_range));
std_AIC_F_fold = zeros(1, length(k_range));
std_BIC_F_fold = zeros(1, length(k_range));
std_MSE_F_fold = zeros(1, length(k_range));
std_NMSE_F_fold = zeros(1, length(k_range));
std_R2_F_fold = zeros(1, length(k_range));
N = size(X,1); P = size(y,1);
in  = 1:N;       % input dimensions
out = N+1:(N+P); % output dimensions


for i=1:length(k_range)
    
    % Select K from K_range
    K = k_range(i);
    std_MSE = zeros(1, F_fold);
    std_NMSE = zeros(1, F_fold);
    std_R2 = zeros(1, F_fold);
    std_AIC = zeros(1, F_fold);
    std_BIC = zeros(1, F_fold);
    
    % Compute the model
    for ii = 1:F_fold
        
        % Split data into a training and testing set
        [X_train, y_train, X_test, y_test] = split_data(X, y, tt_ratio );
        
        % Calculation of MSE NMSE Rsquared on the testing set
        C = vertcat(X_train, y_train);
        [Priors, Mu, Sigma] = my_gmmEM(C, K, cov_type, plot_iter);
        
        [AIC,BIC]  = gmm_metrics(C, Priors, Mu, Sigma, cov_type);
 
        % Use later to Compute the Mean
        AIC_F_fold(i) = AIC_F_fold(i) + AIC;
        BIC_F_fold(i) = BIC_F_fold(i) + BIC;
        
        % Standard deviation
        std_AIC(ii) = AIC;
        std_BIC(ii) = BIC;
        
        [y_est, var_est] = my_gmr(Priors, Mu, Sigma, X_test, in, out);
        [MSE, NMSE, R2] = my_regression_metrics( y_est, y_test );
        
        % Use later to Compute the Mean
        MSE_F_fold(i) = MSE_F_fold(i) + MSE;
        NMSE_F_fold(i) = NMSE_F_fold(i) + NMSE;
        R2_F_fold(i) = R2_F_fold(i) + R2;
        
        % Standard deviation
        std_MSE(ii) = MSE;
        std_NMSE(ii) = NMSE;
        std_R2(ii) = R2;
    end
    
    % Compute the standard deviation
    std_MSE_F_fold(i) = std(std_MSE);
    std_NMSE_F_fold(i) = std(std_NMSE);
    std_R2_F_fold(i) = std(std_R2);
    std_AIC_F_fold(i) = std(std_AIC);
    std_BIC_F_fold(i) = std(std_BIC);
end

% Compute the Mean metrics
AIC_F_fold = AIC_F_fold/F_fold;
BIC_F_fold = BIC_F_fold/F_fold;
MSE_F_fold = MSE_F_fold/F_fold;
NMSE_F_fold = NMSE_F_fold/F_fold;
R2_F_fold = R2_F_fold/F_fold;
end

