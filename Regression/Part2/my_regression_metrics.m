function [MSE, NMSE, Rsquared] = my_regression_metrics( yest, y )
%MY_REGRESSION_METRICS Computes the metrics (MSE, NMSE, R squared) for 
%   regression evaluation
%
%   input -----------------------------------------------------------------
%   
%       o yest  : (P x M), representing the estimated outputs of P-dimension
%       of the regressor corresponding to the M points of the dataset
%       o y     : (P x M), representing the M continuous labels of the M 
%       points. Each label has P dimensions.
%
%   output ----------------------------------------------------------------
%
%       o MSE       : (1 x 1), Mean Squared Error
%       o NMSE      : (1 x 1), Normalized Mean Squared Error
%       o R squared : (1 x 1), Coefficent of determination
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
[P,M] = size(yest);

% Equation 16
MSE = sum(bsxfun(@minus, yest, y).^2, 2)/M;

% Equation 17
Mu = sum(y, 2)/M;
var_y = sum(bsxfun(@minus, y, Mu).^2, 2);

NMSE = MSE*(M-1)/var_y;

% Equation 18
y_est_mean = sum(yest, 2)/M;
num = sum(bsxfun(@times,bsxfun(@minus, y, Mu),bsxfun(@minus, yest, y_est_mean)),2)^2;
den = var_y*sum(bsxfun(@minus, yest, y_est_mean).^2, 2);

Rsquared = num/den;
end

