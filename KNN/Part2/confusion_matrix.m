function [C] =  confusion_matrix(y_test, y_est)
%CONFUSION_MATRIX Implementation of confusion matrix 
%   for classification results.
%   input -----------------------------------------------------------------
%
%       o y_test    : (1 x M), a vector with true labels y \in {0,1} 
%                        corresponding to X_test.
%       o y_est     : (1 x M), a vector with estimated labels y \in {0,1} 
%                        corresponding to X_test.
%
%   output ----------------------------------------------------------------
%       o C          : (2 x 2), 2x2 matrix of |TP & FN|
%                                             |FP & TN|.
%        
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C = zeros(2,2);

% If compare(i) = 3 it's a true negative, compare(i) = 2 False positive,
% compare(i) = 1 False negative and compare(i) = 0 true positive
compare = 2*y_test+y_est;

C(1,1) = sum(compare == 0);
C(1,2) = sum(compare == 1);
C(2,1) = sum(compare == 2);
C(2,2) = sum(compare == 3);

end

