function [ ] = knn_eval( X_train, y_train, X_test, y_test, k_range )
%KNN_EVAL Implementation of kNN evaluation.
%
%   input -----------------------------------------------------------------
%   
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {0,1} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_test   : (1 x M_test), a vector with labels y \in {0,1} corresponding to X_test.
%       o k_range  : (1 X K), Range of k-values to evaluate
%
%   output ----------------------------------------------------------------
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d_type = 'L2';

[~, dim] = size(k_range);
acc = zeros(1,dim);

for i = 1:dim
    % Compute y_estimate from k-NN
    y_est =  my_knn(X_train, y_train, X_test, k_range(i), d_type);
    
    % Compute Accuracy
    acc(i) =  my_accuracy(y_test, y_est);
end

figure()
plot(k_range, acc, '-o');
title('Classification Evaluation for KNN');
xlabel('k');
ylabel('Acc');
end

