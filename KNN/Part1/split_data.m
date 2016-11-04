function [ X_train, y_train, X_test, y_test ] = split_data(X, y, tt_ratio )
%SPLIT_DATA Randomly partitions a dataset into train/test sets using
%   according to the given tt_ratio
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y        : (1 x M), a vector with labels y \in {0,1} corresponding to X.
%       o tt_ratio : train/test ratio.
%   output ----------------------------------------------------------------
%
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {0,1} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_test   : (1 x M_test), a vector with labels y \in {0,1} corresponding to X_test.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[N, M] = size(X);

% Calculation of the number of trainning and testing points
M_train = round(tt_ratio * M);

% Random initialisation of X_train and y_train
random_vect = randperm(M, M_train);
X_train = X(:, random_vect);
y_train = y(random_vect);
 
% Initialisation of X_test and y_test by removing from the input matrix the
% rows alocated to the training sets
X_test = removerows(X.', 'ind', random_vect).';
y_test = removerows(y.', 'ind', random_vect).';

end
























