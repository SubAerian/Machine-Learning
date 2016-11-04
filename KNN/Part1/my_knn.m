function [ y_est ] =  my_knn(X_train,  y_train, X_test, k, type)
%MY_KNN Implementation of the k-nearest neighbor algorithm
%   for classification.
%
%   input -----------------------------------------------------------------
%   
%       o X_train  : (N x M_train), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y_train  : (1 x M_train), a vector with labels y \in {0,1} corresponding to X_train.
%       o X_test   : (N x M_test), a data set with M_test samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o k        : number of 'k' nearest neighbors
%       o type   : (string), type of distance {'L1','L2','LInf'}
%
%   output ----------------------------------------------------------------
%
%       o y_est   : (1 x M_test), a vector with estimated labels y \in {0,1} 
%                   corresponding to X_test.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialisation
[N, M_train] = size(X_train);
[~, M_test] = size(X_test);

% Matrix with each distances between the point to test and points from the
% training set on the same line.
dist = zeros(M_test, M_train);
% Indice matrix sorting in ascending order
D_knn = zeros(M_test, M_train);
y_est = zeros(1, M_test);


% Calcul of the distances for all the points from the test set.
for i = 1:M_test
    for j = 1:M_train
        dist(i,j) = my_distance(X_test(:,i), X_train(:,j), type);
    end
    % Sorting the d(s) in ascending order
    [dist(i,:), D_knn(i,:)] = sort(dist(i,:));
    
    % Class 1 if we have a majority of data from class 1, 0 otherwise
    sum_ = sum(y_train(D_knn(i,[1:k])));
    y_est(i) = sum_ > k - sum_;
end

end









