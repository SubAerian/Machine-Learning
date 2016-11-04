function [ V, L, Mu ] = my_pca( X )
%MY_PCA Step-by-step implementation of Principal Component Analysis
%   In this function, the student should implement the Principal Component 
%   Algorithm following Eq.1, 2 and 3 of Assignment 1.
%
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%
%   output ----------------------------------------------------------------
%
%       o U      : (M x M), Eigenvectors of Covariance Matrix.
%       o L      : (M x M), Eigenvalues of Covariance Matrix
%       o Mu     : (N x 1), Mean Vector of Dataset

% Auxiliary variables
[N, M] = size(X);

% Output variables
V  = zeros(N,N);
L  = zeros(M,M);
Mu = zeros(N,1);

% ====================== Implement Eq. 1 Here ====================== 
Mu = mean(X.').';
X = bsxfun(@minus,X,Mu);

% ====================== Implement Eq.2 Here ======================
C = (1/(M-1))*(X*X.');

% ====================== Implement Eq.3 Here ======================
[V,L] = eig(C);

% =================== Sort Eigenvectors wrt. EigenValues ==========
% Sort Eigenvalue and get indices
[L_sort, ind] = sort(diag(L),'descend');

% arrange the columns in this order
V=V(:,ind); 

% Vectorize sorted eigenvalues
L = diag(L_sort); 

end

