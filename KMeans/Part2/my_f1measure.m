function [f1measure] =  my_f1measure(test_labels, true_labels)
%MY_FMEASURE Computes the f1-measure between two labels for a dataset (as column vectors)
%   depending on the choosen distance type={'L1','L2','LInf'}
%
%   input -----------------------------------------------------------------
%   
%       o true_labels     : (M x 1),  M-dimensional vector with true labels for
%                                     each data point
%       o test_labels     : (M x 1),  M-dimensional vector with classified labels for
%                                     each data point
%   output ----------------------------------------------------------------
%
%       o f1_measure      : f1-measure for the classified labels
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = length(true_labels);
true_K = unique(true_labels);
found_K = unique(test_labels);

nClasses = length(true_K);
nClusters = length(found_K);
C = zeros(1, nClasses);

% Initializing variables

P = zeros(nClasses,nClusters);
R = zeros(nClasses,nClusters);
F1 = zeros(nClasses,nClusters);

for ii = 1:nClasses
    for jj = 1:nClusters
        
        C(ii) = sum(true_labels==ii);
        % Implement the precision equation here
        P(ii,jj) = sum((true_labels==ii).*(test_labels==jj))/sum(test_labels==jj);
        
        % Implement the recall equation here
        R(ii,jj) = sum((true_labels==ii).*(test_labels==jj))/C(ii);
        
        % Implement the F1 measure for each cluster here
        F1(ii,jj) = 2*(R(ii,jj).*P(ii,jj))/(R(ii,jj)+P(ii,jj));
    end
end

% Implement the F1 measure for all clusters here

f1measure = sum((C/M).*max(F1')) ;
end
