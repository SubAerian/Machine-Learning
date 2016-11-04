%%  Test Implementation of K-Means Algorithm on Digits Dataset

%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              1) Load Digts Testing Dataset                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars; close all; clc;

load('digits.csv');
true_K = 4; % You can change this between 1 to 10 to test data with more clusters
[X, true_labels] = ml_load_digits_64('data/digits.csv', 0:true_K-1);

% Generate Variables
[M, N]  = size(X);
sizeIm  = sqrt(N);
idx = randperm(M);
nSamples = round(M);

X = X';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              2) Visualizing the Digits Dataset             %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot 64 random samples of the dataset as images
h0  = ml_plot_images(X(:,idx(1:64))',[sizeIm sizeIm]);

% Plot the first 8 dimensions of the image as data points
plot_options = [];
plot_options.labels = true_labels(idx(1:nSamples));
plot_options.title = '';
h1  = ml_plot_data(X([1 2 3 4 5 6 7 8], idx(1:nSamples))',plot_options);
axis equal;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             3) Run K-means on the raw data                 %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set parameters and evaluate K-means on the data for a range of 
% K from 1 to 10, with 10 repeats to find the optimal K
% K-means Evaluation Parameters
K_range=1:10; type='L2'; repeats = 10;
init='random'; MaxIter = 100;
kmeans_eval(X, K_range, repeats, init, type, MaxIter);

% Run K-means for the original data
plot_iter = 1;
[raw_labels, Mu] =  my_kmeans(X, true_K, init, type, MaxIter, plot_iter);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             3) Project the data using PCA                  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Perform PCA on the data

% Find the number of dimensions to project from the eigen values
[V, L, MuP] = my_pca(X);
Var = 0.7;
p = explained_variance(L, Var);

% Project Data to Choosen Principal Components
[A_p, Y] = project_pca(X, MuP, V, p);

% Visualize as scatter plot
plot_options = [];
plot_options.title = '';
plot_options.labels = true_labels;
h2  = ml_plot_data(Y',plot_options);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             3) Cluster the projected data                  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set parameters and evaluate K-means on the data for a range of 
% K from 1 to 10, with 10 repeats to find the optimal K
K_range=1:10; type='L2'; repeats = 10;
init='random'; MaxIter = 100;

% Cluster the projected data
kmeans_eval(Y, K_range, repeats, init, type, MaxIter);

% Run K-means for the original data
plot_iter = 1;
[labels, Mu] =  my_kmeans(Y, true_K, init, type, MaxIter, plot_iter);

% Plot decision boundary
my_kmeans_result.distance    = type;
my_kmeans_result.K           = true_K;
my_kmeans_result.method_name = 'kmeans';
my_kmeans_result.labels      = labels';
my_kmeans_result.centroids   = Mu';
my_kmeans_result.title       = sprintf('My K-means result. k = %d, dist = %s',true_K, type);

if exist('hd','var') && isvalid(hd), delete(hd);end
hd = ml_plot_class_boundary(Y',my_kmeans_result);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%       4) Find the F1-measure for the clustered data        %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Complete the function, my_f1measure() and run this block to verify

f1_measure_projected = my_f1measure(labels', true_labels);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5) Find the F1-measure for the different number of clusters%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k_range=1:10; type='L2'; 
init='random'; MaxIter = 500; repeats = 10;

f1measure_eval(Y, K_range, repeats, init, type, MaxIter, true_labels);
f1measure_eval(X, K_range, repeats, init, type, MaxIter, true_labels);


