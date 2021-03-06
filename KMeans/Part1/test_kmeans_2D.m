%%  Test Implementation of K-Means Algorithm
%    on 2D Datasets. 
%% ----------> run from ML_toolbox directory: >> addpath(genpath('./'));
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         1) Load 2D KMEANS Function Testing Dataset          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1a) Load 2D data sampled from a GMM
clear all;
close all;
clc;

load('Dataset/2d-gmm-4.mat')

% Visualize Dataset
options.class_names = {};
options.title       = '2D Dataset';

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;
colors     = hsv(4);
ml_plot_centroid(gmm.Mu',colors);
ml_plot_sigma (gmm, colors, 10);

%% 1b) Load 2d Ripley Dataset
clear all;
close all;
clc;

load('Dataset/2d-ripley.mat')

% Visualize Dataset
options.class_names = {};
options.labels      = labels;
options.title       = '2D Ripley Dataset';

if exist('h0','var') && isvalid(h0), delete(h0);end
h0 = ml_plot_data(X',options);hold on;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%              2) Test my_kmeans.m function                  %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;close all;
K = 4; init='random'; type='L2'; MaxIter = 100; plot_iter = 1;
[labels, Mu] =  my_kmeans(X, K, init, type, MaxIter, plot_iter);

% Plot decision boundary
my_kmeans_result.distance    = type;
my_kmeans_result.K           = K;
my_kmeans_result.method_name = 'kmeans';
my_kmeans_result.labels      = labels';
my_kmeans_result.centroids   = Mu';
my_kmeans_result.title       = sprintf('. My K-means result. K = %d, dist = %s',K, type);

if exist('hd','var') && isvalid(hd), delete(hd);end
hd = ml_plot_class_boundary(X',my_kmeans_result);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%             3) Choosing K test kmeans_eval.m               %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% K-means Evaluation Parameters
K_range=1:10; type='L2'; repeats = 10;
init='random'; MaxIter = 100;

% Evaluate k-means to find the optimal k
clc;
kmeans_eval(X, K_range, repeats, init, type, MaxIter);
