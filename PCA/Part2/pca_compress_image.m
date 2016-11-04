% PCA for Data Compression
% author : COLIN Wilson

clear all;
close all;
clc;

% Dataset Path
dataset_path = 'Dataset/Compression/';

% Load 2D Testing Dataset for PCA
load(strcat(dataset_path,'dream-catchers.mat'))

% Display the Original Image
figure()
subplot(2,2,1);
imagesc(X);
colormap('gray');
title('Original');

% Extract Principal Directions and Components
[V, L, Mu] = my_pca(X);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%% Compression of 75%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
Percent = 0.75;
[X_hat]  = compress(X, Mu, V, Percent);
subplot(2,2,2)
imagesc(X_hat);
colormap('gray');
title('Compression Image by 0.75');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%% Compression of 90%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
Percent = 0.90;
[X_hat]  = compress(X, Mu, V, Percent);
subplot(2,2,3)
imagesc(X_hat);
colormap('gray');
title('Compression Image by 0.90');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%% Compression of 95%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
Percent = 0.95;
[X_hat]  = compress(X, Mu, V, Percent);
subplot(2,2,4)
imagesc(X_hat);
colormap('gray');
title('Compression Image by 0.95');

