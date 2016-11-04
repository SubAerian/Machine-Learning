function [] = reconstruction_eigenfaces(X, V, Mu, sizeIm)
%RECONSTRUCTION_EIGENFACES Reconstructs face images from lower dimensional
%space to compare the result to the original face image and to the mean face image
%   
Img = reshape(X(:,1),sizeIm,sizeIm);

figure()
% Plot Original Face
subplot(1,6,6)
imagesc(Img);
colormap('gray');
title('Original');

% Plot Mean Face
subplot(1,6,1)
imagesc(reshape(Mu,sizeIm,sizeIm));
colormap('gray');
title('Mean');

i = 1;
for p = 1 : 50 : 151
    i = i+1;
    
    % Project with PCA
    [A_p, Y] = project_pca(X, Mu, V, p );
    
    % Reconstruct Lossy Data from PCA
    [X_hat]  = reconstruct_pca(Y, A_p, Mu);    
    
    % Plot Reconstructed Faces
    Img = reshape(X_hat(:,1),sizeIm,sizeIm);
    subplot(1,6,i)
    imagesc(Img);
    colormap('gray');
    title(p);
    
end

end