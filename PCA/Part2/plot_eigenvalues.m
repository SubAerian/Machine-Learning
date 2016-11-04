function [] = plot_eigenvalues( L )
%PLOT_EIGENVALUES Simple plotting function to visualize eigenvalues
%   The student should convert the Eigenvalue matrix to a vector and 
%   visualize the values as a 2D plot.
%   input -----------------------------------------------------------------
%   
%       o L      : (N x N), Diagonal Matrix composed of lambda_i 
%                             
N = size(L);
vect = L*ones(N(1),1);

figure()
plot(vect, '--');                       


end

