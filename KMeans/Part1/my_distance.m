function [d] =  my_distance(x_1, x_2, type)
%MY_DISTANCE Computes the distance between two datapoints (as column vectors)
%   depending on the choosen distance type={'L1','L2','LInf'}
%
%   input -----------------------------------------------------------------
%   
%       o x_1   : (N x 1),  N-dimensional datapoint
%       o x_2   : (N x 1),  N-dimensional datapoint
%       o type  : (string), type of distance {'L1','L2','LInf'}
%
%   output ----------------------------------------------------------------
%
%       o d      : distance between x_1 and x_2 depending on distance
%                  type {'L1','L2','LInf'}
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Output variable
d = 0;

switch type
    case 'L1'  % Manhattan Distance
         d = sum(abs(x_1-x_2));

    case 'L2' % Euclidean Distance        
         d = sqrt(sum(abs(x_1-x_2).^2));        
        
    case 'LInf' % Infinity Norm      
         d = max(abs(x_1-x_2));        
    
    otherwise
        warning('Unexpected distance type. No distance computed.')
end



end