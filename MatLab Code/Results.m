
% Determine the actual output activation vector for a full dataset

function [Res] = Results(In,NNs,nL)

nr = size(In,1);
L = length(NNs);
Res = zeros(nr,nL(L));
  
for m = 1:nr
    A = In(m,:)';
    for i = 2:L
        Z = NNs(i).W*A + NNs(i).B;
        A = f_activation(Z);
    end
    Res(m,:) = A';
end

% End of function
