% (c) 2020 Gabriele Gilardi

% Determine the output activation vector and the number of correct
% outputs (accuracy) for a full dataset

function [Res,nHits] = Results(In,Out,NNs,nL)

nr = size(In,1);
L = length(NNs);
Res = zeros(nr,nL(L));

% Output activation vector
for m = 1:nr
    A = In(m,:)';
    for i = 2:L
        Z = NNs(i).W*A + NNs(i).B;
        A = f_activation(Z);
    end
    Res(m,:) = A';
end

% Accuracy
[~,idxS] = max(Res,[],2);
[~,idxO] = max(Out,[],2);
a( idxS == idxO ) = 1;
nHits = sum(a)/nr;

% End of function
