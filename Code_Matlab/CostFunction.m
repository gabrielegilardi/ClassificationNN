% Copyright (c) 2020 Gabriele Gilardi

% Return the cost function (cross-entropy) without the regularization term
% for a full dataset (eqs. 57 and 63, Ch. 3).

function C = CostFunction(In,Out,NNs)

n = size(In,1);
L = length(NNs);
C = 0;

for m = 1:n
    NNs(1).A = In(m,:)';
    NNs = FeedForward(NNs);
    Y = Out(m,:)';
    A = NNs(L).A;
    C = C - sum( Y.*log(A) + (1-Y).*log(1-A) );
end

C = C/n;

% End of function
