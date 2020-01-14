
% Return the cost function (cross-entropy) without the regularization term
% for a dataset.

function C = CostFunctionSet(In,Out,NNs)

n = size(In,1);
L = length(NNs);
C = 0;

for m = 1:n
    NNs(1).A = In(m,:)';
    NNs = FeedForward(NNs);
    C = C + CostFunction( Out(m,:)', NNs(L).A );
end

C = C/n;

% End of function
