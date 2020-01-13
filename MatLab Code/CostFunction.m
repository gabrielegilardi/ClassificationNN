
% Return the cost function (cross-entropy) without the regularization term.

function C = CostFunction(Y,A)

C = -sum( Y.*log(A) + (1-Y).*log(1-A) );
  
% End of function
