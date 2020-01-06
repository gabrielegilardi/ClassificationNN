
% Return the (basic) cost function without the regularization term.

function C = CostFunction(Y,A)

C = -sum( Y.*log(A) + (1-Y).*log(1-A) );
  
% End of function
