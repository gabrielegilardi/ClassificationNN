% (c) 2020 Gabriele Gilardi

% Sigmoid activation function (eq. 3, Ch. 1)

function  A = f_activation(Z)

A = 1./(1+exp(-Z));
  
% End of function
