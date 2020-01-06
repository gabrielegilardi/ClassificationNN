
% Determine the number of correct outputs for a full dataset

function nHits = Accuracy(Res,Out)

% The actual output of the system is assumed to be the highest
% activation value from all nodes in the output layer

[~,idxS] = max(Res,[],2);
[~,idxO] = max(Out,[],2);
a( idxS == idxO ) = 1;
nHits = sum(a);

% End of function



