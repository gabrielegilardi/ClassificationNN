
% Perform the feed-forward step (eq. 25, Ch. 2)

function  NNs = FeedForward(NNs)

for i = 2:length(NNs)
    NNs(i).Z = NNs(i).W*NNs(i-1).A + NNs(i).B;
    NNs(i).A = f_activation(NNs(i).Z);
end
  
% End of function
