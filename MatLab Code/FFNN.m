%
% ====================================================================
% FeedForward Neural Network (FNN) for class membership identification
% ====================================================================
%
% Characteristics:
% --------------------
% Works with any number of inputs/classes.
% Variable number of nodes in the input layer. 
% Variable number of nodes in the output layer.
% Variable number of hidden layers, each one with a variable number of nodes.
% Logistic sigmoid activation for hidden/output layers.
% Cross-entropy with L2 regularization as cost function.
% Variable learning rate.
% No early stop criteria.
%
% Parameters:
% -----------
% nL            Row-vector defining the number of nodes.
% name          Name of the text file with the dataset to analyze.
% split         Row-vector defining the number of data in the training (TR),
%               validation (VA), and test (TE) datasets.
% maxEpoch      Number of iterations.
% eta           Learning rate.
% etaCoeff      Coefficient for the learning rate strategy.
% lambda        Regularization parameter.
%
% Examples for nL:
% ----------------
% nL = [4 5 3]        4 inputs, 1 hidden layer with 5 nodes, 3 outputs
% nL = [2 10 8 5]     2 inputs, 2 hidden layers with 10 and 8 nodes, 5 outputs
%
% - nL(1) must be equal or smaller than the number of columns in the input 
%   data file.
% - nL(end) defines the number of classes.

rng(0);   % Used to generate the same sequence of random numbers

% Parameters (Example: Iris dataset)
nL = [4 5 3];       
name = 'IrisSet.txt';
split = [34 8 8];   
maxEpoch = 500;     
eta = 2;          
etaCoeff = 0.75;
lambda = 0;    
 
% Initialize the quantities in the hidden/output layers

% The first structure in the array is always empty except for the activation
% vector A that corresponds to the inputs
NNs = struct([]);
L = length(nL);                                 % Number of layers
for i = 2:L
    nNeu = nL(i);                               % Number of nodes 
    nInp = nL(i-1);                             % Number of inputs 
    NNs(i).B = randn(nNeu,1);                   % Biases
    NNs(i).W = (1/sqrt(nInp))*randn(nNeu,nInp); % Weights
    NNs(i).Z = zeros(nNeu,1);                   % Weighted inputs 
    NNs(i).A = zeros(nNeu,1);                   % Activations
    NNs(i).D = zeros(nNeu,1);                   % Delta errors
end
  
% Initialize cost function and accuracy vectors for training, validation,
% and test
costTR = zeros(maxEpoch,1);
costVA = zeros(maxEpoch,1);
costTE = zeros(maxEpoch,1);
accTR = zeros(maxEpoch,1);
accVA = zeros(maxEpoch,1);
accTE = zeros(maxEpoch,1);

% Read input data and build the input and output datasets for training,
% validation, and test
[InTR,OutTR,InVA,OutVA,InTE,OutTE] = ReadData(name,split,nL);
nTR = size(InTR,1);     % Number of training data
nVA = size(InVA,1);     % Number of validation data
nTE = size(InTE,1);     % Number of test data
  
% Main loop
for epoch = 1:maxEpoch
  
    % Every 10% of the epochs change learning rate and print count
    if (rem(epoch,fix(maxEpoch/10)) == 0)
        eta = etaCoeff*eta;
        fprintf('\n epoch = %d, eta = %f',epoch,eta)
    end

    % Initialize the derivative of the cost function for the nodes in the
    % hidden/output layers
    for i = 2:L
        nNeu = nL(i);                   % Number of nodes
        nInp = nL(i-1);                 % Number of inputs
        NNs(i).dB = zeros(nNeu,1);      % Derivatives wrt biases
        NNs(i).dW = zeros(nNeu,nInp);   % Derivatives wrt weights
    end

    % Loop over all training data
    for m = 1:nTR

        % Feedforward step
        NNs(1).A = InTR(m,:)';   
        NNs = FeedForward(NNs);  
      
        % Determine delta errors for hidden/output layers
        y = OutTR(m,:)';
        gradC = (y-ones(nL(L),1))./(NNs(L).A-ones(nL(L),1)) - y./NNs(L).A;
        NNs(L).D = gradC.*f1_activation( NNs(L).Z );
        for i = L-1:-1:2
            NNs(i).D = ( NNs(i+1).W'*NNs(i+1).D ).*f1_activation( NNs(i).Z ) ;
        end

        % Determine the derivatives associated with the current training 
        % data and add them to the partial total
        for i = 2:L
            NNs(i).dB = NNs(i).dB + NNs(i).D;
            for j = 1:nL(i)
                NNs(i).dW(j,:) = NNs(i).dW(j,:) + NNs(i).D(j)*NNs(i-1).A';
            end
        end

    end
    
    % Determine the new biases/weights
    for i = 2:L
        NNs(i).W = NNs(i).W*(1-eta*lambda/nTR) - (eta/nTR)*NNs(i).dW;
        NNs(i).B = NNs(i).B - (eta/nTR)*NNs(i).dB;
    end

    % Determine the cost function for all datasets 
    % (basic + regularization term)
    costR = 0;
    for i = 2:L
        % Regularization term
        costR = costR + lambda*sum( sum( NNs(i).W.*NNs(i).W ) )/(2*nTR);
    end
    % Add to the basic part
    costTR(epoch) = CostFunctionSet(InTR,OutTR,NNs) + costR;   
    costVA(epoch) = CostFunctionSet(InVA,OutVA,NNs) + costR;  
    costTE(epoch) = CostFunctionSet(InTE,OutTE,NNs) + costR;   
    
    % Determine the results for all datasets
    [ResTR,accTR(epoch)] = Results(InTR,OutTR,NNs,nL);
    [ResVA,accVA(epoch)] = Results(InVA,OutVA,NNs,nL);
    [ResTE,accTE(epoch)] = Results(InTE,OutTE,NNs,nL);
    
    % Determine the accuracy for all datasets
%    accTR(epoch) = Accuracy(ResTR,OutTR)/nTR;
%    accVA(epoch) = Accuracy(ResVA,OutVA)/nVA;
%    accTE(epoch) = Accuracy(ResTE,OutTE)/nTE;

end

fprintf('\n\n');
  
% Plots results for training, validation, and test datasets
hf = figure(1); 
set(hf,'color','white')
tc = 1:maxEpoch;
% Plot cost functions
subplot(2,1,1)
plot(tc,costTR,tc,costVA,tc,costTE)
xlim([min(tc) max(tc)])
xlabel('epoch')
ylabel('Cost')
title('COST FUNCTIONS','FontWeight','bold')
legend('Training','Validation','Test')
% Plot accuracies
subplot(2,1,2)
plot(tc,100*accTR,tc,100*accVA,tc,100*accTE)
xlim([min(tc) max(tc)])
xlabel('epoch')
ylabel('Accuracy (%)')
title('ACCURACIES','FontWeight','bold')
legend('Training','Validation','Test','Location','SouthEast')

% End of script
