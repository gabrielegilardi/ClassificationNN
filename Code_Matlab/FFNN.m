% Copyright (c) 2020 Gabriele Gilardi
%
% ====================================================================
% Multivariate Classification Using a Feed-Forward Neural Network and
% Backpropagation.
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
%   data file (extra columns are ignored).
% - nL(end) defines the number of classes.
%
% Reference: Michael Nielsen, "Neural Networks and Deep Learning", 
%            Ch. 2 and 3, neuralnetworksanddeeplearning.com
% Datasets: archive.ics.uci.edu/ml/machine-learning-databases/

clear

rng(0);     % Used to generate the same sequence of random numbers
            % (remove/comment otherwise)

% Examples: 
% 1) Iris dataset = 'iris' 
% 2) Wheat Seeds dataset = 'seed'
example = 'seed';

% Exmples (removed the last column from the original databases)
switch example

    % 1) Iris dataset
    % archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
    case 'iris'
        nL = [4 5 3];       
        name = 'IrisDataset.txt';
        split = [34 8 8];   
        maxEpoch = 500;     
        eta = 2;          
        etaCoeff = 0.75;
        lambda = 0;    

    % 2) Wheat Seeds dataset
    % archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt
    case 'seed'
        nL = [7 5 3];       
        name = 'WheatSeedsDataset.txt';
        split = [50 10 10];   
        maxEpoch = 500;     
        eta = 2;          
        etaCoeff = 0.75;
        lambda = 0;    
        
    otherwise
        disp('No example specified')
        disp(' ')
        return

end

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

% Convention:
% TR = training
% VA = validation
% TE = test

% Initialize cost function vectors
costTR = zeros(maxEpoch,1);
costVA = zeros(maxEpoch,1);
costTE = zeros(maxEpoch,1);
% Initialize accuracy vectors
accTR = zeros(maxEpoch,1);
accVA = zeros(maxEpoch,1);
accTE = zeros(maxEpoch,1);

% Read input data and build the input and output datasets
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

        % Feedforward step (eq. 25, Ch. 2)
        NNs(1).A = InTR(m,:)';   
        NNs = FeedForward(NNs);  
      
        % Determine delta errors for hidden/output layers (eqs. BP1 and
        % BP2, Ch. 2)
        y = OutTR(m,:)';
        gradC = (y-ones(nL(L),1))./(NNs(L).A-ones(nL(L),1)) - y./NNs(L).A;
        NNs(L).D = gradC.*f1_activation( NNs(L).Z );
        for i = L-1:-1:2
            NNs(i).D = ( NNs(i+1).W'*NNs(i+1).D ).*f1_activation( NNs(i).Z ) ;
        end

        % Determine the derivatives associated with the current training 
        % data and add them to the partial sum (eqs. BP3 and BP4, Ch. 2)
        for i = 2:L
            NNs(i).dB = NNs(i).dB + NNs(i).D;
            for j = 1:nL(i)
                NNs(i).dW(j,:) = NNs(i).dW(j,:) + NNs(i).D(j)*NNs(i-1).A';
            end
        end

    end
    
    % Determine the new biases/weights (eqs. 93 and 94, Ch. 3)
    for i = 2:L
        NNs(i).W = NNs(i).W*(1-eta*lambda/nTR) - (eta/nTR)*NNs(i).dW;
        NNs(i).B = NNs(i).B - (eta/nTR)*NNs(i).dB;
    end

    % Determine the cost function for all datasets (eq. 85, Ch. 3)
    
    % Regularization contribution 
    costR = 0;
    for i = 2:L
        costR = costR + lambda*sum( sum( NNs(i).W.*NNs(i).W ) )/(2*nTR);
    end
    % Add to the basic cost function
    costTR(epoch) = CostFunction(InTR,OutTR,NNs) + costR;   
    costVA(epoch) = CostFunction(InVA,OutVA,NNs) + costR;  
    costTE(epoch) = CostFunction(InTE,OutTE,NNs) + costR;   
    
    % Output activation vector and accuracy for a full dataset
    [ResTR,accTR(epoch)] = Results(InTR,OutTR,NNs,nL);
    [ResVA,accVA(epoch)] = Results(InVA,OutVA,NNs,nL);
    [ResTE,accTE(epoch)] = Results(InTE,OutTE,NNs,nL);
    
end

fprintf('\n\n');
  
% Plots results for the training, validation, and test dataset
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
