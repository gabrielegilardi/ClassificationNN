
% Read the data from a text file and return the input and output for 
% the training, validation, and test datasets.

% Data must be organize as ??????

function [InTR,OutTR,InVA,OutVA,InTE,OutTE] = ReadData(name,split,nL)
  
Data = load(name);
Data = Data(:,1:nL(1));       % Remove extra columns
[nrd,ncd] = size(Data);
nClas = nL(end);              % Number of classes
nelClas = round(nrd/nClas);   % Number of elements in each class
  
% Map input data in the interval [-1,+1]
ymin = -1;
ymax = +1;
xmax = max(Data);
xmin = min(Data);
for i = 1:nrd
     Data(i,:) = (ymax-ymin)*(Data(i,:)-xmin)./(xmax-xmin) + ymin;
end
  
% Build the training dataset
n  = split(1);      % Number of elements
is = 1;
ie = is+n-1;
InTR = zeros(n*nClas,ncd);      % Initialize input vector/matrix
OutTR = zeros(n*nClas,nClas);   % Initialize output matrix
for i = 1:nClas
    k = (i-1)*nelClas;
    j = (i-1)*n;
    InTR(j+1:i*n,:) = Data(k+is:k+ie,:);    % Copy input values
    OutTR(j+1:i*n,i) = 1.0; % Class membership is specified by a value of 1
end

% Build the validation dataset
n  = split(2);      % Number of elements
is = ie+1;
ie = is+n-1;
InVA = zeros(n*nClas,ncd);      % Initialize input vector/matrix
OutVA = zeros(n*nClas,nClas);   % Initialize output matrix
for i = 1:nClas
    k = (i-1)*nelClas;
    j = (i-1)*n;
    InVA(j+1:i*n,:) = Data(k+is:k+ie,:);    % Copy input values
    OutVA(j+1:i*n,i) = 1.0; % Class membership is specified by a value of 1
end
  
% Build the test dataset
n  = split(3);      % Number of elements
is = ie+1;
ie = is+n-1;
InTE = zeros(n*nClas,ncd);      % Initialize input vector/matrix
OutTE = zeros(n*nClas,nClas);   % Initialize output matrix
for i = 1:nClas
    k = (i-1)*nelClas;
    j = (i-1)*n;
    InTE(j+1:i*n,:) = Data(k+is:k+ie,:);    % Copy input values
    OutTE(j+1:i*n,i) = 1.0; % Class membership is specified by a value of 1
end

% End of function
