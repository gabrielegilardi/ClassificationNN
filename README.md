# Feed-Forward Neural Network (FFNN)

The code has been written in Octave (basic version with no extra packages) and tested on version 4.2.1 for Windows 7-64 bit, and on version 4.0.0 for Lubuntu 16.01.01 (installed on Oracle Virtual Box under Windows 7).

#### FFNN characteristics

- Works with any number of inputs/classes.
- Variable number of nodes in the input layer.
- Variable number of nodes in the output layer.
- Variable number of hidden layers, each one with a variable number of nodes.
- Logistic sigmoid activation for hidden/output layers.
- Cross-entropy with L2 regularization as cost function.
- Variable learning rate.
- No early stop criterion.

### List of functions, parameters, and main variables

|Functions|Use|
|:--------|:------|
|NNBP|Main function (script)|
|ReadData|Read the data from a text file and return the training/validation/test datasets|
|FeedForward|Perform the feedforward step|
|f_activation|Sigmoid activation function|
|f1_activation|First derivative of the sigmoid activation function|
|CostFunction|Return the basic cost function (without the regularization term)|
|CostFunctionSet|Return the basic cost function for a full dataset|
|Results|Determine the actual output activation vector for a full dataset|
|Accuracy|Determine the number of correct outputs for a full dataset|

|Parameters|Use|
|:---------|:------|
|nL|Row-vector defining the number of nodes|
|name|Name of the text file with the dataset to analyze|
|split|Row-vector defining the number of data in the training/validation/test datasets|
|maxEpoch|Number of iterations|
|eta|Learning rate|
|etaCoeff|Coefficient for the learning rate strategy|
|lambda|Regularization parameter|

|Main Variables|Use|
|:---------------------|:------|
|NNs|Structure with the main data associated with each layer. The components are: W=weight matrix, B=bias vector, A=activation vector, Z=weighted input vector, D=delta error vector, dB=derivatives of the basic cost function wrt the biases, dW=derivatives of the basic cost function wrt the weights.
|costTR,costVA,costTE|Cost function of the training/validation/test datasets|
|accTR,accVA,accTE|Accuracy of the training/validation/test datasets|
|InTR,InVA,InTE|Training/validation/test datasets (inputs)|
|OutTR,OutVA,OutTE|Training/validation/test datasets (desired outputs)|
|ResTR,ResVA,ResTE|Training/validation/test results (actual outputs)|

### Input information

- The code works with any number of inputs/classes, as long as the dataset is organized as the Iris dataset (i.e. the data for each class are in consecutive rows) and the problem is a classification problem.

- The layout of the NN system is defined in the row-vector `nL=[n1 n2 .... nL-1 nL]`, where: `n1` is the number of nodes in the input layer; `n2` to `nL-1` are the number of nodes in the hidden layers; and `nL` is the number of nodes in the output layer.

- Data must be in a text file (specified in parameter name), each column representing one of the inputs, and with data belonging to different classes in sequence (as in the Iris dataset). If the number of columns is larger than `n1`, then the extra column(s) are eliminated. The number of classes is defined by `nL`. The input data are mapped in the interval `[-1,+1]`.

- The data partition is defined in the row-vector `split=[nTR nVA nTE]`. Any value can be used as long as the sum is not larger than the number of data in each class. The three datasets are returned in matrixes `InTR`, `InVA`, and `InTE`. Data are taken sequentially (for instance if `split=[15 5 8]` then the first 28 rows are used).

- A very simple strategy is implemented to change the learning rate `eta`: at every 10% of the number of iterations, eta is recalculated as `etaCoeff` times `eta`.

### Output information

- Class membership is defined by a 1 in the corresponding column of matrixes `OutTR`, `OutVA`, and `OutTE`. For the Iris example it is: `[1,0,0]` for setosa, `[0,1,0]` for versicolor, and `[0,0,1]` for virginica.

- The cost function for the three partitions is in `costTR`, `costVA`, and `costTE`. These values are plotted versus the iterations at the end of the computation.

- The accuracy for the three partitions is in `accTR`, `accVA`, and `accTE`. These values are plotted versus the iterations at the end of the computation. The actual output of the system is assumed to be the highest activation value from all nodes in the output layer.

- Matrixes `ResTR`, `ResVA`, and `ResTE`, contain the activation value of all nodes in the output layer at the last iteration.

### Example: the IRIS dataset

The Iris dataset has 3-classes (setosa, versicolor, and virginica), and the input data are organized in a 150 x 4 matrix, with 50 rows for each class. The number of nodes in the input layer can assume a value from 1 to 4, while the number of nodes in the output layer must be 3 (i.e. the number of classes). The parameters used in the example are:

```
nL = [4 5 3]
name = 'IrisSet.txt'
split = [34 8 8]
maxEpoch = 1000
eta = 2.0
etaCoeff = 0.75
lambda = 0.0
```

- Layout: 4 input nodes, one hidden layer with 5 nodes, 3 nodes in the output layer.

- Input data are split with a 68/16/16 percent ratio.

- Every 10% of the iterations the learning rate is reduced by 25%, starting from an initial value of 2.

- No regularization is used.

These values have been set after running cases changing: the number of hidden layers, the number of nodes in the hidden layers, the learning rate, the coefficient in the learning rate strategy, the regularization parameter. The cost function and accuracy of the training/validation datasets were used to evaluate the results and set the parameters. In particular, increasing the number of hidden layers/nodes or using regularization did not have much effects on the final result.

Figure 1 shows the cost function for the three datasets, while Figure 2 shows the accuracy for the three datasets. 
