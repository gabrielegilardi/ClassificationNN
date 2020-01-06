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
