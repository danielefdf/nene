# nene
A neural network from scratch (only Numpy)

The elements of the project are as follows.

### elements

- activation.py: functions and derivatives for different types of activation function

    - STEP     # step function - perceptrons
    - LINEAR   # linear function
    - SIGMOID  # sigmoid
    - TANH     # tanh
    - RELU     # ReLU
    - L_RELU   # leaky ReLU

- cost.py: functions and derivatives for different types of activation function

    - ABS_ERROR  # absolute difference
    - QUADRATIC  # quadratic error

- datasetxy.py: functions for input selection

    - ALL     # all points of the plane
    - RANDOM  # random selected points from the plane

- initialization.py: functions for different types of initialization (for weights and biases)

    - ZERO      # initialization with zero
    - RANDOM    # random gaussian distribution
    - CONSTANT  # constant initialization

- regularization.py: functions for different types of regularization

    - NONE  # no regularization
    - L1    # L1 regularization
    - L2    # L2 regularization

- training.py: types of training

    - GRADIENT_D   # gradient descent
    - STOCHSTC_GD  # stochastic gradient descent

### monitor

- monitor.py: functions for displaying data from the neural network

### neural network

- neuralNetwork.py: the neural network, that uses all other elements

### test

- test.and.py: neural network representation of an AND logic gate
- test.classn.binflags.py: a classification for two values, with output as flags ([1,0] or [0,1])
- test.classn.binvalues.py: a classification for two values, with output as single values ([0] or [1])
- test.classn.example.py: a slightly more complex problem of classification
- test.clustg.autoencdr.py: an example of autoencoder neural network
- test.regrsn.py: a regression problem

## output

The neural network shows the values of prediction on the training dataset and on the evaluation dataset.

An example from **test.classn.binvalues.py**:

    predictions on training data
    [   1    1]    [   1]    [   1.00]    
    [   1    2]    [   1]    [   1.00]    
    [   1    3]    [   1]    [   1.00]    
    [   1    4]    [   1]    [   1.00]    
    [   1    5]    [   1]    [   1.00]    
    [   1    6]    [   1]    [   1.00]    
    [   1   10]    [   1]    [   1.00]    
    [   2    1]    [   1]    [   1.00]    
    [   2    2]    [   1]    [   1.00]    
    [   2    4]    [   1]    [   1.00]    
    [   2    5]    [   1]    [   1.00]    
    [   2    6]    [   1]    [   1.00]    
    [   2    7]    [   1]    [   1.00]    
    [   2    8]    [   1]    [   1.00]    
    [   2   10]    [   1]    [   1.00]    
    [   3    1]    [   1]    [   1.00]    
    [   3    2]    [   1]    [   1.00]    
    [   3    4]    [   1]    [   1.00]    
    [   3    6]    [   1]    [   1.00]    
    [   3    7]    [   1]    [   1.00]    
    [   3    8]    [   1]    [   1.00]    
    [   3    9]    [   1]    [   1.00]    
    [   3   10]    [   1]    [   1.00]    
    [   4    1]    [   1]    [   1.00]    
    [   4    3]    [   1]    [   1.00]    
    [   4    4]    [   1]    [   1.00]    
    [   4    5]    [   1]    [   1.00]    
    [   4    7]    [   1]    [   1.00]    
    [   4    8]    [   1]    [   1.00]    
    [   4    9]    [   1]    [   1.00]    
    [   5    2]    [   1]    [   1.00]    
    [   5    3]    [   1]    [   1.00]    
    [   5    4]    [   1]    [   1.00]    
    [   5    5]    [   1]    [   1.00]    
    [   5    6]    [   1]    [   1.00]    
    [   5    7]    [   1]    [   1.00]    
    [   5   10]    [   1]    [   1.00]    
    [   6    1]    [   1]    [   1.00]    
    [   6    2]    [   1]    [   1.00]    
    [   6    5]    [   1]    [   1.00]    
    [   6    6]    [   1]    [   0.97]    
    [   6    7]    [   1]    [   0.95]    
    [   6    8]    [   1]    [   0.98]    
    [   6    9]    [   1]    [   0.99]    
    [   7    3]    [   1]    [   0.97]    
    [   7    5]    [   1]    [   0.90]    
    [   7    6]    [   0]    [   0.08]    
    [   7    7]    [   0]    [  -0.04]    
    [   7    8]    [   0]    [  -0.04]    
    [   7    9]    [   0]    [  -0.01]    
    [   7   10]    [   0]    [   0.04]    
    [   8    1]    [   1]    [   1.00]    
    [   8    2]    [   1]    [   0.97]    
    [   8    3]    [   0]    [   0.07]    
    [   8    4]    [   0]    [   0.04]    
    [   8    5]    [   0]    [   0.03]    
    [   8    6]    [   0]    [  -0.05]    
    [   8    7]    [   0]    [  -0.02]    
    [   8    8]    [   0]    [  -0.01]    
    [   8    9]    [   0]    [  -0.02]    
    [   8   10]    [   0]    [  -0.02]    
    [   9    3]    [   0]    [  -0.02]    
    [   9    4]    [   0]    [  -0.05]    
    [   9    5]    [   0]    [  -0.05]    
    [   9    8]    [   0]    [   0.02]    
    [   9    9]    [   0]    [   0.03]    
    [   9   10]    [   0]    [   0.02]    
    [  10    1]    [   1]    [   0.96]    
    [  10    3]    [   0]    [  -0.00]    
    [  10    4]    [   0]    [  -0.03]    
    [  10    7]    [   0]    [  -0.01]    
    [  10    8]    [   0]    [   0.03]    
    [  10    9]    [   0]    [   0.04]    

    predictions on evaluation data
    [   1    7]    [   1]    [   1.00]    
    [   1    8]    [   1]    [   1.00]    
    [   1    9]    [   1]    [   1.00]    
    [   2    3]    [   1]    [   1.00]    
    [   2    9]    [   1]    [   1.00]    
    [   3    3]    [   1]    [   1.00]    
    [   3    5]    [   1]    [   1.00]    
    [   4    2]    [   1]    [   1.00]    
    [   4    6]    [   1]    [   1.00]    
    [   4   10]    [   1]    [   1.00]    
    [   5    1]    [   1]    [   1.00]    
    [   5    8]    [   1]    [   1.00]    
    [   5    9]    [   1]    [   1.00]    
    [   6    3]    [   1]    [   1.00]    
    [   6    4]    [   1]    [   1.00]    
    [   6   10]    [   1]    [   1.00]    
    [   7    1]    [   1]    [   1.00]    
    [   7    2]    [   1]    [   1.00]    
    [   7    4]    [   1]    [   0.97]    
    [   9    1]    [   1]    [   0.99]    
    [   9    2]    [   0]    [   0.85]    
    [   9    6]    [   0]    [  -0.04]    
    [   9    7]    [   0]    [  -0.00]    
    [  10    2]    [   1]    [   0.80]    
    [  10    5]    [   0]    [  -0.04]    
    [  10    6]    [   0]    [  -0.03]    
    [  10   10]    [   0]    [   0.04]    

And based on these flags, shows the following graphs

    self.input_log        # graph of the input data
    self.traing_qtts_log  # ???
    self.traing_acts_log  # ???
    self.traing_cost_log  # graph of the cost function on the training dataset
    self.evaltn_cost_log  # graph of the cost function on the evaluation dataset

![plots](https://github.com/danielefdf/nene/blob/main/docs/plots.bmp)

- circled points are ???

![neural network plot](https://github.com/danielefdf/nene/blob/main/docs/nnplot.gif)





