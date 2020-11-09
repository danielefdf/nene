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
    ...
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
    ...
    [   9    6]    [   0]    [  -0.04]    
    [   9    7]    [   0]    [  -0.00]    
    [  10    2]    [   1]    [   0.80]    
    [  10    5]    [   0]    [  -0.04]    
    [  10    6]    [   0]    [  -0.03]    
    [  10   10]    [   0]    [   0.04]    

And based on these flags, shows the following graphs

    self.input_log        # the input data
    self.traing_qtts_log  # the neural network with weights represented by lines of varying thickness
    self.traing_acts_log  # the activation function
    self.traing_cost_log  # the cost function on the training dataset
    self.evaltn_cost_log  # the cost function on the evaluation dataset

![plots](https://github.com/danielefdf/nene/blob/main/docs/plots.bmp)

- circled points are the evaluation data

![neural network plot](https://github.com/danielefdf/nene/blob/main/docs/nnplot.gif)





