ASSIGNMENT 1 - FEED FORWARD NEURAL NETWORK

The above folder "FeedForwardNeuralNetwork" Contains 2 files.

File 1 - Feed_Forward_Neural_Networks.ipynb

This is a google colab file for the first 9 questions in the assignment.

Flow of the code:
-The code is divided into multiple sections.
- We are using all the hyper parameters as gllobal variables which are set when the train function executes.
- Q1. The dataset is imported from Keras as instructed and is visualised both on python and wandb.
- Data Preprocessing:
  - The input data from keras has features of the form (28 x 28 x tr_ex)
  - 1) Flattening the data to (784 x tr_ex)
  - 2) Splitting the training data from Keras into training set and cross validation set.
    - we have used sklearn functions for this purpose.
    - the training data is split into x_cv_train - 90% of training data as training set, x_cv_test - 10% of training data as cross validation set
  - 3) Normalization - To make sure our feature set has zero centred mean and unit variance
  - 4) One Hot Vectors - Given output vector is a vector with 10 labels
    - In this step, we are converting y which ranges from 0-9 into a One Hot Vector
    - After getting transformed, y is (tr_ex x 10), 1 column for every label.
- Error Functions
  - Q1-Q7 require us to work on cross_entropy as our loss function
  - Q8 require us to work on mse and compare both the results.
  - choose_loss_func() is a function used to pick the required loss function depending on the question requirements.
- Activation Functions
  - Q4 requires us to use sigmoid, tanh and relu as activation functions in hidden layers.
  - Since this is a multi-class classification problem, activation function used at output layer is softmax function.
  - choose_activation() is a function used to pick the required activation on each run.
  - activation functions are defined in train() function for each run.
- Derivatives of Activation Functions
  - Q4 requires us to use sigmoid, tanh and relu as activation functions in hidden layers.
  - choose_der_activation() is a function used to pick the required derivative of the activation on each run.
  - activation functions are defined in train() function for each run.
- Optimizers
  - Q3 requires us to implement sgd, momentum, nesterov, rmsprop, adam and nadam optimizers.
  - These optimizers use global dictionaries which are reset in every run using the reset_optimizer_update_dictionaries() function
  - choose_gd() function is used to pick the required optimizer in each run.
  - optimizer for each run are set in train() function for each run.
-Building a Neural Network
 - Initialize parameters
  - Q4 requires us to implement random weight initialization and xavier weight initialization.
  - These functions are used to initialise the weights before performing fwd prp and bkwd prop in every run.
  - choose_init() function is used to pick the required initialization in each run.
  - wt_init variable is used to update the type of the initialization.
 - Forward propagation
  - forward_propagation() function is used to perform fwd prp
  - pre activation calculations and activations for introducing non-linearity are performed in this function
 - Backward propagation
  -Calculating the error from the final layer to the initial layer is performed in this func
  - choose_dz() function is used based on the loss_function used in the run
 - Finally build the model
  - For every epoch, for every batch of input data, perform fwd prp, bkwd prp, use an optimizer function to update the weights
  - For every epoch, update the performance of the model wrt validation set and test set and visualize the output on wandb.
- predict() function is used to predict the output with the weights learnt
- confusion() function is used to visualize confusion matrix on colab
- we are also calling wandb function for confusion matrix to generate the confusion matrix on wandb
- accuracy() is used to calculate the accuracy between predicted and actual op
- Evaluate_perf() is used to evaluate the predicted output on every epoch
- logging() is used to log data to wandb and visualize the charts
- sweep configs
  - for Q4, Q8, the sweep config used is defined here.
  - using multiple values for hyper parameters and visualizing otuput in wandb
  - for Q4, we are using only cross_entropy loss function
  - parameters are taken as mentioned in the questions


File 2 - Q10.ipynb
Contains the same code evaluated on MNIST dataset by varying 3 hyperparameters, namely:
- batch_size
- optimizer
- epochs



Apart from the above folder, 2 other files exist:
wandb report
README


Example run of the code:
Open the file on google colab and execute run all to see output on wandb
- you can change the count in sweep configs to not intensively see all combinations
- you can further comment out the sweep configs which are not used to save time on file run
- to change the dataset, you can change the data being imported in Import dataset section
- Q10 does the same function on MNIST dataset