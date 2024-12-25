# A statistician’s selective review of neural network modeling: algorithms and applications
Readme File: Copy all five directories into a common directory.

1. Numerical Experiments for CNN

   To conduct numerical experiments for CNN, go to the directory 'Matlab_for_CNN/':

    Run "Realdata_MNIST_CNN.m". This script generates the results in Section 6.3.

2. Numerical Experiments for LSTM

   To conduct numerical experiments for LSTM, go to the directory 'Matlab_for_LSTM/':

    Run "simulation_LSTM.m". This script generates the simulation study results in Section B.1.
    Run "Realdata_LSTM.m".   This script generates the real data analysis results in Section B.2.

3. Numerical Experiments for FNN (Nonparametric Regression)

   To conduct numerical experiments for FNN applied to nonparametric regression, go to the directory 'Matlab_for_nonparametric-regression/':

    Run "non_para_func_est_NN.m". This script generates the numerical study results in Section 3.

    Input Details:

    Data type:
        1: Simulated data;  2: Real motorcycle data

    Feedforward NN training method:
        1: Mini-BGD;  2: Adam;  3: Toolbox fitnet

    Figure Inputs:

    Figure (a): Input '1, 1'
    Figure (b): Input '1, 2'
    Figure (c): Input '1, 3'
    Figure (d): Input '2, 1'
    Figure (e): Input '2, 2'
    Figure (f): Input '2, 3'

4. Numerical Experiments for FNN (Binary Classification)

   To conduct numerical experiments for FNN applied to binary classification, go to the directory 'Matlab_for_binary-classification/':

    Run "Example_1_binary_classification.m".

    Figure 2 Panel Inputs:

    Panel (a): Input '4' for logistic regression.
    Panel (e) and Panel (f):
        Panel (e): Input '2' for Adam.
        Panel (f): Input '3' for NLLS.
    Panel (b) and Panel (c): The procedure is the same as for Panel (e) and Panel (f), except modify lines 145--146:
        Change "choice_loss_func_output = 2; % cross-entropy (neg-log-likelihood)"
        to
        "choice_loss_func_output = 1; % quadratic loss".

        Panel (b): Input '2' for Adam.
        Panel (c): Input '3' for NLLS.
    Panel (d): Input '5' for toolbox.

Figure 3 Inputs:

    Left:   Input '1, 1' for SGD.
    Middle: Input '1, 3' for mini-BGD.
    Right:  Input '1, 5' for mini-BGD.
