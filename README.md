# A statistician’s selective review of neural network modeling: algorithms and applications

Readme File for Demonstrating the Numerical Study: Copy all files into a single directory.

1. To conduct Numerical Experiments for CNN  in Section 6.3: 

    Run "Realdata_MNIST_CNN.m." (The real data is internal and embedded in the MATLAB package.)

2. To conduct Numerical Experiments for LSTM  in Section B.1: 

    Run "simulation_LSTM.m". 

3. To conduct Numerical Experiments for FNN (Nonparametric Regression) in Section 4:
   
    Run "non_para_func_est_NN.m". This script generates the numerical study results.

    Input Details: Feedforward NN training method: 1: Mini-BGD;  2: Adam;  3: Toolbox fitnet

  - Figure 4 Inputs:

    Panel (a): Input '1'

    Panel (b): Input '2'

    Panel (c): Input '3'

4. To conduct Numerical Experiments for FNN (Binary Classification) in Section 3: 

    Run "Example_1_binary_classification.m".

  - Figure 2 Panel Inputs:

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

  - Figure 3 Inputs:

    Left panels:   Input '1, 1' for SGD.

    Middle panels: Input '1, 3' for mini-BGD.

    Right panels:  Input '1, 5' for mini-BGD.
