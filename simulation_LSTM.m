%
% Name:   simulation_LSTM.m
%--------------------------------------------------------------------------

clear;
close all;

addpath('shared_functions_LSTM/');

grid_s_choice = 2;
disp(' ');
for LSTM_method = 1:2
    rng(202407);

    %================= generate training data, test data ==================
    reg_func_m_true = @(s) sin(s);

    window = 10;

    grid_s_for_train = (-3*pi : 0.1 : 3*pi);
    n_grid_s_for_train = length(grid_s_for_train);

    grid_s_for_test  = (-3*pi : 0.1 : 3*pi);
    n_grid_s_for_test  = length(grid_s_for_test);

    if grid_s_choice == 2
        grid_s_for_train = linspace(-3*pi, 3*pi, n_grid_s_for_train);
        grid_s_for_test  = linspace(-3*pi, 3*pi, n_grid_s_for_test);
    end

    %------------------------- Training data ------------------------------
    noise_for_train = 0.1*randn(size(grid_s_for_train));
    x1_for_train = reg_func_m_true(grid_s_for_train) + noise_for_train;

    n_train = n_grid_s_for_train - window;
    X_train_array = cell(n_train, 1);
    Y_train_array = cell(n_train, 1);
    for i = 1:n_train
        X_train_array{i} = x1_for_train(    i:(i+window-1));
        Y_train_array{i} = x1_for_train((i+1):(i+window));
    end
    dim_X = size(X_train_array{1}, 1);
    dim_Y = size(Y_train_array{1}, 1);

    %--------------------------- Test data --------------------------------
    noise_for_test = 0.1*randn(size(grid_s_for_test));
    x2_for_test = reg_func_m_true(grid_s_for_test) + noise_for_test;

    n_test = n_grid_s_for_test - window;
    X_test_array = cell(n_test, 1);
    Y_test_array = cell(n_test, 1);
    for i = 1:n_test
        X_test_array{i} = x2_for_test(    i:(i+window-1));
        Y_test_array{i} = x2_for_test((i+1):(i+window));
    end

    Y_test_vector = zeros(1, n_test);
    for i = 1:n_test
        Y_test_vector(i) = Y_test_array{i}(window);
    end

    t_test_vector = ((window+1):n_grid_s_for_test);

    s_test_vector = grid_s_for_test(t_test_vector);
    true_m_test_vector = reg_func_m_true(s_test_vector);

    %=========================== settings =================================
    num_HiddenUnits = 4;
    num_neurons_fc_layer_1 = 10;
    num_neurons_fc_layer_2 =  4;
    num_neurons_fc_layer_3 = dim_Y;
    initial_learn_rate = 1e-4;
    mini_batch_size = 20;
    max_epochs = 1000;
    L2_regularization = 0.1;
    if     LSTM_method == 1
        name_method = 'ourcode'; name_method_title = 'Our code';

    elseif LSTM_method == 2
        name_method = 'toolbox'; name_method_title = 'Toolbox';

        %------- Define the CNN architecture: sequenceInputLayer ----------
        seq_input_layer_LSTM = [
            sequenceInputLayer(dim_X, 'Name', 'input')
            lstmLayer(num_HiddenUnits, 'Name', 'lstm')
            fullyConnectedLayer(num_neurons_fc_layer_1, 'Name', 'fc1')
            fullyConnectedLayer(num_neurons_fc_layer_2, 'Name', 'fc2')
            fullyConnectedLayer(num_neurons_fc_layer_3, 'Name', 'fc3')
            regressionLayer('Name', 'regressionoutput')];

        %------- Specify training options: trainingOptions ----------------
        training_options_LSTM = trainingOptions('sgdm', ...
            InitialLearnRate = initial_learn_rate, ...
            MiniBatchSize = mini_batch_size, ...
            MaxEpochs = max_epochs, ...
            Momentum = 0, ...
            L2Regularization = L2_regularization, ...
            Shuffle = 'every-epoch', ...
            GradientThreshold = inf, ...
            Plots = 'training-progress', ...
            Verbose = false);
    end

    tic;
    %====================== train the LSTM model ==========================
    %-------------------- trained model -----------------------------------
    if     LSTM_method == 1
        model_LSTM_ourcode = LSTM_ourcode('regression');
        loss_ourcode = ...
            model_LSTM_ourcode.train(X_train_array, Y_train_array, ...
            b = mini_batch_size, lr = initial_learn_rate, ...
            dc_lr = false, max_epochs = max_epochs, ...
            dim_h = num_HiddenUnits, ...
            dim_fc1 = num_neurons_fc_layer_1, ...
            dim_fc2 = num_neurons_fc_layer_2, ...
            dropout = 0, lambda_L2 = L2_regularization);

    elseif LSTM_method == 2
        model_LSTM_toolbox = trainNetwork(X_train_array, Y_train_array, ...
            seq_input_layer_LSTM, training_options_LSTM);
    end
    toc;

    %============== predict the test data Y_test_vector ===================
    %--------- predict the test data Y_test_vector
    if     LSTM_method == 1
        pred_Y_test_array = model_LSTM_ourcode.predict(X_test_array);
    elseif LSTM_method == 2
        pred_Y_test_array = model_LSTM_toolbox.predict(X_test_array);
    end

    pred_Y_test_vector = zeros(1, n_test);
    for i = 1:n_test
        pred_Y_test_vector(i) = pred_Y_test_array{i}(window);
    end

    MSE_test = mean( (Y_test_vector - pred_Y_test_vector).^2 );

    %============================== output ================================

    h_1 = figure(LSTM_method);
    subplot(2, 2, 1);

    plot(s_test_vector, Y_test_vector, 'k.'); hold on;
    plot(s_test_vector, pred_Y_test_vector, 'r-');
    plot(s_test_vector, true_m_test_vector, 'b--');
    h_legend = legend('\textbf{orig.}', '\textbf{pred.}', '\boldmath{$m(s)$}');
    set(h_legend, 'FontWeight', 'bold', 'Location', 'southwest', ...
        'Interpreter', 'latex', ...
        'FontSize', 5, 'ItemTokenSize', [20, 20]);
    title([name_method_title, ': \textbf{test}, MSE = ', ...
        sprintf('%.3f', MSE_test)], 'Interpreter', 'latex');

    xlabel('\boldmath{$s$}', 'Interpreter', 'latex')
    xlim([-10 +10]); ylim([-(1+0.2) +(1+0.2)]);

end
