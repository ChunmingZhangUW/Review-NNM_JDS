%
% Name:   Realdata_LSTM.m
%--------------------------------------------------------------------------

clear;
close all;

addpath('shared_functions_LSTM/');

for LSTM_method = 1:2
    rng(2024);

    %---------------------- Training data processing ----------------------
    training_data = ...
        readtable('Daily_Climate_time-series_data/DailyDelhiClimateTrain.csv');
    [training_data, C_vector, S_vector] = normalize(training_data{:, 2:5});

    window = 7;

    n_train = size(training_data, 1) - window;
    X_train_array = cell(n_train, 1);
    Y_train_array = cell(n_train, 1);
    for i = 1:n_train
        X_train_array{i} = training_data(    i:(i+window-1), :)';
        Y_train_array{i} = training_data((i+1):(i+window),   1)';
    end
    dim_X = size(X_train_array{1}, 1);
    dim_Y = size(Y_train_array{1}, 1);

    Y_train_vector = zeros(1, n_train);
    for i = 1:n_train
        Y_train_vector(i) = Y_train_array{i}(window);
    end
    Y_train_vector = Y_train_vector*S_vector(1) + C_vector(1);

    t_train_vector = ((window+1):size(training_data, 1));

    %------------------------ Test data processing ------------------------
    test_data = ...
        readtable('Daily_Climate_time-series_data/DailyDelhiClimateTest.csv');
    test_data = normalize(test_data{:, 2:5}, center = C_vector, scale = S_vector);

    n_test = size(test_data, 1) - window;
    X_test_array = cell(n_test, 1);
    Y_test_array = cell(n_test, 1);
    for i = 1:n_test
        X_test_array{i} = test_data(    i:(i+window-1), :)';
        Y_test_array{i} = test_data((i+1):(i+window),   1)';
    end

    Y_test_vector = zeros(1, n_test);
    for i = 1:n_test
        Y_test_vector(i) = Y_test_array{i}(window);
    end
    Y_test_vector = Y_test_vector*S_vector(1) + C_vector(1);

    t_test_vector = ((window+1):size(test_data, 1));

    %=========================== settings =================================
    num_HiddenUnits = 16;
    num_neurons_fc_layer_1 = 28;
    num_neurons_fc_layer_2 =  4;
    num_neurons_fc_layer_3 = dim_Y;
    initial_learn_rate = 1e-6;
    mini_batch_size = 32;
    max_epochs = 2000;
    L2_regularization = 1;
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
    %=========================== train the LSTM model =====================
    %-------------------- trained model -----------------------------------
    if     LSTM_method == 1
        model_LSTM_ourcode = LSTM_ourcode('regression');
        loss_ourcode = model_LSTM_ourcode.train(X_train_array, Y_train_array, ...
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
    toc

    %================== fit the training data =============================
    %--------- fit the training data Y_train_vector
    if     LSTM_method == 1
        fit_Y_train_array = model_LSTM_ourcode.predict(X_train_array);
    elseif LSTM_method == 2
        fit_Y_train_array = model_LSTM_toolbox.predict(X_train_array);
    end

    fit_Y_train_vector = zeros(1, n_train);
    for i = 1:n_train
        fit_Y_train_vector(i) = fit_Y_train_array{i}(window);
    end
    fit_Y_train_vector = fit_Y_train_vector*S_vector(1) + C_vector(1);

    MSE_train = mean( (Y_train_vector - fit_Y_train_vector).^2 );

    %---------------------------- output ----------------------------------

    h_1 = figure(2*LSTM_method-1);
    subplot(2, 2, 1);

    plot(t_train_vector, Y_train_vector, 'k.'); hold on;
    plot(t_train_vector, fit_Y_train_vector, 'r-');
    h_legend = legend('\textbf{obs.}', '\textbf{fitted.}');
    set(h_legend, 'FontWeight', 'bold', 'Location', 'northwest', ...
        'Interpreter', 'latex', ...
        'FontSize', 5, 'ItemTokenSize', [20, 20]);
    title([name_method_title, ': \textbf{training}, MSE = ', ...
        sprintf('%.3f', MSE_train)], 'Interpreter', 'latex');

    xlabel('\boldmath{$t$}', 'interpreter', 'latex');
    ylabel('meantemp');


    %====================== predict the test data =========================
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
    pred_Y_test_vector = pred_Y_test_vector*S_vector(1) + C_vector(1);

    MSE_test = mean( (Y_test_vector - pred_Y_test_vector).^2 );

    %---------------------------- output ----------------------------------

    h_2 = figure(2*LSTM_method);
    subplot(2, 2, 1);

    plot(t_test_vector, Y_test_vector, 'k.'); hold on;
    plot(t_test_vector, pred_Y_test_vector, 'r-');
    h_legend = legend('\textbf{obs.}', '\textbf{pred.}');
    set(h_legend, 'FontWeight', 'bold', 'Location', 'northwest', ...
        'Interpreter', 'latex', ...
        'FontSize', 7, 'ItemTokenSize', [20, 20]);
    title([name_method_title, ': \textbf{test}, MSE = ', ...
        sprintf('%.3f', MSE_test)], 'Interpreter', 'latex');

    xlabel('\boldmath{$t$}', 'interpreter', 'latex');
    ylabel('meantemp');


end

