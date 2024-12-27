%
% Name    : non_para_func_est_NN.m
%--------------------------------------------------------------------------

clear;
close all;

rng(100)

%==========================================================================
choice_data = 1; % 1: simulated data
choice_opt_method = input([' Input feedforward NN training method \n', ...
    '      (1: for mini-BGD; 2: for Adam; 3: for toolbox fitnet) = ']);

%======================== training data, test data ========================

%----------------------------- training data ------------------------------
if     choice_data == 1 % simulated data
    num_obs_train = 100;
    X_train_vector = 0.4 * rand(1, num_obs_train) + 0.1;

    reg_func_true = @(x) 5*sin(1./x);
    noise_train_vector = 1 * randn(1, num_obs_train);
    Y_train_vector = reg_func_true(X_train_vector) + noise_train_vector;

    title_data = 'simulated data';

end
dim_X = size(X_train_vector, 1);
dim_Y = size(Y_train_vector, 1);

%------------------------------- test data --------------------------------
num_obs_test = min(100, num_obs_train);

X_test_vector = linspace(min(X_train_vector), max(X_train_vector), ...
    num_obs_test);

%====================== Train the feedforward NN model ====================
if     choice_opt_method == 1 || choice_opt_method == 2 || ...
        choice_opt_method == 3

    %-------- Define the NN architecture ----------------------------------
    if     choice_opt_method == 1 || choice_opt_method == 2

        if     choice_data == 1 % simulated data
            hiddenLayerSizes = [6, 4, 4, 6];
        end
    elseif choice_opt_method == 3
        hiddenLayerSizes = [6, 4, 4];
    end
    num_nodes_vector = [dim_X, hiddenLayerSizes, dim_Y];

    num_layers = length(num_nodes_vector);
    if num_layers < 3
        disp('# of layers < 3. return!!!'); return
    end

    %--- activation functions
    choice_act_func_hidden = 3;
    choice_act_func_output = 0;

    choices_act_funcs_array = cell(num_layers, 1);
    for lay = 2 : (num_layers-1)
        choices_act_funcs_array{lay} = choice_act_func_hidden;
    end
    choices_act_funcs_array{num_layers} = choice_act_func_output;

    choice_loss_func_output = 1;

end

if     choice_opt_method == 1 || choice_opt_method == 2
    %------------------------ setting of our code -------------------------
    if     choice_opt_method == 1
        batch_size = num_obs_train;
        if batch_size == num_obs_train
            title_NN_method = 'BGD';
        end
        options_mini_BGD.batch_size = batch_size;

        options_mini_BGD.specified_num_batches = [];

        %--------- Specify training options in mini-BGD algorithm ---------
        learning_rate_specified = 0.01;
        options_mini_BGD.learning_rate = learning_rate_specified;
        options_mini_BGD.num_iterations = 10000;

    elseif choice_opt_method == 2
        title_NN_method = 'Adam';

    end

    %---------------- train the NN model using our code -------------------
    cc = 1;

    W_matrices_array_init = cell(num_layers, 1);
    b_vectors_array_init  = cell(num_layers, 1);
    for lay = 2 : num_layers
        W_matrices_array_init{lay} = cc * randn(num_nodes_vector(lay), ...
            num_nodes_vector(lay-1));
        b_vectors_array_init{lay}  = zeros(num_nodes_vector(lay), 1);
    end

    if     choice_opt_method == 1
        tic;
        [W_matrices_array_final, b_vectors_array_final, costs_iters_vector] ...
            = mini_BGD_NN(X_train_vector, Y_train_vector, ...
            W_matrices_array_init, b_vectors_array_init, num_nodes_vector, ...
            choices_act_funcs_array, choice_loss_func_output, ...
            options_mini_BGD);
        toc;

    elseif choice_opt_method == 2
        para_vector_init = convert_arrays_W_b_to_para_vector...
            (W_matrices_array_init, b_vectors_array_init);

        %---------------- train the neural network model ------------------
        tic;
        DEF_stepSize = 0.001;
        [para_vector_final] = fmin_adam(@(para_vector) ...
            cost_function_Adam_NN(para_vector, ...
            X_train_vector, Y_train_vector, num_nodes_vector, ...
            choices_act_funcs_array, choice_loss_func_output), ...
            para_vector_init, DEF_stepSize);
        toc;

        [W_matrices_array_final, b_vectors_array_final] = ...
            convert_para_vector_to_W_b_arrays(para_vector_final, ...
            num_nodes_vector);

    end

elseif choice_opt_method == 3
    title_NN_method = 'fitnet';

    %-------------- Train the neural network model ------------------------
    tic;
    model_NN_toolbox = fitnet(hiddenLayerSizes);

    if choice_act_func_hidden == 3
        for lay = 1:length(hiddenLayerSizes)
            model_NN_toolbox.layers{lay}.transferFcn = 'tansig';
        end
    end
    if choice_act_func_output == 0
        model_NN_toolbox.layers{end}.transferFcn = 'purelin';
    end

    if choice_loss_func_output == 1
        model_NN_toolbox.performFcn = 'mse';
    end

    model_NN_toolbox.trainFcn = 'trainlm';
    model_NN_toolbox.trainParam.epochs = 1000;

    model_NN_toolbox = train(model_NN_toolbox, X_train_vector, Y_train_vector);
    toc;

end

%================= predict the test data (Estimate m(x)) ==================
if     choice_opt_method == 1 || choice_opt_method == 2
    pred_Y_test_vector = zeros(size(X_test_vector));
    for i = 1:num_obs_test
        [hat_reg_func_vector_test_i, ...
            z_vectors_array_test, a_vectors_array_test] = ...
            forward_pass_within_NN(X_test_vector(i), ...
            W_matrices_array_final, b_vectors_array_final, ...
            choices_act_funcs_array);
        pred_Y_test_vector(i) = hat_reg_func_vector_test_i;
    end

elseif choice_opt_method == 3
    pred_Y_test_vector = model_NN_toolbox(X_test_vector);
end

%=============================== Output ===================================
fig_1 = figure(choice_data);
subplot(2, 2, 1)

plot(X_train_vector, Y_train_vector, 'k.'); hold on;
plot(X_test_vector, pred_Y_test_vector, 'r-', 'LineWidth', 1.5);
if choice_data == 1
    true_m_test_vector = reg_func_true(X_test_vector);
    plot(X_test_vector, true_m_test_vector, 'b--', 'LineWidth', 1.5);
end
if     choice_data == 1
    legend = legend('\textbf{orig.}', '\boldmath{$\widehat{m}(x)$}', ...
        '\boldmath{$m(x)$}');
    set(legend, 'FontWeight', 'bold', 'Location', 'SouthEast', ...
        'Interpreter', 'latex', ...
        'FontSize', 8, 'ItemTokenSize', [20, 20]);
end
if     choice_opt_method == 1
    title({...
        ['\textbf{', title_data, ', ', title_NN_method, '}'], ...
        }, 'Interpreter', 'latex');
elseif choice_opt_method == 2
    title({...
        ['\textbf{', title_data, ', ', title_NN_method, '}'], ...
        }, 'Interpreter', 'latex');
elseif choice_opt_method == 3
    title({...
        ['\textbf{', title_data, ', toolbox ', title_NN_method, '}'], ...
        }, 'Interpreter', 'latex');
end
xlabel('\boldmath{$x$}', 'Interpreter', 'latex');
ylabel('\boldmath{$y$}', 'Interpreter', 'latex');
grid on;
xlim([min(X_train_vector), max(X_train_vector)])

