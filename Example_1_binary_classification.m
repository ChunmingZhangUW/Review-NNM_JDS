%
% Name    : Example_1_binary_classification.m
%--------------------------------------------------------------------------

clear;
close all;

rng(5000);

%==== Part 1: A training dataset (X_train_matrix, Y_class_train_vector) ===
X_train_matrix = [...
    0.1000    0.1000
    0.3000    0.4000
    0.1000    0.5000
    0.6000    0.9000
    0.4000    0.2000
    0.6000    0.3000
    0.5000    0.6000
    0.9000    0.2000
    0.4000    0.4000
    0.7000    0.6000]';
[dim_X, num_obs_train] = size(X_train_matrix);

%-------------------
class_labels = {'A', 'B'};
num_classes = numel(class_labels);

Y_class_train_vector = zeros(num_obs_train, 1);
Y_class_train_vector(1:floor(num_obs_train/2)) = 1;
Y_class_train_vector((floor(num_obs_train/2)+1):num_obs_train) = 2;

%============================= inputs =====================================
choice_opt_method = input([' Input choice_opt_method \n ' ...
    '  (1: mini-BGD; 2: Adam; 3: NonLinear Least-Squares; ' ...
    '5: toolbox patternnet) \n ' ...
    '  in training neural network model \n ' ...
    '  or other methods (4: Logistic Regression) \n ' ...
    '  = ']);

opt_methods = cell(5, 1);
if     choice_opt_method == 1
    opt_methods{choice_opt_method} = 'mini-BGD';

    batch_size = input(['   Input the batch_size \n ' ...
        ['  (m_B=1: SGD; m_B \in [2, n_train-1]: mini-BGD; ' ...
        'm_B=n_train: BGD) \n '], ...
        '  within each minibatch = ']);
    if     batch_size == 1
        title_NN_method = 'SGD';
    elseif 2 <= batch_size && batch_size <= (num_obs_train-1)
        title_NN_method = 'mini-BGD';
    elseif batch_size == num_obs_train
        title_NN_method = 'BGD';
    end
    options_mini_BGD.batch_size = batch_size;

    if batch_size == 1
        specified_num_batches = 1;
        options_mini_BGD.specified_num_batches = specified_num_batches;
    else
        options_mini_BGD.specified_num_batches = [];
    end

elseif choice_opt_method == 2
    opt_methods{choice_opt_method} = 'Adam';

    title_NN_method = 'Adam';

elseif choice_opt_method == 3
    opt_methods{choice_opt_method} = 'NLLS';

    title_NN_method = 'NLLS';

elseif choice_opt_method == 4
    opt_methods{choice_opt_method} = 'LogiReg';

    title_NN_method = 'Logistic Regression';

elseif choice_opt_method == 5
    opt_methods{choice_opt_method} = 'toolbox';

    title_NN_method = 'patternnet';
end

%=================== Part 2: Train the neural network model ===============

if choice_opt_method == 1 || choice_opt_method == 2 ...
        || choice_opt_method == 3 || choice_opt_method == 5

    Y_train_matrix = zeros(num_classes, num_obs_train);
    for i = 1:num_obs_train
        Y_train_matrix(:, i) = ...
            [(Y_class_train_vector(i) == 1);  (Y_class_train_vector(i) == 2)];
    end
    dim_Y = size(Y_train_matrix, 1);

    %--- layers
    if     choice_opt_method == 1 || choice_opt_method == 2 ...
            || choice_opt_method == 3
        hiddenLayerSizes = [2, 2];
    elseif choice_opt_method == 5
        hiddenLayerSizes = [2, 2];
    end
    num_nodes_vector = [dim_X, hiddenLayerSizes, dim_Y];

    num_layers = length(num_nodes_vector);
    if num_layers < 3
        disp('# of layers < 3. return!!!'); return
    end

    %--- activation functions
    choice_act_func_hidden = 3;

    choice_act_func_output = 4;

    choices_act_funcs_array = cell(num_layers, 1);
    for lay = 2 : (num_layers-1)
        choices_act_funcs_array{lay} = choice_act_func_hidden;
    end
    choices_act_funcs_array{num_layers} = choice_act_func_output;

    %--- loss function at output Layer-L
    %%%choice_loss_func_output = 1; % quadratic loss
    choice_loss_func_output = 2; % cross-entropy (neg-log-likelihood)

    if choice_opt_method == 1 || choice_opt_method == 5
        learning_rate_specified = 0.01;
    end
end

F_X_train_matrix = zeros(num_classes, num_obs_train);
if     choice_opt_method == 1 || choice_opt_method == 2 ...
        || choice_opt_method == 3

    cc = 0.5;

    W_matrices_array_init = cell(num_layers, 1);
    b_vectors_array_init  = cell(num_layers, 1);
    for lay = 2 : num_layers
        W_matrices_array_init{lay} = cc * randn(num_nodes_vector(lay), ...
            num_nodes_vector(lay-1));
        b_vectors_array_init{lay}  = zeros(num_nodes_vector(lay), 1);
    end

    if     choice_opt_method == 1
        %------------------- Specify training options ---------------------
        options_mini_BGD.learning_rate = learning_rate_specified;
        options_mini_BGD.num_iterations = 1e6;

        %--------------- train the neural network model -------------------
        tic;
        [W_matrices_array_final, b_vectors_array_final, costs_iters_vector] ...
            = mini_BGD_NN(X_train_matrix, Y_train_matrix, ...
            W_matrices_array_init, b_vectors_array_init, num_nodes_vector, ...
            choices_act_funcs_array, choice_loss_func_output, ...
            options_mini_BGD);
        toc

        %------------------------------------------------------------------
        h_2 = figure(2);
        subplot(1, 1, 1);
        grids_x = (1 : 1e4 : options_mini_BGD.num_iterations);
        semilogy(grids_x, costs_iters_vector(grids_x), 'b-', 'LineWidth', 2);
        xlabel('Iteration Number')
        ylabel('cost function')
        if batch_size == 1 && isempty(specified_num_batches) == 0
            title(['$\textbf{', title_NN_method, '}$', ...
                ', \boldmath{$m_{\mathrm{B}} = ', num2str(batch_size), '$}', ...
                ', \boldmath{$N_{\mathrm{B}} = ', ...
                num2str(specified_num_batches), '$}'], ...
                'interpreter', 'latex', 'FontSize', 25);

        else
            title(['$\textbf{', title_NN_method, '}$', ...
                ', \boldmath{$m_{\mathrm{B}} = ', num2str(batch_size), '$}', ...
                ], ...
                'interpreter', 'latex', 'FontSize', 25);
        end
        set(gca, 'FontWeight', 'Bold', 'FontSize', 20)

    elseif choice_opt_method == 2
        %--------------- train the neural network model -------------------
        para_vector_init = convert_arrays_W_b_to_para_vector...
            (W_matrices_array_init, b_vectors_array_init);

        tic;
        DEF_stepSize = 0.001;
        [para_vector_final] = fmin_adam(@(para_vector) ...
            cost_function_Adam_NN(para_vector, ...
            X_train_matrix, Y_train_matrix, num_nodes_vector, ...
            choices_act_funcs_array, choice_loss_func_output), ...
            para_vector_init, DEF_stepSize);
        toc

    elseif choice_opt_method == 3
        %---------------- train the neural network model ------------------
        para_vector_init = convert_arrays_W_b_to_para_vector...
            (W_matrices_array_init, b_vectors_array_init);

        tic
        [para_vector_final, ~] = lsqnonlin(@(para_vector) ...
            cost_function_NLLS_NN(para_vector, ...
            X_train_matrix, Y_train_matrix, num_nodes_vector, ...
            choices_act_funcs_array, choice_loss_func_output), ...
            para_vector_init);
        toc
    end

    if choice_opt_method == 2 || choice_opt_method == 3
        [W_matrices_array_final, b_vectors_array_final] = ...
            convert_para_vector_to_W_b_arrays(para_vector_final, ...
            num_nodes_vector);
    end

    %----------------------------------------------------------------------
    for i = 1:num_obs_train
        F_X_train_matrix(:, i) = forward_pass_within_NN(X_train_matrix(:, i), ...
            W_matrices_array_final, b_vectors_array_final, ...
            choices_act_funcs_array);
    end

elseif choice_opt_method == 4
    Y_Ber_train_vector = 2 - Y_class_train_vector;

    choice_Logistic_Regression = 2;
    tic
    mdl = fitglm(X_train_matrix', Y_Ber_train_vector', ...
        'Distribution', 'binomial');
    hat_beta_logistic = mdl.Coefficients.Estimate;
    toc

    sigmoid_func = @(z) 1 ./ (1+exp(-z));
    for i = 1:num_obs_train
        design_train_vector = [1; X_train_matrix(:, i)];
        hat_mean_reg_func = ...
            sigmoid_func(hat_beta_logistic' * design_train_vector);

        F_X_train_matrix(:, i) = [hat_mean_reg_func; 1-hat_mean_reg_func];
    end

elseif choice_opt_method == 5
    tic
    model_NN_toolbox = patternnet(hiddenLayerSizes);

    if choice_act_func_hidden == 3
        for lay = 1:length(hiddenLayerSizes)
            model_NN_toolbox.layers{lay}.transferFcn = 'tansig';
        end
    end
    if choice_act_func_output == 4
        model_NN_toolbox.layers{end}.transferFcn = 'softmax';
    end

    if choice_loss_func_output == 2
        model_NN_toolbox.performFcn = 'crossentropy';
    end

    model_NN_toolbox.trainFcn = 'trainscg';
    model_NN_toolbox.trainParam.epochs = 1000;
    model_NN_toolbox.trainParam.goal = 1e-5;
    model_NN_toolbox.trainParam.min_grad = 1e-7;
    model_NN_toolbox.trainParam.lr = learning_rate_specified;
    model_NN_toolbox.trainParam.max_fail = 6;

    model_NN_toolbox = train(model_NN_toolbox, X_train_matrix, Y_train_matrix);
    toc

    F_X_train_matrix = model_NN_toolbox(X_train_matrix);

end

[~, loc_Y_class_train_vector] = max(F_X_train_matrix, [], 1);
pred_Y_class_train_vector = loc_Y_class_train_vector';

%================ Part 3: For a test dataset ==============================
n1_grids = 501; n2_grids = 501;
x1_grids = linspace(0, 1, n1_grids);
x2_grids = linspace(0, 1, n2_grids);
[x1_mesh, x2_mesh] = meshgrid(x1_grids, x2_grids);

x_grid_points  = [x1_mesh(:), x2_mesh(:)]';
F_X_grid_points = zeros(size(x_grid_points));
if     choice_opt_method == 1 || choice_opt_method == 2 || ...
        choice_opt_method == 3
    for i = 1:size(x_grid_points, 2)
        F_X_grid_points(:, i) = forward_pass_within_NN(x_grid_points(:, i), ...
            W_matrices_array_final, b_vectors_array_final, ...
            choices_act_funcs_array);
    end

elseif choice_opt_method == 4
    for i = 1:size(x_grid_points, 2)
        design_test_vector = [1; x_grid_points(:, i)];
        hat_mean_reg_func = ...
            sigmoid_func(hat_beta_logistic' * design_test_vector);

        F_X_grid_points(:, i) = [hat_mean_reg_func; 1-hat_mean_reg_func];
    end

elseif choice_opt_method == 5
    F_X_grid_points = model_NN_toolbox(x_grid_points);
end

[~, pred_Y_class_grid_points] = max(F_X_grid_points, [], 1);
pred_Y_class_mesh = reshape(pred_Y_class_grid_points, size(x1_mesh));

%--------------------------------------------------------------------------
h_3 = figure(3);
subplot(1, 1, 1);
contourf(x1_mesh, x2_mesh, pred_Y_class_mesh, 'LineColor', 'none');
hold on
colormap([0.8 0.8 0.8; 1 1 1]);

plot(...
    X_train_matrix(1, Y_class_train_vector == 1), ...
    X_train_matrix(2, Y_class_train_vector == 1), ...
    'ro', 'MarkerSize', 12, 'LineWidth', 4);
hold on;
plot(...
    X_train_matrix(1, Y_class_train_vector == 2), ...
    X_train_matrix(2, Y_class_train_vector == 2), ...
    'bx', 'MarkerSize', 12, 'LineWidth', 4);

xlabel('\boldmath{$x_1$}', 'interpreter', 'latex', 'FontSize', 25);
ylabel('\boldmath{$x_2$}', 'interpreter', 'latex', 'FontSize', 25);
set(gca, 'XTick', [0 1], 'YTick', [0 1], 'FontWeight', 'Bold', 'FontSize', 20)
xlim([0, 1])
ylim([0, 1])

