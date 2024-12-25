
function [W_matrices_array_final, b_vectors_array_final, costs_iters_vector] ...
    = mini_BGD_NN(X_matrix, Y_matrix, ...
    W_matrices_array_init, b_vectors_array_init, num_nodes_vector, ...
    choices_act_funcs_array, choice_loss_func_output, options_mini_BGD)

%--------------------------------------------------------------------------

learning_rate  = options_mini_BGD.learning_rate;
num_iterations = options_mini_BGD.num_iterations;
batch_size     = options_mini_BGD.batch_size;
m_B = batch_size;
specified_num_batches = options_mini_BGD.specified_num_batches;

num_obs = size(X_matrix, 2);
num_layers = length(num_nodes_vector);
L = num_layers;

costs_iters_vector = zeros(num_iterations, 1);

W_matrices_array = W_matrices_array_init;
b_vectors_array  = b_vectors_array_init;
for iter = 1:num_iterations
    %disp([' iter = ', num2str(iter)]);
    if     m_B <= num_obs-1
        k_vector_mini_BGD = randperm(num_obs);
        X_matrix_shuffled = X_matrix(:, k_vector_mini_BGD);
        Y_matrix_shuffled = Y_matrix(:, k_vector_mini_BGD);
    elseif m_B == num_obs
        X_matrix_shuffled = X_matrix;
        Y_matrix_shuffled = Y_matrix;
    end

    ID_vector_init_in_batches = 1:m_B:num_obs;
    N_B_full = length(ID_vector_init_in_batches);

    if     isempty(specified_num_batches) == 1
        used_N_B = N_B_full;
    elseif isempty(specified_num_batches) == 0
        used_N_B = specified_num_batches;
    end

    for j = 1:used_N_B
        B_j = ID_vector_init_in_batches(j) : ...
            min(ID_vector_init_in_batches(j) + m_B - 1, num_obs);

        s_grad_W_array = cell(L, 1);
        s_grad_b_array = cell(L, 1);
        for lay = 2 : L
            s_grad_W_array{lay} = zeros(num_nodes_vector(lay), ...
                num_nodes_vector(lay-1));
            s_grad_b_array{lay} = zeros(num_nodes_vector(lay), 1);
        end
        for k = 1:length(B_j)
            i = B_j(k);
            X_i_vector = X_matrix_shuffled(:, i);
            Y_i_vector = Y_matrix_shuffled(:, i);

            [grad_cost_W_matrices_array, grad_cost_b_vectors_array] = ...
                gradient_loss_function_NN(X_i_vector, Y_i_vector, ...
                W_matrices_array, b_vectors_array, ...
                choices_act_funcs_array, choice_loss_func_output);

            for lay = 2 : L
                s_grad_W_array{lay} = s_grad_W_array{lay} ...
                    + grad_cost_W_matrices_array{lay};
                s_grad_b_array{lay} = s_grad_b_array{lay} ...
                    + grad_cost_b_vectors_array{lay};
            end
        end

        for lay = 2 : L
            grad_cost_W_matrices_array{lay} = s_grad_W_array{lay} / length(B_j);
            grad_cost_b_vectors_array{lay}  = s_grad_b_array{lay} / length(B_j);
        end

        for lay = 2 : L
            W_matrices_array{lay} = W_matrices_array{lay} ...
                - learning_rate * grad_cost_W_matrices_array{lay};
            b_vectors_array{lay} = b_vectors_array{lay}   ...
                - learning_rate * grad_cost_b_vectors_array{lay};
        end
    end

    costs_iters_vector(iter) = cost_function_NN(X_matrix, Y_matrix, ...
        W_matrices_array, b_vectors_array, ...
        choices_act_funcs_array, choice_loss_func_output);

end

W_matrices_array_final = W_matrices_array;
b_vectors_array_final  = b_vectors_array;

end
