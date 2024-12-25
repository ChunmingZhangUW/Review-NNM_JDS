
function [cost_val, grad_cost_para_vector] = cost_function_Adam_NN(...
    para_vector, ...
    X_matrix, Y_matrix, num_nodes_vector, ...
    choices_act_funcs_array, choice_loss_func_output)

%--------------------------------------------------------------------------

num_obs = size(X_matrix, 2);
num_layers = length(num_nodes_vector);
L = num_layers;

[W_matrices_array, b_vectors_array] = convert_para_vector_to_W_b_arrays...
    (para_vector, num_nodes_vector);

[cost_val, ~] = cost_function_NN(X_matrix, Y_matrix, ...
    W_matrices_array, b_vectors_array, ...
    choices_act_funcs_array, choice_loss_func_output);

s_grad_W_array = cell(L, 1);
s_grad_b_array = cell(L, 1);
for lay = 2 : L
    s_grad_W_array{lay} = zeros(num_nodes_vector(lay), ...
        num_nodes_vector(lay-1));
    s_grad_b_array{lay} = zeros(num_nodes_vector(lay), 1);
end
for i = 1:num_obs
    X_i_vector = X_matrix(:, i);
    Y_i_vector = Y_matrix(:, i);

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
    grad_cost_W_matrices_array{lay} = s_grad_W_array{lay} / num_obs;
    grad_cost_b_vectors_array{lay}  = s_grad_b_array{lay} / num_obs;
end

grad_cost_para_vector = convert_arrays_W_b_to_para_vector...
    (grad_cost_W_matrices_array, grad_cost_b_vectors_array);

end
