
function [cost_val, loss_vector_NLLS] = cost_function_NN(...
    X_matrix, Y_matrix, W_matrices_array, b_vectors_array, ...
    choices_act_funcs_array, choice_loss_func_output)

%--------------------------------------------------------------------------

num_obs = size(X_matrix, 2);

loss_vector_gene = zeros(num_obs, 1);
for i = 1:num_obs
    X_i_vector = X_matrix(:, i);
    Y_i_vector = Y_matrix(:, i);

    a_L_X_i_vector = forward_pass_within_NN(X_i_vector, ...
        W_matrices_array, b_vectors_array, ...
        choices_act_funcs_array);

    if     choice_loss_func_output == 1
        loss_vector_gene(i) = sum( (Y_i_vector - a_L_X_i_vector).^2 );
    elseif choice_loss_func_output == 2
        loss_vector_gene(i) = ...
            - Y_i_vector' * log( max(eps, min(a_L_X_i_vector, 1-eps)) );
    end
end
loss_vector_NLLS = sqrt(loss_vector_gene);

if     choice_loss_func_output == 1
    cost_val = 1/2 * mean(loss_vector_gene);
elseif choice_loss_func_output == 2
    cost_val = mean(loss_vector_gene);
end

end

