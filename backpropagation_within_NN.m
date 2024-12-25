
function [grad_cost_W_matrices_array, grad_cost_b_vectors_array] = ...
    backpropagation_within_NN(a_L_X_vector, Y_vector, ...,
    W_matrices_array, z_vectors_array, a_vectors_array, ...
    choices_act_funcs_array, choice_loss_func_output)

%--------------------------------------------------------------------------

num_layers = size(W_matrices_array, 1);
L = num_layers;

delta_vectors_array = cell(L, 1);

lay = L;
D_matrix_lay = compute_D_matrix_lay(lay, z_vectors_array, ...
    choices_act_funcs_array);

if     choice_loss_func_output == 1
    gradient_vector_loss_C_wrt_a = -(Y_vector - a_L_X_vector);
elseif choice_loss_func_output == 2
    gradient_vector_loss_C_wrt_a = -(Y_vector ./ a_L_X_vector);
end
delta_vectors_array{L} = D_matrix_lay * gradient_vector_loss_C_wrt_a;

for lay = (L-1) : (-1) : 2
    D_matrix_lay = compute_D_matrix_lay(lay, z_vectors_array, ...
        choices_act_funcs_array);

    delta_vectors_array{lay} = D_matrix_lay ...
        * (W_matrices_array{lay+1}' * delta_vectors_array{lay+1});
end

grad_cost_W_matrices_array = cell(L, 1);
grad_cost_b_vectors_array  = cell(L, 1);

for lay = 2 : L
    grad_cost_W_matrices_array{lay} = delta_vectors_array{lay} ...
        * a_vectors_array{lay-1}';

    grad_cost_b_vectors_array{lay}  = delta_vectors_array{lay};
end

%==========================================================================
function D_matrix_lay = compute_D_matrix_lay(lay, z_vectors_array, ...
    choices_act_funcs_array)

%--------------------------------------------------------------------------

n_lay = length(z_vectors_array{lay});

D_vector_lay = zeros(n_lay, 1);
D_matrix_lay = zeros(n_lay, n_lay);
if     choices_act_funcs_array{lay} == 0 || ...
        choices_act_funcs_array{lay} == 1 || ...
        choices_act_funcs_array{lay} == 2 || ...
        choices_act_funcs_array{lay} == 3
    for j = 1:n_lay
        D_vector_lay(j, 1) = activation_func_NN(...
            choices_act_funcs_array{lay}, z_vectors_array{lay}(j), 1);
    end
    D_matrix_lay = diag(D_vector_lay);

elseif choices_act_funcs_array{lay} == 4
    D_vector_lay = activation_func_NN(...
        choices_act_funcs_array{lay}, z_vectors_array{lay}, 0);
    D_matrix_lay = diag(D_vector_lay) - D_vector_lay * D_vector_lay';
end

