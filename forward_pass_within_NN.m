
function [a_L_X_vector, z_vectors_array, a_vectors_array]...
    = forward_pass_within_NN(...
    X_vector, W_matrices_array, b_vectors_array, ...
    choices_act_funcs_array)

%--------------------------------------------------------------------------

num_layers = size(b_vectors_array, 1);
L = num_layers;

z_vectors_array = cell(L, 1);
a_vectors_array = cell(L, 1);

a_vectors_array{1} = X_vector;

for lay = 2 : L
    z_vectors_array{lay} = ...
        W_matrices_array{lay} * a_vectors_array{lay-1} + b_vectors_array{lay};
    a_vectors_array{lay} = ...
        activation_func_NN(choices_act_funcs_array{lay}, z_vectors_array{lay}, 0);
end

a_L_X_vector = a_vectors_array{L};

end

