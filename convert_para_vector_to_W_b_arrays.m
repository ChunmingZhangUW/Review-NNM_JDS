
function [W_matrices_array, b_vectors_array] = ...
    convert_para_vector_to_W_b_arrays(para_vector, num_nodes_vector)

%--------------------------------------------------------------------------

num_layers = length(num_nodes_vector);
L = num_layers;

num_parameters = length(para_vector);
num_parameters_in_W = ...
    sum( (num_nodes_vector(2: L)) .* (num_nodes_vector(1: L-1)) );

W_matrices_to_vector = para_vector(1 : num_parameters_in_W);
b_vectors_to_vector = para_vector((1+num_parameters_in_W) : num_parameters);

W_matrices_array = cell(num_layers, 1);
d_vector = zeros(L, 1);
I_2 = 0;
for lay = 2 : L
    d_vector(lay) = num_nodes_vector(lay) * num_nodes_vector(lay-1);
    I_1 = I_2 + 1; I_2 = I_2 + d_vector(lay);

    W_matrices_array{lay} = reshape(W_matrices_to_vector(I_1 : I_2), ...
        num_nodes_vector(lay), num_nodes_vector(lay-1));
end

b_vectors_array = cell(num_layers, 1);
d_vector = zeros(L, 1);
I_2 = 0;
for lay = 2 : L
    d_vector(lay) = num_nodes_vector(lay);
    I_1 = I_2 + 1; I_2 = I_2 + d_vector(lay);

    b_vectors_array{lay} = b_vectors_to_vector(I_1 : I_2);
end

end
