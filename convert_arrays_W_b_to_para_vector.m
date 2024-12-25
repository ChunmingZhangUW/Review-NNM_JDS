
function para_vector = convert_arrays_W_b_to_para_vector...
    (W_matrices_array, b_vectors_array)

%--------------------------------------------------------------------------

num_layers = size(b_vectors_array, 1);
L = num_layers;

W_matrices_to_vector = [];
b_vectors_to_vector  = [];
for lay = 2 : L
    W_matrices_to_vector = [ W_matrices_to_vector; ...
        reshape(W_matrices_array{lay}, [], 1) ];
    b_vectors_to_vector  = [ b_vectors_to_vector; ...
        b_vectors_array{lay} ];
end

para_vector = [...
    W_matrices_to_vector; ...
    b_vectors_to_vector];

end
