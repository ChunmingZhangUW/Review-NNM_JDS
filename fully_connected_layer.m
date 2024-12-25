
function vec_out = fully_connected_layer(vec_in, W_fc, b_fc)
%--------------------------------------------------------------------------
% Description:
% Fully connected layer.
%--------------------------------------------------------------------------
% Input:
% vec_in: 1*d array like
% W_fc: Weight used to compute vec_out
% b_fc: Bias used to compute vec_out
%--------------------------------------------------------------------------
% Output:
% vec_out: Output vector
%--------------------------------------------------------------------------

vec_out = vec_in * W_fc + b_fc;

end % of a function

