
function [o, h_new] = output_gate(h_old, x, C_new, W_o, b_o)
%--------------------------------------------------------------------------
% Description:
% This function compute the new hidden state.
% Called: sigmoid.m, tanh.m
%--------------------------------------------------------------------------
% Input:
% h_old: Old hidden state
% x: Current input data
% C_new: New cell state
% W_o: Weight used to compute o
% b_o: bias used to compute o
%--------------------------------------------------------------------------
% Output:
% o: Output gate output
% h_new: New hidden state
%--------------------------------------------------------------------------

o     = sigmoid(h_old * W_o{1} + x * W_o{2} + b_o);
h_new = o .* tanh(C_new);

end

