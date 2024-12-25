
function f = forget_gate(h_old, x, W_f, b_f)
%--------------------------------------------------------------------------
% Description:
% This function decides what to be forgotten from the old cell state.
% Called: sigmoid.m 
%--------------------------------------------------------------------------
% Input:
% h_old: Old hidden state
% x: Current input data
% W_f: Weight used to compute f
% b_f: bias used to compute f
%--------------------------------------------------------------------------
% Output:
% f: Forget gate output
%--------------------------------------------------------------------------

f = sigmoid(h_old * W_f{1} + x * W_f{2} + b_f);

end % of a function