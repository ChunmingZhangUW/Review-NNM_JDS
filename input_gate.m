
function [i, Cd] = input_gate(h_old, x, W_i, b_i, W_Cd, b_Cd)
%--------------------------------------------------------------------------
% Description:
% This function decides what to be stored in the new cell state.
% Called: sigmoid.m, tanh.m 
%--------------------------------------------------------------------------
% Input:
% h_old: Old hidden state
% x: Current input data
% W_i: Weight used to compute i
% b_i: bias used to compute i
% W_Cd: Weight used to compute Cd
% b_Cd: bias used to compute Cd
%--------------------------------------------------------------------------
% Output:
% i: Input gate output
% Cd: Candidate array
%--------------------------------------------------------------------------

i  = sigmoid(h_old * W_i{1} + x * W_i{2} + b_i);
Cd = tanh(h_old * W_Cd{1} + x * W_Cd{2} + b_Cd);

end % of a function