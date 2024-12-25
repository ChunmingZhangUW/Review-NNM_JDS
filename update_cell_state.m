
function C_new = update_cell_state(C_old, f, i, Cd)
%--------------------------------------------------------------------------
% Description:
% This function updates the cell state.
%--------------------------------------------------------------------------
% Input:
% C_old: Old cell state
% f: Forget gate output
% i: Input gate output
% Cd: Candidate array
%--------------------------------------------------------------------------
% Output:
% C_new: New cell state
%--------------------------------------------------------------------------

C_new = f .* C_old + i .* Cd;

end