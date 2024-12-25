
function y = sigmoid(x)
%--------------------------------------------------------------------------
% Description:
% The sigmoid activation operation applies the sigmoid function to the input data.
%--------------------------------------------------------------------------
% Input:
% x: Input
%--------------------------------------------------------------------------
% Output:
% y: Output
%--------------------------------------------------------------------------

y = 1 ./ (1+exp(-x)); % sigmoid function

end