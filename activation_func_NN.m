
function activation_func_val = activation_func_NN(...
    choice_act_func, z_vector, deri)

%--------------------------------------------------------------------------

activation_func_val = zeros(size(z_vector));

if     choice_act_func == 0
    activation_func_o = z_vector;

    if     deri == 0
        activation_func_val = activation_func_o;
    elseif deri == 1
        activation_func_val = 1;
    end

elseif choice_act_func == 1
    activation_func_o = 1 ./ (1+exp(-z_vector));

    if     deri == 0
        activation_func_val = activation_func_o;
    elseif deri == 1
        activation_func_val = activation_func_o .* (1 - activation_func_o);
    end

elseif choice_act_func == 2
    activation_func_o = max(z_vector, 0);

    if     deri == 0
        activation_func_val = activation_func_o;
    elseif deri == 1
        activation_func_val = (z_vector > 0) + 0 * (z_vector < 0);
    end

elseif choice_act_func == 3
    sigmoid_func = @(z) 1 ./ (1+exp(-z));
    activation_func_o = 2 * sigmoid_func(2*z_vector) - 1;

    if     deri == 0
        activation_func_val = activation_func_o;
    elseif deri == 1
        activation_func_val = 1 - activation_func_o.^2;
    end

    %----------------------------------------------------------------------
elseif choice_act_func == 4

    if length(z_vector) >= 2
        activation_func_o = exp(z_vector) ./ sum(exp(z_vector));

        if deri == 0
            activation_func_val = activation_func_o;
        end
    end
end

end
