% LSTM_ourcode.m
%--------------------------------------------------------------------------

classdef LSTM_ourcode < handle
    % Description:
    % LSTM model

    properties
        target
        h_0
        C_0
        W_f
        b_f
        W_i
        b_i
        W_Cd
        b_Cd
        W_o
        b_o
        W_fc1
        b_fc1
        W_fc2
        b_fc2
        W_fc3
        b_fc3
    end

    methods
        function obj = LSTM_ourcode(target)
            % Description:
            % Instantialization.
            %
            % Input:
            % target: What will the model be used for. String.
            %         Either "classification" or "regression".

            obj.target = target;
        end

        function detail(obj)
            % Description:
            % Show details of the model.

        end

        function loss = train(obj,x_train,y_train,ops)
            % Description:
            % Train the model using SGD.
            %
            % Input:
            % x_train: Data for training. 1*n or n*1 cell.
            %          Each cell should contain a matrix whose columns are sorted
            %          by the sequence order and rows are the features.
            % y_train: True values of the x_train. 1*n or n*1 cell.
            %          Each cell should contain a one-hot code (an array) if
            %          classification and a 1 * length of y double array if regression.
            % loss_func: Loss function. Default Cross Entropy Loss if
            %            classification and Mean Squared Error if regression.
            %            Not modifiable this version.
            % b: Batch size. Default 16.
            % lr: Learning rate. Default 0.0001.
            % dc_lr: If decreasing learning rate or not. Logical. Default false.
            % max_epochs: Maximum training epochs. Default 1000.
            % min_loss: Minimum training loss. Default 0.001.
            % dim_h: The dimension of hidden state. Default 128.
            % dim_fc1: The dimension of the output array of the first fully
            %          connected layer. Default 64.
            % dim_fc2: The dimension of the output array of the second fully
            %          connected layer. Default 32.
            % dropout: Dropout probability for fully connected layers.
            %          Between 0 and 1. Default 0.
            % lambda_L2: The coefficient of L2 regularization term. Default 0.
            % h_0: Initial hidden state.
            %      1 * dim_h array. Default randn().
            % C_0: Initial cell state.
            %      1 * dim_h array. Default randn().
            % W_f: Weight used to compute f in the forget gate. 2*1 cell.
            %      W_f{1}: dim_h * dim_h matrix. Default randn().
            %      W_f{2}: number of features * dim_h matrix. Default randn().
            % b_f: bias used to compute f in the forget gate.
            %      1 * dim_h array. Default randn().
            % W_i: Weight used to compute i in the input gate. 2*1 cell.
            %      W_i{1}: dim_h * dim_h matrix. Default randn().
            %      W_i{2}: number of features * dim_h matrix. Default randn().
            % b_i: bias used to compute i in the input gate.
            %      1 * dim_h array. Default randn().
            % W_Cd: Weight used to compute Cd in the input gate. 2*1 cell.
            %      W_Cd{1}: dim_h * dim_h matrix. Default randn().
            %      W_Cd{2}: number of features * dim_h matrix. Default randn().
            % b_Cd: bias used to compute Cd in the input gate.
            %      1 * dim_h array. Default randn().
            % W_o: Weight used to compute o in the output gate. 2*1 cell.
            %      W_o{1}: dim_h * dim_h matrix. Default randn().
            %      W_o{2}: number of features * dim_h matrix. Default randn().
            % b_o: bias used to compute o in the output gate.
            %      1 * dim_h array. Default randn().
            % W_fc1: Weight used in the first fully connected layer.
            %        dim_h * dim_fc1 matrix. Default randn().
            % b_fc1: Bias used in the first fully connected layer.
            %        1 * dim_fc1 array. Default randn().
            % W_fc2: Weight used in the second fully connected layer.
            %        dim_fc1 * dim_fc2 matrix. Default randn().
            % b_fc2: Bias used in the second fully connected layer
            %        1 * dim_fc2 array. Default randn().
            % W_fc3: Weight used in the third fully connected layer.
            %        dim_fc2 * length of y matrix. Default randn().
            % b_fc3: Bias used in the second fully connected layer
            %        1 * length of y array. Default randn().
            %
            % Output:
            % loss: Loss for every epoch. 1 * number of training epochs array.

            % Set default values
            arguments
                obj
                x_train cell
                y_train cell
                % ops.loss_func
                ops.b (1,1) double = 16
                ops.lr (1,1) double = 0.0001
                ops.dc_lr (1,1) logical = false
                ops.max_epochs (1,1) double = 1000
                ops.min_loss (1,1) double = 0.001
                ops.dim_h (1,1) double = 128
                ops.dim_fc1 (1,1) double = 64
                ops.dim_fc2 (1,1) double = 32
                ops.dropout (1,1) double = 0
                ops.lambda_L2 (1,1) double = 0
                ops.h_0 double = []
                ops.C_0 double = []
                ops.W_f cell = {}
                ops.b_f double = []
                ops.W_i cell = {}
                ops.b_i double = []
                ops.W_Cd cell = {}
                ops.b_Cd double = []
                ops.W_o cell = {}
                ops.b_o double = []
                ops.W_fc1 double = []
                ops.b_fc1 double = []
                ops.W_fc2 double = []
                ops.b_fc2 double = []
                ops.W_fc3 double = []
                ops.b_fc3 double = []
            end

            n_fea = size(x_train{1},1); % number of features
            len_y = size(y_train{1},2); % length of y

            if isempty(ops.h_0)
                ops.h_0 = randn(1,ops.dim_h);
            end

            if isempty(ops.C_0)
                ops.C_0 = randn(1,ops.dim_h);
            end

            if isempty(ops.W_f)
                ops.W_f{1} = randn(ops.dim_h,ops.dim_h);
                ops.W_f{2} = randn(n_fea,ops.dim_h);
            end

            if isempty(ops.b_f)
                ops.b_f = randn(1,ops.dim_h);
            end

            if isempty(ops.W_i)
                ops.W_i{1} = randn(ops.dim_h,ops.dim_h);
                ops.W_i{2} = randn(n_fea,ops.dim_h);
            end

            if isempty(ops.b_i)
                ops.b_i = randn(1,ops.dim_h);
            end

            if isempty(ops.W_Cd)
                ops.W_Cd{1} = randn(ops.dim_h,ops.dim_h);
                ops.W_Cd{2} = randn(n_fea,ops.dim_h);
            end

            if isempty(ops.b_Cd)
                ops.b_Cd = randn(1,ops.dim_h);
            end

            if isempty(ops.W_o)
                ops.W_o{1} = randn(ops.dim_h,ops.dim_h);
                ops.W_o{2} = randn(n_fea,ops.dim_h);
            end

            if isempty(ops.b_o)
                ops.b_o = randn(1,ops.dim_h);
            end

            if isempty(ops.W_fc1)
                ops.W_fc1 = randn(ops.dim_h,ops.dim_fc1);
            end

            if isempty(ops.b_fc1)
                ops.b_fc1 = randn(1,ops.dim_fc1);
            end

            if isempty(ops.W_fc2)
                ops.W_fc2 = randn(ops.dim_fc1,ops.dim_fc2);
            end

            if isempty(ops.b_fc2)
                ops.b_fc2 = randn(1,ops.dim_fc2);
            end

            if isempty(ops.W_fc3)
                ops.W_fc3 = randn(ops.dim_fc2,len_y);
            end

            if isempty(ops.b_fc3)
                ops.b_fc3 = randn(1,len_y);
            end

            % Train
            if obj.target == "classification"
                % to be complemented
            elseif obj.target == "regression"
                n = length(x_train); % number of training data
                loss = inf; % training loss
                start_time = datetime("now","Format","dd-MMM-uuuu HH:mm:ss");
                for epoch = 1:ops.max_epochs
                    if loss(end) <= ops.min_loss
                        break
                    end

                    if mod(epoch,10) == 0
                        fprintf("Epoch: %d    ",epoch);
                    end

                    if (ops.dc_lr == true && epoch > 1)
                        ops.lr = ops.lr - (ops.lr-ops.lr/10)/ops.max_epochs;
                    end

                    dpt_fc1 = rand(1,size(ops.W_fc1,2)) > ops.dropout;
                    W_fc1_dpt = ops.W_fc1(:,dpt_fc1);
                    b_fc1_dpt = ops.b_fc1(:,dpt_fc1);

                    dpt_fc2 = rand(1,size(ops.W_fc2,2)) > ops.dropout;
                    W_fc2_dpt = ops.W_fc2(dpt_fc1,dpt_fc2);
                    b_fc2_dpt = ops.b_fc2(:,dpt_fc2);

                    W_fc3_dpt = ops.W_fc3(dpt_fc2,:);
                    b_fc3_dpt = ops.b_fc3;

                    l = 0;
                    dW_fh = 0; dW_fx = 0; db_f = 0;
                    dW_ih = 0; dW_ix = 0; db_i = 0;
                    dW_Cdh = 0; dW_Cdx = 0; db_Cd = 0;
                    dW_oh = 0; dW_ox = 0; db_o = 0;
                    dW_fc1 = 0; dW_fc2 = 0; dW_fc3 = 0;
                    db_fc1 = 0; db_fc2 = 0; db_fc3 = 0;

                    idxs = randperm(n); % shuffle
                    idxs = reshape(idxs(1:(n-mod(n,ops.b))), [floor(n/ops.b),ops.b]); % discard excess
                    for batch = 1:size(idxs,1)
                        for idx = idxs(batch,:)
                            x = x_train{idx}';
                            h = ops.h_0;
                            C = ops.C_0;
                            f_group = {}; i_group = {}; Cd_group = {};
                            C_group = {}; o_group = {}; h_group = {};

                            % Forward pass
                            for t = 1:size(x,1)
                                f = forget_gate(h,x(t,:),ops.W_f,ops.b_f);
                                [i,Cd] = input_gate(h,x(t,:),ops.W_i,ops.b_i,ops.W_Cd,ops.b_Cd);
                                C = update_cell_state(C,f,i,Cd);
                                [o,h] = output_gate(h,x(t,:),C,ops.W_o,ops.b_o);
                                % save for backward pass
                                f_group{t} = f; i_group{t} = i; Cd_group{t} = Cd;
                                C_group{t} = C; o_group{t} = o; h_group{t} = h;
                            end
                            y1 = fully_connected_layer(h,W_fc1_dpt,b_fc1_dpt);
                            y2 = fully_connected_layer(y1,W_fc2_dpt,b_fc2_dpt);
                            y = fully_connected_layer(y2,W_fc3_dpt,b_fc3_dpt);
                            l = l + (y-y_train{idx})*(y-y_train{idx})'/len_y; % SSE

                            % Backward pass
                            for t = size(x,1):-1:1
                                if t == size(x,1)
                                    dh = 2*(y-y_train{idx})*W_fc3_dpt'*W_fc2_dpt'*W_fc1_dpt';
                                    dC = dh.*o_group{t}.*(1-tanh(C_group{t}).^2);
                                else
                                    dh = df*ops.W_f{1}' + di*ops.W_i{1}' + dCd*ops.W_Cd{1}' + do*ops.W_o{1}';
                                    dC = dh.*o_group{t}.*(1-tanh(C_group{t}).^2) + dC.*f_group{t+1};
                                end

                                if t == 1
                                    df = dC.*ops.C_0.*f_group{t}.*(1-f_group{t});
                                else
                                    df = dC.*C_group{t-1}.*f_group{t}.*(1-f_group{t});
                                end
                                di = dC.*Cd_group{t}.*i_group{t}.*(1-i_group{t});
                                dCd = dC.*i_group{t}.*(1-Cd_group{t}.^2);
                                do = dh.*tanh(C_group{t}).*o_group{t}.*(1-o_group{t});

                                if t == 1
                                    dW_fh = dW_fh + ops.h_0'*df;
                                    dW_ih = dW_ih + ops.h_0'*di;
                                    dW_Cdh = dW_Cdh + ops.h_0'*dCd;
                                    dW_oh = dW_oh + ops.h_0'*do;
                                else
                                    dW_fh = dW_fh + h_group{t-1}'*df;
                                    dW_ih = dW_ih + h_group{t-1}'*di;
                                    dW_Cdh = dW_Cdh + h_group{t-1}'*dCd;
                                    dW_oh = dW_oh + h_group{t-1}'*do;
                                end

                                dW_fx = dW_fx + x(t,:)'*df;
                                dW_ix = dW_ix + x(t,:)'*di;
                                dW_Cdx = dW_Cdx + x(t,:)'*dCd;
                                dW_ox = dW_ox + x(t,:)'*do;

                                db_f = db_f + df;
                                db_i = db_i + di;
                                db_Cd = db_Cd + dCd;
                                db_o = db_o + do;
                            end

                            dW_fc1 = dW_fc1 + h'*2*(y-y_train{idx})*W_fc3_dpt'*W_fc2_dpt';
                            dW_fc2 = dW_fc2 + y1'*2*(y-y_train{idx})*W_fc3_dpt';
                            dW_fc3 = dW_fc3 + y2'*2*(y-y_train{idx});

                            db_fc1 = db_fc1 + 2*(y-y_train{idx})*W_fc3_dpt'*W_fc2_dpt';
                            db_fc2 = db_fc2 + 2*(y-y_train{idx})*W_fc3_dpt';
                            db_fc3 = db_fc3 + 2*(y-y_train{idx});
                        end

                        L2 = ops.W_f{1}(:)'*ops.W_f{1}(:) + ...
                            ops.W_i{1}(:)'*ops.W_i{1}(:) + ...
                            ops.W_Cd{1}(:)'*ops.W_Cd{1}(:) + ...
                            ops.W_o{1}(:)'*ops.W_o{1}(:) + ...
                            ops.W_f{2}(:)'*ops.W_f{2}(:) + ...
                            ops.W_i{2}(:)'*ops.W_i{2}(:) + ...
                            ops.W_Cd{2}(:)'*ops.W_Cd{2}(:) + ...
                            ops.W_o{2}(:)'*ops.W_o{2}(:) + ...
                            ops.W_fc1(:)'*ops.W_fc1(:) + ...
                            ops.W_fc2(:)'*ops.W_fc2(:) + ...
                            ops.W_fc3(:)'*ops.W_fc3(:);

                        % Update parameters
                        ops.W_f{1} = ops.W_f{1} - ops.lr*(dW_fh/ops.b + 2*ops.lambda_L2*ops.W_f{1});
                        ops.W_i{1} = ops.W_i{1} - ops.lr*(dW_ih/ops.b + 2*ops.lambda_L2*ops.W_i{1});
                        ops.W_Cd{1} = ops.W_Cd{1} - ops.lr*(dW_Cdh/ops.b + 2*ops.lambda_L2*ops.W_Cd{1});
                        ops.W_o{1} = ops.W_o{1} - ops.lr*(dW_oh/ops.b + 2*ops.lambda_L2*ops.W_o{1});

                        ops.W_f{2} = ops.W_f{2} - ops.lr*(dW_fx/ops.b + 2*ops.lambda_L2*ops.W_f{2});
                        ops.W_i{2} = ops.W_i{2} - ops.lr*(dW_ix/ops.b + 2*ops.lambda_L2*ops.W_i{2});
                        ops.W_Cd{2} = ops.W_Cd{2} - ops.lr*(dW_Cdx/ops.b + 2*ops.lambda_L2*ops.W_Cd{2});
                        ops.W_o{2} = ops.W_o{2} - ops.lr*(dW_ox/ops.b + 2*ops.lambda_L2*ops.W_o{2});

                        ops.b_f = ops.b_f - ops.lr*db_f/ops.b;
                        ops.b_i = ops.b_i - ops.lr*db_i/ops.b;
                        ops.b_Cd = ops.b_Cd - ops.lr*db_Cd/ops.b;
                        ops.b_o = ops.b_o - ops.lr*db_o/ops.b;

                        ops.W_fc1(:,dpt_fc1) = ops.W_fc1(:,dpt_fc1) - ops.lr*(dW_fc1/ops.b + 2*ops.lambda_L2*ops.W_fc1(:,dpt_fc1));
                        ops.W_fc2(dpt_fc1,dpt_fc2) = ops.W_fc2(dpt_fc1,dpt_fc2) - ops.lr*(dW_fc2/ops.b + 2*ops.lambda_L2*ops.W_fc2(dpt_fc1,dpt_fc2));
                        ops.W_fc3(dpt_fc2,:) = ops.W_fc3(dpt_fc2,:) - ops.lr*(dW_fc3/ops.b + 2*ops.lambda_L2*ops.W_fc3(dpt_fc2,:));

                        ops.b_fc1(:,dpt_fc1) = ops.b_fc1(:,dpt_fc1) - ops.lr*db_fc1/ops.b;
                        ops.b_fc2(:,dpt_fc2) = ops.b_fc2(:,dpt_fc2) - ops.lr*db_fc2/ops.b;
                        ops.b_fc3 = ops.b_fc3 - ops.lr*db_fc3/ops.b;
                    end

                    loss = [loss,l/(n-mod(n,ops.b))]; % MSE
                    % loss = [loss,l/ops.b + ops.lambda_L2*L2]; % MSE + L2

                    if mod(epoch,10) == 0
                        end_time = datetime("now","Format","dd-MMM-uuuu HH:mm:ss");
                        fprintf("Loss: %f    Time: %s\n",loss(end),end_time - start_time);
                        start_time = datetime("now","Format","dd-MMM-uuuu HH:mm:ss");
                    end
                end

                % Save parameters
                obj.h_0 = ops.h_0;
                obj.C_0 = ops.C_0;
                obj.W_f = ops.W_f;
                obj.b_f = ops.b_f;
                obj.W_i = ops.W_i;
                obj.b_i = ops.b_i;
                obj.W_Cd = ops.W_Cd;
                obj.b_Cd = ops.b_Cd;
                obj.W_o = ops.W_o;
                obj.b_o = ops.b_o;
                obj.W_fc1 = ops.W_fc1;
                obj.b_fc1 = ops.b_fc1;
                obj.W_fc2 = ops.W_fc2;
                obj.b_fc2 = ops.b_fc2;
                obj.W_fc3 = ops.W_fc3;
                obj.b_fc3 = ops.b_fc3;

                loss(1) = []; % delete inf
            end
        end

        function y_pred = predict(obj,x_test)
            % Description:
            % To predict y for test data.
            %
            % Input:
            % x_test: Data for testing. 1*n or n*1 cell.
            %         Each cell should contain a matrix whose rows are sorted
            %         by the sequence order and columns are the features.
            %
            % Output:
            % y_pred: Prediction on x_test. 1*n cell.

            y_pred = {};
            if obj.target == "classification"
                % to be complemented
            elseif obj.target == "regression"
                for idx = 1:length(x_test)
                    x = x_test{idx}';
                    h = obj.h_0;
                    C = obj.C_0;
                    for t = 1:size(x,1)
                        f = forget_gate(h,x(t,:),obj.W_f,obj.b_f);
                        [i,Cd] = input_gate(h,x(t,:),obj.W_i,obj.b_i,obj.W_Cd,obj.b_Cd);
                        C = update_cell_state(C,f,i,Cd);
                        [~,h] = output_gate(h,x(t,:),C,obj.W_o,obj.b_o);
                    end
                    y1 = fully_connected_layer(h,obj.W_fc1,obj.b_fc1);
                    y2 = fully_connected_layer(y1,obj.W_fc2,obj.b_fc2);
                    y = fully_connected_layer(y2,obj.W_fc3,obj.b_fc3);
                    y_pred{idx,1} = y;
                end
            end
        end
    end
end

