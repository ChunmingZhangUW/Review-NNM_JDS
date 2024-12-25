%
% Name:   Realdata_MNIST_CNN.m
%--------------------------------------------------------------------------

clear;
close all;

rng(2024);

%----------------------- Load the MNIST training dataset ------------------
[X_Train_array, Y_Train_array] = digitTrain4DArrayData;

num_classes = 10;
dim_X = size(X_Train_array, 1:3);
dim_Y = num_classes;

%---------------------- visualize sample images ---------------------------
ID_sample_Images_Train = zeros(num_classes, 1);
freq_train_vector      = zeros(num_classes, 1);
for k = 1:num_classes
    digit = k - 1;
    ID_sample_Images_Train(k) = find(Y_Train_array == categorical(digit), 1);
    freq_train_vector(k)      = sum( Y_Train_array == categorical(digit), 1);
end

h_1 = figure(1);
tiledlayout(2, 5)

for k = 1:num_classes
    i = ID_sample_Images_Train(k);
    nexttile
    sampleImage_train = X_Train_array(:, :, 1, i);
    imshow(sampleImage_train);
    title(['Digit: ', char(Y_Train_array(i, 1))]);
end

%----------------------- Load the MNIST test dataset ----------------------
[X_Test_array,  Y_Test_array]  = digitTest4DArrayData;

%---------------------- visualize sample images ---------------------------
ID_sample_Images_Test = zeros(num_classes, 1);
freq_test_vector      = zeros(num_classes, 1);
for k = 1:num_classes
    digit = k - 1;
    ID_sample_Images_Test(k) = find(Y_Test_array == categorical(digit), 1);
    freq_test_vector(k)      = sum( Y_Test_array == categorical(digit), 1);
end

h_2 = figure(2);
tiledlayout(2, 5)

for  k = 1:num_classes
    i = ID_sample_Images_Test(k);
    nexttile
    sampleImage_test = X_Test_array(:, :, :, i);
    imshow(sampleImage_test);
    title(['Digit: ', char(Y_Test_array(i))]);
end

%=========================== Train the CNN model ==========================
name_method = 'toolbox'; %name_method_title = 'Toolbox';

%------- Define the CNN architecture: imageInputLayer -----------------
img_input_layer_CNN = [
    imageInputLayer([size(X_Train_array, 1:3)], 'Name', 'input')

    convolution2dLayer(3,  8, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')

    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')

    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')

    fullyConnectedLayer(64, 'Name', 'fc1')

    reluLayer('Name', 'relu3')

    fullyConnectedLayer(num_classes, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')];

%------- Specify training options: trainingOptions --------------------
choice_training_options_CNN = 1.2;
if choice_training_options_CNN == 1.2
    training_options_CNN = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.01, ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 32, ...
        'Momentum', 0.9, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'training-progress');

end

%------------------------ train the neural network ------------------------
model_CNN_toolbox = trainNetwork(X_Train_array, Y_Train_array, ...
    img_input_layer_CNN, training_options_CNN);

%================ Predict labels for images in test data ==================
pred_Y_Test = classify(model_CNN_toolbox, X_Test_array);

accuracy_pred_test = mean( pred_Y_Test == Y_Test_array );

%------------------------------- output -------------------------------
disp(accuracy_pred_test)

