function [model_x, model_y] = model_training(data, past_data_considered, train_data)
%MODEL_TRAINING Summary of this function goes here
%   Detailed explanation goes here

disp("Model Training Started");
pause(1)

numFeatures = 1;
numHiddenUnits1 = 125;
numHiddenUnits2 = 100;
numClasses = 1;

layers = [sequenceInputLayer(numFeatures) 
    lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    dropoutLayer(0.2) 
    lstmLayer(numHiddenUnits2,'OutputMode','last') 
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses) 
    regressionLayer];

maxEpochs = 15;
miniBatchSize = 20;

options = trainingOptions('adam', 'MaxEpochs',maxEpochs, 'MiniBatchSize', ...
    miniBatchSize, 'GradientThreshold',1, 'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', 'LearnRateDropPeriod',maxEpochs/2, ...
    'LearnRateDropFactor',0.2, 'Verbose',0, 'Plots','training-progress',...
    'Shuffle','never');

% Training for the Mobile Sink X-Data
x_train_start = false;
for training_count = 1:train_data
    values = data(training_count).X;
    X = num2cell(values(1:end-1)');
    Y = values(2:end)';
    
    if x_train_start
        model_x = trainNetwork(model_x, X, Y, layers, options);
        fprintf('.'); 
    else
        model_x = trainNetwork(X, Y, layers, options);
        x_train_start = true;
        fprintf('Start\n.'); 
    end
    
end
fprintf('\nEnd'); 

% Saving Model X
save model_x;
fprintf('\nModel Training Ended');
disp("............................................................");


end

