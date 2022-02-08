function [model_x, model_y] = model_training(data, past_data_considered, train_data)
%MODEL_TRAINING Summary of this function goes here
%   Detailed explanation goes here

% Training for the Mobile Sink X-Data
for training_count = 1:train_data
    X = data(training_count).X;
    writematrix(X,'X.csv','Delimiter','comma');
    Y = data(training_count).Y;
    writematrix(Y,'Y.csv','Delimiter','comma');
    
    
    
end



end

