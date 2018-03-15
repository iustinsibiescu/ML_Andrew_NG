function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Useful variables
paramRange = [0.01 0.03 0.1 0.3 1 3 10 30];
lengthParamRange = length(paramRange);
minimumPredictionError = intmax('int32');

% Choosing the best parameters
for i = 1 : lengthParamRange
    for j = 1 : lengthParamRange
        C_train = paramRange(i);
        sigma_train = paramRange(j);
        
        model= svmTrain(X, y, C_train, @(x1, x2) gaussianKernel(x1, x2, sigma_train));
        predictions = svmPredict(model, Xval);
        predictionError = mean(double(predictions ~= yval));
        
        if minimumPredictionError > predictionError
            C = C_train;
            sigma = sigma_train;
            minimumPredictionError = predictionError;
        end
    end
end

end
