function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

%number of examples
m = size(X,1);

% the associated mean and std vectors for the matrix X
mu = mean(X, 1);
sigma = std(X, 1);

% normalizing each value of X by substracting and dividing by its
% accociated mean and std values
X_norm = (X - repmat(mu, m, 1)) ./ repmat(sigma, m, 1);

end
