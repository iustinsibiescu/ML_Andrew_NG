function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));

% Regularized Linear Regression Cost
error = X * theta - y;
theta_no_bias = theta(2:end);
J = (error' * error + lambda * (theta_no_bias' * theta_no_bias)) / (2*m);

% Gradient for Regularized Linear Regression
grad = (X' * (X * theta - y) + lambda * [0; theta_no_bias]) / m;
grad = grad(:);

end
