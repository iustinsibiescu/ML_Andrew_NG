function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% Things that may be computed only once to increas performance
y_pred = sigmoid(X * theta);

% Vectorized form of obtaining J and grad
J = (log(y_pred)' * y + log(1 - y_pred)' * (1 - y)) / (-m);
grad = (X' * (y_pred - y)) ./ m;

end