function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
theta_no_bias = theta(2:end);

% Things that may be computed only once to increas performance
y_pred = sigmoid(X * theta);

% Cost is calculated same as before and regularization is afterwards added
J_without_reg = (log(y_pred)' * y + log(1 - y_pred)' * (1 - y)) / (-m);
J = J_without_reg + (theta_no_bias' * theta_no_bias) / (2 * m) * lambda;

% The gradient vetctor is calculated same as before and then regularization
% is added to all the terms with the exception of theta_0
grad = (X' * (y_pred - y)) ./ m;
grad(2:end) = grad(2:end) + theta_no_bias / m * lambda;

end