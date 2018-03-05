function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% ========================= Part1: Cost function ==========================

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Convert y into m by num_labels matrix, with probability 1 for correct class and 0 otherwise 
y_binary = zeros(m, num_labels);
index = sub2ind(size(y_binary), 1:m, y');
y_binary(index) = 1;

% Feed-forward algorithm: vectorised implementation
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

% Calculating cost: The main diagonal of the following matrix contains the errors
J = sum(diag(y_binary * log(h2') + (1 - y_binary) * log(1 - h2'))) / (-m);

% Adding regularization
Theta1_no_bias = Theta1(:, 2:end);
Theta2_no_bias = Theta2(:, 2:end);
regularization = lambda / (2*m) * (sum(diag(Theta1_no_bias * Theta1_no_bias')) + sum(diag(Theta2_no_bias * Theta2_no_bias')));
J = J + regularization;

% ============================= Part2: Gradient ===========================

% For-loop Backpropagation algorithm
for nr = 1 : m
    training_example = X(nr, :);

    % Feed-forward
    a1 = sigmoid(Theta1 * [1; training_example']);
    a2 = sigmoid(Theta2 * [1; a1]);
end

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
