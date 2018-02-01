function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Iteration_1
a_1 = X;
m_1 = size(a_1, 1);
a_1 = [ones(m_1, 1) a_1];

% Iteration_2
z_2 = a_1 * Theta1';
a_2 = sigmoid(z_2);
m_2 = size(a_2, 1);
a_2 = [ones(m_2, 1) a_2];

% Iteration_3
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);

% Predict probability for an element to belong in each class
y_probs = a_3;

% Find the max probability on each row and return its index, because it
% also signifies its class
[element, index] = max(y_probs, [], 2);
p = index;


end
