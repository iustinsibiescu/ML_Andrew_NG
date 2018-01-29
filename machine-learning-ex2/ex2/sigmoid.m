function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% Return sigmoid of an element or performs sigmoid of each element from a matrix
g = 1 ./ (exp(z .* -1) + 1);

end
