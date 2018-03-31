function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% Need to make sure cost function gets computed only using rated movies
predictions = X * Theta';
error = (predictions .* R - Y);
regularization = lambda/2 * (sum(diag(Theta * Theta')) + sum(diag(X * X')));
J = sum(diag(error * error')) / 2 + regularization;

% Derivating on movie features x_k(i)
for i = 1 : num_movies
    idx = find(R(i, :) == 1);
    Theta_temp = Theta(idx, :);
    Y_temp = Y(i, idx);
    regularization = lambda * X(i, :);
    X_grad(i, :) = ((X(i, :) * Theta_temp' - Y_temp) * Theta_temp) + regularization;
end

% Derivating on user preferences theta_k(j)
for j = 1 : num_users
    idx = find(R(:, j) == 1);
    X_temp = X(idx, :);
    Y_temp = Y(idx, j);
    regularization = lambda * Theta(j, :);
    Theta_grad(j, :) =  (X_temp * Theta(j, :)' - Y_temp)' * X_temp + regularization;
end

grad = [X_grad(:); Theta_grad(:)];

end
