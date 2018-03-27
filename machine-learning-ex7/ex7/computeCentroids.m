function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

% Computing the mean of each centroid
for i = 1:K
    points_from_centroid = (idx == i);
    
    % Nr of points in that centroid
    C = sum(points_from_centroid);
    
    % Calculating the mean of the current centroid
    centroids(i, :) = sum(X(points_from_centroid, :), 1) / C;
end


end

