function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% This matrix will store the distance from each cluster point to each input point in X
distance_matrix = [];

% For each cluster point, vectorised implementation of calculating the distance to each and every input point in X
for i = 1 : K
    current_centroid = centroids(i, :);
    distance_vector = sum((X - current_centroid) .^ 2, 2);
    distance_matrix = [distance_matrix distance_vector];
end

% Finding the cluster that minimizes the distance to each input point in X
[M, I] = min(distance_matrix, [], 2);
idx = I;

end

