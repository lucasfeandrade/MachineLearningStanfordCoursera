function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
hTheta = (X * theta);
J =1/(2*m)*sum((hTheta - y).^2)+lambda*sum(theta(2:end).^2)/(2*m);

grad(1) = sum(X(:,1)'*(hTheta - y))/m;

grad(2) = sum(X(:,2)'*(hTheta -y))/m + lambda*theta(2)/m;
#grad = sum((hTheta-y)'*X)/m + lambda*[0;theta(2:end)]/m;
% =========================================================================

grad = grad(:);

end