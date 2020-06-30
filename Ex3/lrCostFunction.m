function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h=sigmoid(X*theta); %hypothesis

%cost function
J=((y')*log(h)+((1-y)')*log(1-h))/(-m); %unregularized
temp=theta;
temp(1)=0; %exclude bias term theta0
J=J+lambda*sum(temp.^2)/(2*m); %regularized

%gradient
grad=X'*(h-y)/m;
grad=grad+lambda*temp/m; %regularized

% =============================================================

grad = grad(:);

end
