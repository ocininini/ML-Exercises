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
Theta1_grad = zeros(size(Theta1)); %theta1 has size 25*401
Theta2_grad = zeros(size(Theta2)); %theta2 size 10*26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X=[ones(m,1),X]; %5000*401

%forward
a2=sigmoid(X*Theta1'); %5000x401*401x25=5000x25
a2=[ones(m,1),a2]; %5000x26

a3=sigmoid(a2*Theta2'); %5000x26*26x10=5000*10

yy=zeros(m,num_labels);
for i=1:m
    yy(i,y(i))=1;
end

J=sum(sum(-yy.*log(a3)-(1-yy).*log(1-a3)))/m;
reg=lambda*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))/(2*m);
J=J+reg;

for i=1:m %m is number of samples
    a1=X(i,:); %take one row each time %1x401
    a2=[1;sigmoid(Theta1*a1')]; %1+25x401*401x1=1+25x1=26x1
    a3=sigmoid(Theta2*a2); %10x1
    
    y_out=(1:num_labels)'==y(i); %10x1
    
    d3=a3-y_out; %10x1
    d2=(Theta2'*d3).*[1;sigmoidGradient(Theta1*a1')];
    %(26x10*10x1=26x1).*(1+25x401*401x1=26x1)
    d2=d2(2:end); %get rid of bias 25x1
   %step 4
    Theta1_grad=Theta1_grad+(d2*a1); %25x1*1x401=25x401
    Theta2_grad=Theta2_grad+(d3*a2'); % 10x1*1x26=10x26
end
%step 5   
Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;

Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+lambda*Theta1(:,2:end)/m;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+lambda*Theta2(:,2:end)/m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
