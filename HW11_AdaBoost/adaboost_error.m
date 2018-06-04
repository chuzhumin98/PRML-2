function e = adaboost_error(X, y, k, a, d, alpha)
% adaboost_error: returns the final error rate of a whole adaboost
% 
% Input
%     X     : n * p matrix, each row a sample
%     y     : n * 1 vector, each row a label
%     k     : iter * 1 vector,  selected dimension of features
%     a     : iter * 1 vector, selected threshold for feature-k
%     d     : iter * 1 vector, 1 or -1
%     alpha : iter * 1 vector, weights of the classifiers
%
% Output
%     e     : error rate      

%%% Your Code Here %%%
n = length(y); %sample size
iter = length(k); %iter size
predict = zeros(n, 1); %Ô¤²â½á¹û
for i = 1:iter
    if (k(i) == 0) %if this model has not used, throw all the following
        break
    end
    thisPredict = ((X(:, k(i)) <= a(i)) - 0.5) * 2 * d(i); % this model's predicted label
    predict = predict + alpha(i) * thisPredict; % add this model's info
end
predict = sign(predict); %get each sample's predict result
predict = sign(predict + 0.001); %handle zero prob problem
e = mean(predict ~= y); %calculate the error rate
%%% Your Code Here %%%

end