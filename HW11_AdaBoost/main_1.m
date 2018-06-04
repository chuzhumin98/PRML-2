clear all
load('ada_data.mat');
trainSize = length(X_train);
w = ones(trainSize,1)/trainSize;
maxIter = 300; %the max iteration number
[e_train, e_test] = adaboost(X_train, y_train, X_test, y_test, 300);
plot(1:maxIter, e_train, 'LineWidth', 1.5)
hold on 
plot(1:maxIter, e_test, 'LineWidth', 1.5)
xlabel('iteration')
ylabel('error rate')
title('error rate vs iteration')
legend('train set', 'test set')
saveas(gcf, 'error1.png')