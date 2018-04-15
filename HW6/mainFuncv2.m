clear; clc;
load least_sq.mat;

%% Step 1: Data preprocessing
dataTrain = train_small; % select the training data

X = dataTrain.X;
y = dataTrain.y;

Lambda = 0.01: 0.01: 2.0; % a series of L1-norm penalty 
w_0 = pinv(X' * X) * (X' * y); % least-square estimation without L1-norm ...
                               % is supposed to be a good initial 
                               
%% Step 2: Train weight vectors with different penalty constants
W = least_sq_multi(X, y, Lambda, w_0); % each column a weight vector  

%% Step 3: plot different errors versus lambda
err_Lambda = errorEstimate(Lambda, W, X, y, test);

dataTrain = train_mid; % select the training data
X = dataTrain.X;
y = dataTrain.y;
w_0 = pinv(X' * X) * (X' * y);
W = least_sq_multi(X, y, Lambda, w_0); % each column a weight vector  
err_Lambda2 = errorEstimate(Lambda, W, X, y, test);

dataTrain = train_large; % select the training data
X = dataTrain.X;
y = dataTrain.y;
w_0 = pinv(X' * X) * (X' * y);
W = least_sq_multi(X, y, Lambda, w_0); % each column a weight vector  
err_Lambda3 = errorEstimate(Lambda, W, X, y, test);

figure;
plot(Lambda, err_Lambda(:, 1));
hold on
plot(Lambda, err_Lambda2(:, 1));
plot(Lambda, err_Lambda3(:, 1));
title('training error vs lambda');

figure;
plot(Lambda, err_Lambda(:,2));
hold on
plot(Lambda, err_Lambda2(:,2));
plot(Lambda, err_Lambda3(:,2));
title('L1 regularization penalty vs lambda');

figure;
plot(Lambda, err_Lambda(:,3));
hold on
plot(Lambda, err_Lambda2(:,3));
plot(Lambda, err_Lambda3(:,3));
title('objective vs lambda');

figure;
plot(Lambda, err_Lambda(:, 4)');
hold on
plot(Lambda, err_Lambda2(:, 4)');
plot(Lambda, err_Lambda3(:, 4)');
title('number features vs lambda');

figure;
plot(Lambda, err_Lambda(:, 5));
hold on
plot(Lambda, err_Lambda2(:, 5));
plot(Lambda, err_Lambda3(:, 5));
title('test error vs lambda');

