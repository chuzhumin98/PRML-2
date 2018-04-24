% get for err_Lambda matrix
function err_Lambda = errorEstimate(Lambda, W, X, y, test)
L = length(Lambda);
err_Lambda = zeros(L, 5); % each row a different lambda
for l = 1: L
  w = W(:, l);
  %%% Your code here %%%
  % training error multiplying 1/2
  delta = y - X * w; %(y - w^T x),n * 1 matrix
  err_Lambda(l, 1) = (delta' * delta)/2/length(y);

  % L1 regularization penalty
  err_Lambda(l, 2) = sum(abs(w));

  % minimized objective
  err_Lambda(l, 3) = err_Lambda(l,1) + Lambda(l) * err_Lambda(l,2);

  % L0 norm: non-zero parameters  
  err_Lambda(l, 4) = sum((abs(w) >= 1e-16));
  
  % test error
  deltaTest = test.y - test.X * w; %(y - w^T x),n * 1 matrix
  err_Lambda(l, 5) = (deltaTest' * deltaTest)/length(test.y);
  %%% Your code here %%%
end

end