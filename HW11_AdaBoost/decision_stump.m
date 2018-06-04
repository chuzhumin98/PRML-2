function [k, a, d] = decision_stump(X, y, w)
% decision_stump returns a rule ...
% h(x) = d if x(k) <= a, -d otherwise,
%
% Input
%     X : n * p matrix, each row a sample
%     y : n * 1 vector, each row a label
%     w : n * 1 vector, each row a weight
%
% Output
%     k : the optimal dimension
%     a : the optimal threshold
%     d : the optimal d, 1 or -1

% total time complexity required to be O(p*n*logn) or less
%%% Your Code Here %%%
n = length(X); %样本点个数
p = length(X(1,:)); %特征维数
minLoss = 2; %记录当前的最小分类损失
w1 = sum((y(:) == 1) .* w); %分类为1的权重
w_1 = 1 - w1; %分类为-1的权重
for i = 1:p
    [data, index] = sort(X(:,i));
    wlow1 = 0; %属于较低那一部分的分类为1的权重
    whigh1 = w1;
    wlow_1 = 0;
    whigh_1 = w_1;
    for j = 1:n-1
        thisIndex = index(j);
        if (y(thisIndex) == 1)
            whigh1 = whigh1 - w(thisIndex);
            wlow1 = wlow1 + w(thisIndex);
        else
            whigh_1 = whigh_1 - w(thisIndex);
            wlow_1 = wlow_1 + w(thisIndex);
        end
        [loss, thisD] = information_loss(wlow1, whigh1, wlow_1, whigh_1); %计算所带来的分类损失
        if (loss < minLoss) %找到一个分类损失更小的，则将之替换
            minLoss = loss; 
            k = i;
            a = (data(j)+data(j+1)) / 2;
            d = thisD;
        end
    end
end
%%% Your Code Here %%%
end