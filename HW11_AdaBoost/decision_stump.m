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
n = length(X); %���������
p = length(X(1,:)); %����ά��
minLoss = 2; %��¼��ǰ����С������ʧ
w1 = sum((y(:) == 1) .* w); %����Ϊ1��Ȩ��
w_1 = 1 - w1; %����Ϊ-1��Ȩ��
for i = 1:p
    [data, index] = sort(X(:,i));
    wlow1 = 0; %���ڽϵ���һ���ֵķ���Ϊ1��Ȩ��
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
        [loss, thisD] = information_loss(wlow1, whigh1, wlow_1, whigh_1); %�����������ķ�����ʧ
        if (loss < minLoss) %�ҵ�һ��������ʧ��С�ģ���֮�滻
            minLoss = loss; 
            k = i;
            a = (data(j)+data(j+1)) / 2;
            d = thisD;
        end
    end
end
%%% Your Code Here %%%
end