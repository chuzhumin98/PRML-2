clear all
%% 读入数据并进行初始化操作
[data] = textread('data_gauss.txt');
mu = [0, 0, 0];
sigma = eye(3);
%% 接下来不断迭代EM过程，直到收敛
while (true)
    %% E过程
    size = length(data); %总样本点个数
    sigma_ni = sigma^(-1); %得到sigma的逆矩阵
    for (i = 2:2:size) 
        data(i, 3) = mu(3)-sigma(3,1)*(data(i,1)-mu(1))/sigma(3,3)-sigma(3,2)*(data(i,2)-mu(2))/sigma(3,3); %更新缺失点数据
    end
    %% M过程
    mu_pred = mu; %用来存储上一轮的mu，以判断是否达到判停条件
    sigma_pred = sigma;
    mu = mean(data);
    sigma = zeros(3,3);
    for i = 1:size
        sigma = sigma + (data(i,:)-mu)' * (data(i,:)-mu)/size;
    end
    break;
end