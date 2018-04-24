clear all
%% 读入数据并进行初始化操作
[data] = textread('data_gauss.txt');
mu = [0, 0, 0];
sigma = eye(3);
Mu3 = [];
%% 接下来不断迭代EM过程，直到收敛
count = 0; %记录共循环的次数
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
    count = count + 1;
    Mu3 = [Mu3, mu(3)];
    error1 = sum(abs(mu - mu_pred));
    error2 = sum(sum(abs(sigma - sigma_pred)));
    % 发现结果存在震荡，于是加了一个均值动量项
    if (count == 1000) 
        mu = (mu+mu_pred)/2;
        sigma = (sigma+sigma_pred)/2;
    end
    if (error1 + error2 < 0.01 || count >= 1004) 
        break;
    end
end
%% 相关的数据可视化工作
% plot(990:1030, Mu3(990:1030))
% xlabel('迭代次数')
% ylabel('\mu_3')
% title('\mu_3取值随迭代次数增加的变化关系')
% saveas(gcf, 'mu_3取值随迭代次数增加的变化关系3.png')
% figure
% plot(1960:2000, Mu3(1960:2000))
% xlabel('迭代次数')
% ylabel('\mu_3')
% title('\mu_3取值随迭代次数增加的变化关系')
% saveas(gcf, 'mu_3取值随迭代次数增加的变化关系4.png')
%% 与全数据结果进行比较
[data0] = textread('data_gauss.txt');
mu0 = mean(data0);
sigma0 = zeros(3,3);
for i = 1:size
    sigma0 = sigma0 + (data0(i,:)-mu)' * (data0(i,:)-mu)/size;
end