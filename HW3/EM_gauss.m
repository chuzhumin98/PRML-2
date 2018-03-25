clear all
%% �������ݲ����г�ʼ������
[data] = textread('data_gauss.txt');
mu = [0, 0, 0];
sigma = eye(3);
%% ���������ϵ���EM���̣�ֱ������
while (true)
    %% E����
    size = length(data); %�����������
    sigma_ni = sigma^(-1); %�õ�sigma�������
    for (i = 2:2:size) 
        data(i, 3) = mu(3)-sigma(3,1)*(data(i,1)-mu(1))/sigma(3,3)-sigma(3,2)*(data(i,2)-mu(2))/sigma(3,3); %����ȱʧ������
    end
    %% M����
    mu_pred = mu; %�����洢��һ�ֵ�mu�����ж��Ƿ�ﵽ��ͣ����
    sigma_pred = sigma;
    mu = mean(data);
    sigma = zeros(3,3);
    for i = 1:size
        sigma = sigma + (data(i,:)-mu)' * (data(i,:)-mu)/size;
    end
    break;
end