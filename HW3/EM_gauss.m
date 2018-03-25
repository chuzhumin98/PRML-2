clear all
%% �������ݲ����г�ʼ������
[data] = textread('data_gauss.txt');
mu = [0, 0, 0];
sigma = eye(3);
Mu3 = [];
%% ���������ϵ���EM���̣�ֱ������
count = 0; %��¼��ѭ���Ĵ���
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
    count = count + 1;
    Mu3 = [Mu3, mu(3)];
    error1 = sum(abs(mu - mu_pred));
    error2 = sum(sum(abs(sigma - sigma_pred)));
    % ���ֽ�������𵴣����Ǽ���һ����ֵ������
    if (count == 1000) 
        mu = (mu+mu_pred)/2;
        sigma = (sigma+sigma_pred)/2;
    end
    if (error1 + error2 < 0.01 || count >= 1004) 
        break;
    end
end
%% ��ص����ݿ��ӻ�����
% plot(990:1030, Mu3(990:1030))
% xlabel('��������')
% ylabel('\mu_3')
% title('\mu_3ȡֵ������������ӵı仯��ϵ')
% saveas(gcf, 'mu_3ȡֵ������������ӵı仯��ϵ3.png')
% figure
% plot(1960:2000, Mu3(1960:2000))
% xlabel('��������')
% ylabel('\mu_3')
% title('\mu_3ȡֵ������������ӵı仯��ϵ')
% saveas(gcf, 'mu_3ȡֵ������������ӵı仯��ϵ4.png')
%% ��ȫ���ݽ�����бȽ�
[data0] = textread('data_gauss.txt');
mu0 = mean(data0);
sigma0 = zeros(3,3);
for i = 1:size
    sigma0 = sigma0 + (data0(i,:)-mu)' * (data0(i,:)-mu)/size;
end