clear all
%% �������ݲ����г�ʼ������
[data] = textread('data_uniform.txt');
xl = [-2, -2, -2];
xu = [2, 2, 2];
%% ���������ϵ���EM���̣�ֱ������
count = 0; %��¼��ѭ���Ĵ���
while (true)
    %% E����
    size = length(data); %�����������
    for (i = 2:2:size) 
        data(i, 3) = xl(3)+(xu(3)-xl(3))*rand();
    end
    %% M����
    xl_pred = xl; %�洢��һ�ֵĲ���ֵ��������ͣ
    xu_pred = xu;
    xl = min(data);
    xu = max(data);
    count = count + 1;
    error1 = sum(abs(xl-xl_pred));
    error2 = sum(abs(xu-xu_pred));
    if (error1 + error2 < 0.001 || count >= 1000) 
        break;
    end
end
%% ��ȫ���ݽ�����бȽ�
