clear all
%% 读入数据并进行初始化操作
[data] = textread('data_uniform.txt');
xl = [-2, -2, -2];
xu = [2, 2, 2];
%% 接下来不断迭代EM过程，直到收敛
count = 0; %记录共循环的次数
while (true)
    %% E过程
    size = length(data); %总样本点个数
    for (i = 2:2:size) 
        data(i, 3) = xl(3)+(xu(3)-xl(3))*rand();
    end
    %% M过程
    xl_pred = xl; %存储上一轮的参数值，便于判停
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
%% 与全数据结果进行比较
