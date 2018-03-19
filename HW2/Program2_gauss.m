clear all
pfun = @(x) 0.2*normpdf(x,-1,1)+0.8*normpdf(x,1,1);
delta = 0.01;
x = -5:delta:5;
y = pfun(x);
%% 生成随机数
n = 5; %所生成的随机数数量
ita = 0.8; %两类的临界点
X0 = randn(n, 1); %初始的随机数
flag = rand(n, 1); %判断属于哪类
Xrandom = X0 + (flag < ita) - (flag >= ita); %生成的n个随机数
%% 观察Parzen窗法（高斯窗）的效果
%% 计算数值部分
pnfun = @(x) 0; %定义一个Parzen窗估计函数
sigma = 0.25; %选取一个a的值
for i = 1:n 
    pnfun = @(x) pnfun(x) + normpdf(x, Xrandom(i),sigma)/n;
end
for i = 1:length(x)
    yn(i) = pnfun(x(i));
end
%% 具体绘制图像部分
% plot(x, y)
% hold on 
% plot(x, yn)
% title(strcat('n=',num2str(n),',\sigma=',num2str(sigma),'时高斯窗拟合效果图'))
% xlabel('x')
% ylabel('p.d.f.')
% legend('p(x)的p.d.f.','p_n(x)的p.d.f.')
% box on
% saveas(gcf, strcat('n=',num2str(n),',sigma=',num2str(sigma),'时高斯窗拟合效果图.png'))
%% 求解\epsilon(p_n)的均值方差在不同参数下
m = 50; %取重采样次数为50
n = 500; %p_n中n的大小
sigma = 1; %参数a的取值
X0 = [];
flag = [];
Xrandom = [];
yn = [];
for i=1:m
    X0 = randn(n, 1); %初始的随机数
    flag = rand(n, 1); %判断属于哪类
    Xrandom = X0 + (flag < ita) - (flag >= ita); %生成的n个随机数
    pnfun = @(x) 0; %定义一个Parzen窗估计函数
    for i = 1:n 
        pnfun = @(x) pnfun(x) + normpdf(x, Xrandom(i),sigma)/n;
    end
    for j = 1:length(x)
        yn(j) = pnfun(x(j));
    end
    deltaP = (yn - y) .* (yn - y);
    epsilon(i) = delta*sum(deltaP);
end
expect = mean(epsilon)
variance = var(epsilon)