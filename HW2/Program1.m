pfun = @(x) 0.2*normpdf(x,-1,1)+0.8*normpdf(x,1,1);
delta = 0.01;
x = -5:delta:5;
y = pfun(x);
%% 生成随机数
n = 100; %所生成的随机数数量
ita = 0.8; %两类的临界点
X0 = randn(n, 1); %初始的随机数
flag = rand(n, 1); %判断属于哪类
Xrandom = X0 + (flag < ita) - (flag >= ita); %生成的n个随机数
%% 绘制p(x)下随机数选取效果图
% len = (max(Xrandom)-min(Xrandom))/10;
% hist(Xrandom, 10)
% h = findobj(gca,'Type','patch'); 
% set(h,'FaceColor',[0.2, 0.4, 0.1],'EdgeColor','w') 
% hold on 
% plot(x,y*n*len, 'r')
% title('p(x)下随机数选取效果图')
% xlabel('x')
% ylabel('放缩过的的p.d.f.')
% box on
% saveas(gcf, 'p(x)下随机数选取效果图.png')
%% 观察Parzen窗法（方窗）的效果
%% 计算数值部分
pnfun = @(x) 0; %定义一个Parzen窗估计函数
a = 2; %选取一个a的值
for i = 1:n 
    pnfun = @(x) pnfun(x) + (x <= Xrandom(i)+a/2 && x >= Xrandom(i)-a/2)/(a*n);
end
for i = 1:length(x)
    yn(i) = pnfun(x(i));
end
%% 具体绘制图像部分
% plot(x, y)
% hold on 
% plot(x, yn)
% title(strcat('n=',num2str(n),',a=',num2str(a),'时方窗拟合效果图'))
% xlabel('x')
% ylabel('p.d.f.')
% legend('p(x)的p.d.f.','p_n(x)的p.d.f.')
% box on
% saveas(gcf, strcat('n=',num2str(n),',a=',num2str(a),'时方窗拟合效果图.png'))