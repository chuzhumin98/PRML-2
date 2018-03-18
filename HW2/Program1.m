pfun = @(x) 0.2*normpdf(x,-1,1)+0.8*normpdf(x,1,1);
x = -5:0.0001:5;
y = pfun(x);
%% 生成随机数
n = 100; %所生成的随机数数量
ita = 0.8; %两类的临界点
X0 = randn(n, 1); %初始的随机数
flag = rand(n, 1); %判断属于哪类
Xrandom = X0 + (flag < ita) - (flag >= ita);
len = (max(Xrandom)-min(Xrandom))/10;
hist(Xrandom, 10)
h = findobj(gca,'Type','patch'); 
set(h,'FaceColor',[0.2, 0.4, 0.1],'EdgeColor','w') 
hold on 
plot(x,y*n*len, 'r')
title('p(x)下随机数选取效果图')
xlabel('x')
ylabel('放缩过的的p.d.f.')
box on
saveas(gcf, 'p(x)下随机数选取效果图.png')
