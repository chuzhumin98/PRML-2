clear all
pfun = @(x) 0.2*normpdf(x,-1,1)+0.8*normpdf(x,1,1);
delta = 0.01;
x = -5:delta:5;
y = pfun(x);
%% ���������
n = 1; %�����ɵ����������
ita = 0.8; %������ٽ��
X0 = randn(n, 1); %��ʼ�������
flag = rand(n, 1); %�ж���������
Xrandom = X0 + (flag < ita) - (flag >= ita); %���ɵ�n�������
%% �۲�Parzen��������˹������Ч��
%% ������ֵ����
pnfun = @(x) 0; %����һ��Parzen�����ƺ���
a = 0.5; %ѡȡһ��a��ֵ
for i = 1:n 
    pnfun = @(x) pnfun(x) + (x > Xrandom(i)-a/2 && x < Xrandom(i)+a/2)*cos((x-Xrandom(i))*pi/a)*pi/(2*a*n);
end
for i = 1:length(x)
    yn(i) = pnfun(x(i));
end
%% �������ͼ�񲿷�
% plot(x, y)
% hold on 
% plot(x, yn)
% title(strcat('n=',num2str(n),',a=',num2str(a),'ʱ�볬�����Ч��ͼ'))
% xlabel('x')
% ylabel('p.d.f.')
% legend('p(x)��p.d.f.','p_n(x)��p.d.f.')
% box on
% saveas(gcf, strcat('n=',num2str(n),',a=',num2str(a),'ʱ�볬�����Ч��ͼ.png'))
%% ���\epsilon(p_n)�ľ�ֵ�����ڲ�ͬ������
m = 50; %ȡ�ز�������Ϊ50
n = 5; %p_n��n�Ĵ�С
a = 2; %����a��ȡֵ
X0 = [];
flag = [];
Xrandom = [];
yn = [];
for i=1:m
    X0 = randn(n, 1); %��ʼ�������
    flag = rand(n, 1); %�ж���������
    Xrandom = X0 + (flag < ita) - (flag >= ita); %���ɵ�n�������
    pnfun = @(x) 0; %����һ��Parzen�����ƺ���
    for i = 1:n 
        pnfun = @(x) pnfun(x) + (x > Xrandom(i)-a/2 && x < Xrandom(i)+a/2)*cos((x-Xrandom(i))*pi/a)*pi/(2*a*n);
    end
    for j = 1:length(x)
        yn(j) = pnfun(x(j));
    end
    deltaP = (yn - y) .* (yn - y);
    epsilon(i) = delta*sum(deltaP);
end
expect = mean(epsilon)
variance = var(epsilon)