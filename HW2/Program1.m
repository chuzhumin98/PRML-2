pfun = @(x) 0.2*normpdf(x,-1,1)+0.8*normpdf(x,1,1);
delta = 0.01;
x = -5:delta:5;
y = pfun(x);
%% ���������
n = 100; %�����ɵ����������
ita = 0.8; %������ٽ��
X0 = randn(n, 1); %��ʼ�������
flag = rand(n, 1); %�ж���������
Xrandom = X0 + (flag < ita) - (flag >= ita); %���ɵ�n�������
%% ����p(x)�������ѡȡЧ��ͼ
% len = (max(Xrandom)-min(Xrandom))/10;
% hist(Xrandom, 10)
% h = findobj(gca,'Type','patch'); 
% set(h,'FaceColor',[0.2, 0.4, 0.1],'EdgeColor','w') 
% hold on 
% plot(x,y*n*len, 'r')
% title('p(x)�������ѡȡЧ��ͼ')
% xlabel('x')
% ylabel('�������ĵ�p.d.f.')
% box on
% saveas(gcf, 'p(x)�������ѡȡЧ��ͼ.png')
%% �۲�Parzen��������������Ч��
%% ������ֵ����
pnfun = @(x) 0; %����һ��Parzen�����ƺ���
a = 2; %ѡȡһ��a��ֵ
for i = 1:n 
    pnfun = @(x) pnfun(x) + (x <= Xrandom(i)+a/2 && x >= Xrandom(i)-a/2)/(a*n);
end
for i = 1:length(x)
    yn(i) = pnfun(x(i));
end
%% �������ͼ�񲿷�
% plot(x, y)
% hold on 
% plot(x, yn)
% title(strcat('n=',num2str(n),',a=',num2str(a),'ʱ�������Ч��ͼ'))
% xlabel('x')
% ylabel('p.d.f.')
% legend('p(x)��p.d.f.','p_n(x)��p.d.f.')
% box on
% saveas(gcf, strcat('n=',num2str(n),',a=',num2str(a),'ʱ�������Ч��ͼ.png'))