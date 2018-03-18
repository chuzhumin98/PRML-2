pfun = @(x) 0.2*normpdf(x,-1,1)+0.8*normpdf(x,1,1);
x = -5:0.0001:5;
y = pfun(x);
%% ���������
n = 100; %�����ɵ����������
ita = 0.8; %������ٽ��
X0 = randn(n, 1); %��ʼ�������
flag = rand(n, 1); %�ж���������
Xrandom = X0 + (flag < ita) - (flag >= ita);
len = (max(Xrandom)-min(Xrandom))/10;
hist(Xrandom, 10)
h = findobj(gca,'Type','patch'); 
set(h,'FaceColor',[0.2, 0.4, 0.1],'EdgeColor','w') 
hold on 
plot(x,y*n*len, 'r')
title('p(x)�������ѡȡЧ��ͼ')
xlabel('x')
ylabel('�������ĵ�p.d.f.')
box on
saveas(gcf, 'p(x)�������ѡȡЧ��ͼ.png')
