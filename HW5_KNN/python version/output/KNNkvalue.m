[result] = xlsread('KNN_KValue.xls');
size = length(result);
[AX,H1,H2] = plotyy(1:size, result(:,2),1:size, result(:,3),'plot');
set(AX(1),'XColor','k','YColor','k');
set(AX(2),'XColor','k','YColor','k');
HH1=get(AX(1),'Ylabel');
set(HH1,'String','��������ʱ��(s)');
set(HH1,'color','k');
HH2=get(AX(2),'Ylabel');
set(HH2,'String','����׼ȷ��');
set(HH2,'color','k');
set(H1,'LineStyle','-','LineWidth',2,'MarkerSize',20);
set(H1,'color','b');
set(H2,'LineStyle','-.');
set(H2,'color','r','LineWidth',2,'MarkerSize',20);
set(AX(1),'yTick',[75:5:90])  %�������Y��Ŀ̶� 
set(AX(2),'yTick',[0.935:0.005:0.955]) %�����ұ�Y��Ŀ̶�
set(AX(1),'ylim',[75 90])  %�������Y��Ŀ̶� 
set(AX(2),'ylim',[0.935 0.955]) %�����ұ�Y��Ŀ̶�
xlim([0.5 size+0.5])
set(gca, 'xTick', 1:size);  
set(gca,'XTickLabel',{'1','3','5','7','9','11','13','15','17','19','21'}) 
legend('��������ʱ��','ģ��׼ȷ��','location','northwest');
xlabel('k��ȡֵ');
title('k-������������k��ȡֵ�ı仯��ϵ');
grid on
box on
saveas(gcf,'KNNKValue.png')