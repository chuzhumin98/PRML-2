[result] = xlsread('MNN_trainSize.xls');
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
set(AX(1),'yTick',[0:100:500])  %�������Y��Ŀ̶� 
set(AX(2),'yTick',[0.6:0.1:1.0]) %�����ұ�Y��Ŀ̶�
set(AX(1),'ylim',[0 520])  %�������Y��Ŀ̶� 
set(AX(2),'ylim',[0.6 1.0]) %�����ұ�Y��Ŀ̶�
xlim([0.5 size+0.5])
set(gca, 'xTick', 1:size);  
set(gca,'XTickLabel',{'50','100','200','500','1000','2000','5000','10000','20000','60000'}) 
legend('��������ʱ��','ģ��׼ȷ��','location','northwest');
xlabel('ѵ������ģ');
title('�������������ѵ������ģ�ı仯��ϵ');
grid on
box on
saveas(gcf,'MNNtrainsize.png')