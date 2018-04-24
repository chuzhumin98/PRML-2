[result] = xlsread('MNN_NormP.xls');
size = length(result);
[AX,H1,H2] = plotyy(1:size, result(:,2),1:size, result(:,3),'plot');
set(AX(1),'XColor','k','YColor','k');
set(AX(2),'XColor','k','YColor','k');
HH1=get(AX(1),'Ylabel');
set(HH1,'String','分类所花时间(s)');
set(HH1,'color','k');
HH2=get(AX(2),'Ylabel');
set(HH2,'String','分类准确率');
set(HH2,'color','k');
set(H1,'LineStyle','-','LineWidth',2,'MarkerSize',20);
set(H1,'color','b');
set(H2,'LineStyle','-.');
set(H2,'color','r','LineWidth',2,'MarkerSize',20);
set(AX(1),'yTick',[60:30:180])  %设置左边Y轴的刻度 
set(AX(2),'yTick',[0.7:0.1:1.0]) %设置右边Y轴的刻度
set(AX(1),'ylim',[60 180])  %设置左边Y轴的刻度 
set(AX(2),'ylim',[0.7 1.0]) %设置右边Y轴的刻度
xlim([0.5 size+0.5])
set(gca, 'xTick', 1:size);  
set(gca,'XTickLabel',{'1','2','3','4','5','10','20','200'}) 
legend('分类所花时间','模型准确率','location','southwest');
xlabel('范数p的取值');
title('最近邻性能随着距离度量的变化关系');
grid on
box on
saveas(gcf,'MNNNormP.png')