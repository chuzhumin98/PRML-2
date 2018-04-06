clear all
[result1] = xlsread('MNN_trainSize.xls');
[result2] = xlsread('MNN_translate_trainSize.xls');
size = length(result1);
plot(1:size, result1(:,2), 'b', 'LineWidth', 2)
hold on
plot(1:size, result2(:,2), 'r', 'LineWidth', 2)
xlim([0.5 size+0.5])
ylim([0 520])
set(gca, 'yTick', 0:100:500);
set(gca, 'xTick', 1:size);  
set(gca,'XTickLabel',{'50','100','200','500','1000','2000','5000','10000','20000','60000'}) 
legend('平移前','平移后','location','northwest');
xlabel('训练集规模');
ylabel('分类所花时间')
title('平移前后时间性能随训练集规模变化的比较');
grid on
box on
saveas(gcf,'MNNtranslateTime.png')

figure
plot(1:size, result1(:,3), 'b', 'LineWidth', 2)
hold on
plot(1:size, result2(:,3), 'r', 'LineWidth', 2)
xlim([0.5 size+0.5])
ylim([0.6 1.0])
set(gca, 'yTick', 0.6:0.1:1.0);
set(gca, 'xTick', 1:size);  
set(gca,'XTickLabel',{'50','100','200','500','1000','2000','5000','10000','20000','60000'}) 
legend('平移前','平移后','location','northwest');
xlabel('训练集规模');
ylabel('分类准确率')
title('平移前后准确率性能随训练集规模变化的比较');
grid on
box on
saveas(gcf,'MNNtranslateAccuracy.png')
