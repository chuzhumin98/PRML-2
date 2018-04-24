clear all
%% 生成200个随机样本(点)，并将它们分为线性可分的两类
size = 200;
sizePerClass = 100;
up = 10000; %随机取值的上界
bottom = -10000; %随机取值的下界 
gamma = 0.1; %设置阈值0.01
points = rand(size,2)*(up-bottom)+bottom*ones(size,2); %我将随机样本范围取在[0,up]×[0,up]区间中
A = rand(2,1)+[-0.5;-0.5]; %分别为x前的系数和y前的系数，即在这里给出原始的分类线的斜率
score = A(1)*points(:,1)+A(2)*points(:,2);
scoreSort = sort(score);
splitscore = (scoreSort(sizePerClass)+scoreSort(sizePerClass+1))/2; %找到两类间的分割线
label = -(score < splitscore) + (score >= splitscore);
index_1 = find(score < splitscore);
class_1 = points;
class_1(index_1,:) = NaN;
index_2 = find(score >= splitscore);
class_2 = points;
class_2(index_2,:) = NaN;
plot(class_1(:,1),class_1(:,2),'b.','MarkerSize',6)
hold on
plot(class_2(:,1),class_2(:,2),'r.','MarkerSize',6)
%% 采用SSC进行分类
iter = 0; %记录迭代的次数
alpha = rand(3,1); %初始化参数,最后一项时常数项
k = 0;
countCorrect = 0; %计数连续判断无误的点个数
while (true)
    k = mod(k,size)+1;
    iter = iter + 1;
    mode = sqrt(points(k,1)*points(k,1)+points(k,2)*points(k,2)+1); %样本归一化因子
    if (label(k)*(alpha(1)*points(k,1)+alpha(2)*points(k,2)+alpha(3))/mode <= gamma) 
        alpha(1) = alpha(1) + label(k)*points(k,1)/mode; %更新各自的权值
        alpha(2) = alpha(2) + label(k)*points(k,2)/mode;
        alpha(3) = alpha(3) + label(k)/mode;
        countCorrect = 0;
    else
        countCorrect = countCorrect + 1; 
    end
    if (countCorrect >= size) 
        break;
    end
end
%% 将分界线绘制到图像中，alpha(1)*x+alpha(2)*y+alpha(3)=0
X = bottom:1:up;
Y = -(alpha(1)*X+alpha(3))/alpha(2);
plot(X,Y, 'g')
xlim([bottom up])
ylim([bottom up])
legend('-1类样本点','+1类样本点','分界面直线','Location','NorthEastOutside')
xlabel('x坐标')
ylabel('y坐标')
title(strcat('\delta=',num2str(gamma),'下margin感知器算法解决线性可分问题'))
set(gcf,'position',[0,0,1000,1000])
saveas(gcf,strcat( 'delta=',num2str(gamma),'SSCwithMargin.png'))