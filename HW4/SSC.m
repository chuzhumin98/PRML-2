clear all
%% 生成200个随机样本(点)，并将它们分为线性可分的两类
size = 200;
sizePerClass = 100;
points = rand(200,2)*1000; %我将随机样本范围取在[0,1000]×[0,1000]区间中
A = rand(2,1)+[-0.5;-0.5]; %分别为x前的系数和y前的系数，即在这里给出原始的分类线的斜率
score = A(1)*points(:,1)+A(2)*points(:,2);
scoreSort = sort(score);
splitscore = (scoreSort(sizePerClass)+score(sizePerClass+1))/2; %找到两类间的分割线
label = -(score < splitscore) + (score >= splitscore);
index_1 = find(score < splitscore);
class_1 = points;
class_1(index_1,:) = NaN;
index_2 = find(score >= splitscore);
class_2 = points;
class_2(index_2,:) = NaN;
plot(class_1(:,1),class_1(:,2),'b.','MarkerSize',8)
hold on
plot(class_2(:,1),class_2(:,2),'r.','MarkerSize',8)