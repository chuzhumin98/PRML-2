clear all
%% ����200���������(��)���������Ƿ�Ϊ���Կɷֵ�����
size = 200;
sizePerClass = 100;
points = rand(200,2)*1000; %�ҽ����������Χȡ��[0,1000]��[0,1000]������
A = rand(2,1)+[-0.5;-0.5]; %�ֱ�Ϊxǰ��ϵ����yǰ��ϵ���������������ԭʼ�ķ����ߵ�б��
score = A(1)*points(:,1)+A(2)*points(:,2);
scoreSort = sort(score);
splitscore = (scoreSort(sizePerClass)+score(sizePerClass+1))/2; %�ҵ������ķָ���
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