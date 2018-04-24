clear all
%% ����200���������(��)���������Ƿ�Ϊ���Կɷֵ�����
size = 200;
sizePerClass = 100;
up = 10000; %���ȡֵ���Ͻ�
bottom = -10000; %���ȡֵ���½� 
gamma = 0.1; %������ֵ0.01
points = rand(size,2)*(up-bottom)+bottom*ones(size,2); %�ҽ����������Χȡ��[0,up]��[0,up]������
A = rand(2,1)+[-0.5;-0.5]; %�ֱ�Ϊxǰ��ϵ����yǰ��ϵ���������������ԭʼ�ķ����ߵ�б��
score = A(1)*points(:,1)+A(2)*points(:,2);
scoreSort = sort(score);
splitscore = (scoreSort(sizePerClass)+scoreSort(sizePerClass+1))/2; %�ҵ������ķָ���
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
%% ����SSC���з���
iter = 0; %��¼�����Ĵ���
alpha = rand(3,1); %��ʼ������,���һ��ʱ������
k = 0;
countCorrect = 0; %���������ж�����ĵ����
while (true)
    k = mod(k,size)+1;
    iter = iter + 1;
    mode = sqrt(points(k,1)*points(k,1)+points(k,2)*points(k,2)+1); %������һ������
    if (label(k)*(alpha(1)*points(k,1)+alpha(2)*points(k,2)+alpha(3))/mode <= gamma) 
        alpha(1) = alpha(1) + label(k)*points(k,1)/mode; %���¸��Ե�Ȩֵ
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
%% ���ֽ��߻��Ƶ�ͼ���У�alpha(1)*x+alpha(2)*y+alpha(3)=0
X = bottom:1:up;
Y = -(alpha(1)*X+alpha(3))/alpha(2);
plot(X,Y, 'g')
xlim([bottom up])
ylim([bottom up])
legend('-1��������','+1��������','�ֽ���ֱ��','Location','NorthEastOutside')
xlabel('x����')
ylabel('y����')
title(strcat('\delta=',num2str(gamma),'��margin��֪���㷨������Կɷ�����'))
set(gcf,'position',[0,0,1000,1000])
saveas(gcf,strcat( 'delta=',num2str(gamma),'SSCwithMargin.png'))