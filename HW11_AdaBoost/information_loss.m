function [loss, d] = information_loss(wlow1, whigh1, wlow_1, whigh_1)
    % Ŀ�꣺�����������ɵ���Ϣ��ʧ
    loss = wlow1 + whigh_1;
    d = -1;
    if (loss > wlow_1 + whigh1)
        d = 1;
        loss = wlow_1 + whigh1;
    end
end