function [loss, d] = information_loss(wlow1, whigh1, wlow_1, whigh_1)
    % 目标：计算二分类造成的信息损失
    loss = wlow1 + whigh_1;
    d = -1;
    if (loss > wlow_1 + whigh1)
        d = 1;
        loss = wlow_1 + whigh1;
    end
end