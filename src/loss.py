import torch

def _exp_loss(self, pred, target, time, toa, fps=30.0):
    '''
    :param pred：（batch，2）
    :param target: onehot codings for binary classification， （0，1）事故，（1，0）非事故
    :param time:每个视频共150帧，循环处理每一帧，time从0开始，0-149
    :param toa:异常结束帧
    :param fps:帧率
    :return:
    '''

    target_cls = target[:, 1]
    # print(target[:, 1])
    # print("label:",target_cls, ":", pred)
    target_cls = target_cls.to(torch.long)
    penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype),
                         (toa.to(pred.dtype) - time - 1) / fps)
    # print("penalty:",  penalty)
    pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
    neg_loss = self.ce_loss(pred, target_cls)
    loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
    # print('l1:', torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), 0)))
    # print('l2:', torch.mean(torch.add(0, torch.mul(neg_loss, target[:, 0]))))
    return loss


# kl散度，注意力重构
def kl_loss(self, y_true, y_pred, eps=1e-07):
    P = y_pred
    P = P / (eps + torch.sum(P, dim=(0, 1, 2, 3), keepdim=True))
    Q = y_true
    Q = Q / (eps + torch.sum(Q, dim=(0, 1, 2, 3), keepdim=True))
    kld = torch.sum(Q * torch.log(eps + Q / (eps + P)), dim=(0, 1, 2, 3))
    # kld=torch.exp(-kld)
    return kld