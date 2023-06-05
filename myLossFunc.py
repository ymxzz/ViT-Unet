import torch
import torch.nn as nn
import torch.nn.functional as F


# Dice损失函数
class DiceLoss(nn.Module): # pre:(1, 2, 64, 128, 128) tar:(1, 1, 64, 128, 128)

    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets):
        N = preds.size(0) # 1
        C = preds.size(1) # 2
        

        P = F.softmax(preds, dim=1) # torch.Size([1, 2, 64, 128, 128])
        smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001) # torch.Size([2])

        class_mask = torch.zeros(preds.shape).to(preds.device) # torch.Size([1, 2, 64, 128, 128])
        class_mask.scatter_(1, targets, 1.)  # ?
        shape = preds.shape
        ones = torch.ones(preds.shape).to(preds.device) # torch.Size([1, 2, 64, 128, 128])
        
        P_ = ones - P 
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(preds.device)
        self.alpha = FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) / ((FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) + FN.transpose(0, 1).reshape(C, -1).sum(dim=(1))) + smooth)
    
        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8) 
        #print('alpha:', self.alpha)
        self.beta = 1 - self.alpha
        num = torch.sum(TP.transpose(0, 1).reshape(C, -1), dim=(1)).float()
        den = num + self.alpha * torch.sum(FP.transpose(0, 1).reshape(C, -1), dim=(1)).float() + self.beta * torch.sum(FN.transpose(0, 1).reshape(C, -1), dim=(1)).float()

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss

        loss = 1 - dice
        loss = loss.sum()

        if self.size_average:
            loss /= C

        return loss
    
if __name__ == "__main__":
    loss_func = DiceLoss()
    pre = torch.randn(1, 2, 64, 128, 128).to("cuda")
    tar = torch.zeros(1, 1, 64, 128, 128).long().to("cuda")
    
    loss = loss_func(pre, tar)
    print(loss)