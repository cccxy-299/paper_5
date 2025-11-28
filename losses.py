import numpy as np
import torch
from torch import nn
import pytorch_msssim
import torch.nn.functional as F

criterion = nn.BCELoss()
def td_loss(feature_1,feature_2,label,threshold=0.85,margin=0.05):
    feature_1 = feature_1.squeeze(1)  # (B, 512)
    feature_2 = feature_2.squeeze(1)  # (B, 512)


    cos_sim = F.cosine_similarity(feature_1, feature_2, dim=1)  # (B,)
    loss = torch.where(label == 1, (cos_sim + 1 + margin)*8, 1 - cos_sim)

    # pred = torch.where(cos_sim > 0.85, 0.0, 1.0)
    # loss_bce = criterion(pred, label)
    # anomaly_score = 1 - cos_sim  # 越大越异常

    # 设置自适应权重+聚类损失
    # pred = torch.where(anomaly_score > threshold, 1, 0)
    # pred_wrong = torch.where(pred==label,0,1)

    return loss.mean()


class SSIM_L1_Loss(nn.Module):
    def __init__(self, alpha=0.84):  # alpha 越大越偏重结构相似性
        super().__init__()
        self.alpha = alpha
        self.ssim = pytorch_msssim.SSIM(data_range=1.0, size_average=True)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        ssim_loss = 1 - self.ssim(pred, target)
        l1_loss = self.l1(pred, target)
        return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss

if __name__ == "__main__":
    a = torch.rand(3,512)
    b = torch.rand(3,512)
    label = torch.from_numpy(np.array([0, 0, 1]))

    print(td_loss(a,b,label))
    # print(c)
