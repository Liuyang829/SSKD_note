import torch
import torch.nn as nn
import torch.nn.functional as F

class wrapper(nn.Module):

    def __init__(self, module):

        super(wrapper, self).__init__()

        self.backbone = module
        feat_dim = list(module.children())[-1].in_features
        # 两层的多层感知机,做一个投影头,计算论文中的z，记作MLP
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
            )

    def forward(self, x, bb_grad=True):
        # is_feat大概是输出每一阶段最后的featuremap
        feats, out = self.backbone(x, is_feat=True)
        # print("feats.shape in list:")
        # for i in feats:
        #     print(i.shape)
        # torch.Size([64, 16, 32, 32])
        # torch.Size([64, 32, 32, 32])
        # torch.Size([64, 64, 16, 16])
        # torch.Size([64, 128, 8, 8])
        # torch.Size([64, 128])

        # 最后一个出来的feature
        feat = feats[-1].view(feats[-1].size(0), -1)
        # 如果bb_grad==false 执行 这个特征不需要梯度回传
        if not bb_grad:
            feat = feat.detach()
        return out, self.proj_head(feat), feat
        
