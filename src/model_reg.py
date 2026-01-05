# model_reg.py
import math, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# =============== MixUp / CutMix (for images) ===============
def mixup_data(x, y, alpha=0.15):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha=0.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    b, _, H, W = x.size()
    idx = torch.randperm(b, device=x.device)
    cut_rat = math.sqrt(1. - lam)
    cw, ch = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = np.clip(cx - cw // 2, 0, W), np.clip(cy - ch // 2, 0, H)
    x2, y2 = np.clip(cx + cw // 2, 0, W), np.clip(cy + ch // 2, 0, H)
    x2c = x.clone()
    x2c[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x2c, y, y[idx], lam

# For regression: turn (ya, yb, lam) into a single mixed target
def mix_targets_regression(ya, yb, lam):
    if ya.dim() == 1: ya = ya.unsqueeze(1)
    if yb.dim() == 1: yb = yb.unsqueeze(1)
    return lam * ya + (1 - lam) * yb

# =============== Attention blocks ===============
class ECA(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=(k - 1) // 2, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sig(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)

class cSE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.attn(x)

class sSE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, 1, 1); self.sig = nn.Sigmoid()
    def forward(self, x):
        return x * self.sig(self.conv(x))

class P_scSE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.cse, self.sse = cSE(c, r), sSE(c)
    def forward(self, x):
        a, b = self.cse(x), self.sse(x)
        return torch.max(a, b) + a

# =============== GeM Pooling ===============
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x

# =============== Regression Model (EffB3 ⨉ MobileNetV3-Small) ===============
class EffB3_MNV3S_Parallel_Regression(nn.Module):
    """
    Dual-backbone (EffB3 + MNV3-Small) with attention fusion; 1D regression head.
    """
    def __init__(self, in_ch=3, drop=0.25, fuse_ch=512):
        super().__init__()
        self.in_ch = in_ch

        eff = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.eff_feat = eff.features            # out 1536

        m3s = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.mnv3_feat = m3s.features          # out 576

        # first convs adapted to in_ch
        self.eff_feat[0][0] = nn.Conv2d(in_ch, 40, kernel_size=3, stride=2, padding=1, bias=False)
        self.mnv3_feat[0][0] = nn.Conv2d(in_ch, 16, kernel_size=3, stride=2, padding=1, bias=False)

        # reduce channels to 256 each
        self.red_a = nn.Sequential(
            nn.Conv2d(1536, 1536, 3, padding=1, groups=1536, bias=False),
            nn.BatchNorm2d(1536), nn.ReLU(inplace=True),
            nn.Conv2d(1536, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.red_b = nn.Sequential(
            nn.Conv2d(576, 576, 3, padding=1, groups=576, bias=False),
            nn.BatchNorm2d(576), nn.ReLU(inplace=True),
            nn.Conv2d(576, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(512, fuse_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_ch), nn.ReLU(inplace=True),
            P_scSE(fuse_ch, r=8), ECA(5), nn.Dropout2d(drop)
        )
        self.pool = GeM()  # <-- swapped in

        # Head: LayerNorm + smaller hidden (256) for a touch more regularization
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(fuse_ch),
            nn.Linear(fuse_ch, 256), nn.ReLU(inplace=True), nn.Dropout(drop),
            nn.Linear(256, 1)
        )

    # convenience for warmup freeze
    def freeze_backbones(self, flag: bool = True):
        for p in self.eff_feat.parameters(): p.requires_grad = (not flag)
        for p in self.mnv3_feat.parameters(): p.requires_grad = (not flag)

    def forward(self, x):
        a = self.eff_feat(x)
        b = self.mnv3_feat(x)
        if a.shape[-2:] != b.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)
        a, b = self.red_a(a), self.red_b(b)
        y = self.fuse(torch.cat([a, b], 1))
        y = self.pool(y)                # (B,C,1,1)
        out = self.head(y)              # (B,1)
        return out

# =============== Losses for regression ===============
class PearsonCorrLoss(nn.Module):
    """ 1 - Pearson correlation (maximize correlation) """
    def __init__(self, eps=1e-8):
        super().__init__(); self.eps = eps
    def forward(self, preds, target):
        if preds.dim()==2 and preds.size(1)==1: preds = preds.squeeze(1)
        if target.dim()==2 and target.size(1)==1: target = target.squeeze(1)
        vx = preds - preds.mean()
        vy = target - target.mean()
        corr = (vx*vy).sum() / (torch.sqrt((vx*vx).sum()+self.eps) * torch.sqrt((vy*vy).sum()+self.eps))
        return 1.0 - corr

class MSEPlusCorrLoss(nn.Module):
    """ Mix MSE with correlation to align scale + trend """
    def __init__(self, alpha=0.7):
        super().__init__(); self.alpha=alpha
        self.mse = nn.MSELoss()
        self.corr = PearsonCorrLoss()
    def forward(self, preds, target):
        return self.alpha*self.mse(preds, target) + (1-self.alpha)*self.corr(preds, target)

class HuberPlusCorr(nn.Module):
    """ SmoothL1 (Huber) + (1-alpha)*Corr """
    def __init__(self, beta=0.25, alpha=0.6):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=beta)
        self.alpha = alpha
        self.corr  = PearsonCorrLoss()
    def forward(self, preds, target):
        return self.alpha*self.huber(preds, target) + (1-self.alpha)*self.corr(preds, target)

def get_regression_loss(name="huber", **kwargs):
    name = name.lower()
    if name == "mse":          return nn.MSELoss()
    if name == "mae":          return nn.L1Loss()
    if name == "huber":        return nn.SmoothL1Loss(beta=kwargs.get("beta", 0.5))
    if name in ("mse+pearson","mse_corr","combo"):
        return MSEPlusCorrLoss(alpha=kwargs.get("alpha", 0.7))
    if name in ("huber_corr","huber+pearson"):
        return HuberPlusCorr(beta=kwargs.get("beta", 0.25), alpha=kwargs.get("alpha", 0.6))
    raise ValueError(f"Unknown loss: {name}")

# =============== Metrics (for ad-hoc use) ===============
@torch.no_grad()
def regression_metrics(preds, target):
    if preds.dim()==2 and preds.size(1)==1: preds = preds.squeeze(1)
    if target.dim()==2 and target.size(1)==1: target = target.squeeze(1)
    mae = (preds - target).abs().mean().item()
    rmse = torch.sqrt(torch.mean((preds - target)**2)).item()
    ss_res = torch.sum((preds - target) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2) + 1e-12
    r2 = 1 - ss_res/ss_tot
    vx = preds - preds.mean()
    vy = target - target.mean()
    pearson = (vx*vy).sum() / (torch.sqrt((vx*vx).sum()+1e-8) * torch.sqrt((vy*vy).sum()+1e-8))
    return {"MAE": mae, "RMSE": rmse, "R2": r2.item(), "Pearson": pearson.item()}
