"""
Transfer Learning Model using Pretrained ResNet
Drop-in replacement for SimpleCNN - works with same training code
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetTransfer(nn.Module):
    """
    Pretrained ResNet18 with custom regression head.
    Uses transfer learning - much better for small datasets.
    
    Input: [B, in_ch, 256, 256]
    Output: [B, 1]
    """
    
    def __init__(self, in_ch=3, dropout=0.3):
        super().__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Replace first conv layer to handle in_ch channels
        if in_ch != 3:
            # Create new first conv layer
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_ch, 
                64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
            # Initialize with ImageNet pretrained weights (average across channels)
            with torch.no_grad():
                if in_ch == 1:
                    resnet.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
                else:
                    # For in_ch > 3, repeat pretrained weights
                    resnet.conv1.weight[:, :3].copy_(original_conv.weight)
                    if in_ch > 3:
                        resnet.conv1.weight[:, 3:].copy_(original_conv.weight[:, :in_ch-3])
        
        # Remove final FC layer (will replace with regression head)
        num_features = resnet.fc.in_features
        
        # Extract features (everything except final FC)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Full fine-tuning: All backbone layers are trainable
        # This allows maximum adaptation to medical images
        
        # Custom regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
    
    def freeze_backbone_params(self, freeze=True):
        """Helper method to freeze/unfreeze backbone parameters"""
        for param in self.features.parameters():
            param.requires_grad = not freeze


class EfficientNetTransfer(nn.Module):
    """
    Pretrained EfficientNet-B0 with custom regression head.
    Lightweight and efficient - good for medical imaging.
    
    Input: [B, in_ch, 256, 256]
    Output: [B, 1]
    """
    
    def __init__(self, in_ch=3, dropout=0.3):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Replace first conv layer to handle in_ch channels
        if in_ch != 3:
            original_conv = efficientnet.features[0][0]
            efficientnet.features[0][0] = nn.Conv2d(
                in_ch,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            # Initialize weights
            with torch.no_grad():
                if in_ch == 1:
                    efficientnet.features[0][0].weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
                else:
                    efficientnet.features[0][0].weight[:, :3].copy_(original_conv.weight)
                    if in_ch > 3:
                        efficientnet.features[0][0].weight[:, 3:].copy_(original_conv.weight[:, :in_ch-3])
        
        # Extract features
        self.features = efficientnet.features
        
        # Get number of features
        num_features = efficientnet.classifier[1].in_features
        
        # Custom regression head
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


if __name__ == "__main__":
    # Test ResNet
    print("Testing ResNetTransfer...")
    model_rn = ResNetTransfer(in_ch=3)
    dummy = torch.randn(2, 3, 256, 256)
    out = model_rn(dummy)
    print(f"ResNet - Input: {dummy.shape}, Output: {out.shape}")
    total = sum(p.numel() for p in model_rn.parameters())
    trainable = sum(p.numel() for p in model_rn.parameters() if p.requires_grad)
    print(f"ResNet - Total params: {total:,}, Trainable: {trainable:,}")
    
    # Test EfficientNet
    print("\nTesting EfficientNetTransfer...")
    model_ef = EfficientNetTransfer(in_ch=3)
    out = model_ef(dummy)
    print(f"EfficientNet - Input: {dummy.shape}, Output: {out.shape}")
    total = sum(p.numel() for p in model_ef.parameters())
    trainable = sum(p.numel() for p in model_ef.parameters() if p.requires_grad)
    print(f"EfficientNet - Total params: {total:,}, Trainable: {trainable:,}")

