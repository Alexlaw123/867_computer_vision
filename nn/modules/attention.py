import torch
import torch.nn as nn
import inspect

# YOLOv8-compatible SE Attention Block
class SEBlock(nn.Module):
    def __init__(self, c1, c2=None, r=16):  # 输入通道c1，输出通道c2（通常一样）
        super().__init__()
        c2 = c1 if c2 is None else c2

        if c1 % r != 0:
            raise ValueError(f"Channel c1={c1} should be divisible by reduction ratio r={r}.")

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),              # 全局池化
            nn.Conv2d(c1, c1 // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // r, c2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.se(x)
        print(f"[SEBlock] x.shape={x.shape}, se(x).shape={y.shape}")
        return x * y



class LightTransformerBlock(nn.Module):
    def __init__(self, c1, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(c1, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(c1, c1),
            nn.ReLU(inplace=True),
            nn.Linear(c1, c1),
        )
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)

    def forward(self, x):
        B, C, H, W = x.shape
        print(f"[LightTransformer] input: {x.shape}")
        x_ = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        attn_out, _ = self.attn(x_, x_, x_)
        x = x_ + attn_out
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        print(f"[LightTransformer] output: {x.shape}")
        return x


class MyMobileViTBlock(nn.Module):
    def __init__(self, c1, c2=0, patch_size=(2, 2), num_heads=2, *args, **kwargs):
        super().__init__()
        print(f"[MyMobileViTBlock] 输入参数: c1={c1}, c2={c2}, patch_size={patch_size}, num_heads={num_heads}")

        # ✅ 类型处理
        if isinstance(patch_size, str):
            patch_size = eval(patch_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, (list, tuple)) and len(patch_size) == 2:
            patch_size = tuple(patch_size)
        else:
            raise ValueError(f"[MyMobileViTBlock] patch_size 参数无效: {patch_size}")

        self.ph, self.pw = patch_size
        if self.ph <= 0 or self.pw <= 0:
            raise ValueError("[MyMobileViTBlock] patch_size 中的高度和宽度必须大于 0")

        c1 = int(c1)
        c2 = int(c2) if isinstance(c2, (int, float)) and c2 > 0 else c1
        num_heads = int(num_heads) if isinstance(num_heads, (int, float, str)) else 2

        # ✅ 关键：告知 YOLO 模型该模块输入/输出通道不变
        self.in_channels = c1
        self.out_channels = c1
        self.ch = c1

        # Local branch
        self.local = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        # Transformer branch
        self.proj = nn.Conv2d(c1, c2, 1, bias=False)
        self.attn = nn.TransformerEncoderLayer(
            d_model=c2 * self.ph * self.pw,
            nhead=num_heads,
            batch_first=True
        )
        self.reproj = nn.Conv2d(c2, c1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_local = self.local(x)

        # Patch division
        new_H = H - (H % self.ph) if H % self.ph != 0 else H
        new_W = W - (W % self.pw) if W % self.pw != 0 else W
        x_crop = x_local[:, :, :new_H, :new_W]

        x2 = self.proj(x_crop)
        B, C2, H2, W2 = x2.shape

        # Patch unfolding
        x2 = x2.view(B, C2, H2 // self.ph, self.ph, W2 // self.pw, self.pw)
        x2 = x2.permute(0, 2, 4, 3, 5, 1).contiguous().view(B, -1, self.ph * self.pw * C2)

        # Transformer
        x2 = self.attn(x2)

        # Reshape back
        x2 = x2.view(B, H2 // self.ph, W2 // self.pw, self.ph, self.pw, C2)
        x2 = x2.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C2, H2, W2)

        x2 = self.bn(self.reproj(x2))

        if new_H != H or new_W != W:
            pad = (0, W - new_W, 0, H - new_H)
            x2 = F.pad(x2, pad)

        out = self.fuse(torch.cat([x_local, x2], dim=1))
        return out
