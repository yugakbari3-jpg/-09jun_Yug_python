"""
SybilNet -- a faithful re-implementation of the network described in

    Mikhael, P.G., Wohlwend, J., Yala, A. et al. "Sybil: A Validated Deep Learning
    Model to Predict Future Lung Cancer Risk From a Single Low-Dose Chest Computed
    Tomography." J Clin Oncol 41, 2191-2200 (2023).  (MIT Jameel Clinic / MGH)

Architecture
------------
    LDCT volume (B, 3, 200, 256, 256)
        -> 3D ResNet-18 (r3d_18, Kinetics-400 init) without avgpool / fc
        -> feature map (B, 512, T', H', W')
        -> MultiAttentionPool
             * per-slice spatial attention  (image_attention_1)
             * across-slice attention       (volume_attention_1)
             * per-slice max-pool + Conv1d + across-slice attention
             * global max-pool
        -> 512-d hidden -> ReLU -> Dropout
        -> Cumulative_Probability_Layer  (additive hazards, monotone in time)
        -> logits for years 1..6

Layer and parameter names match the official implementation, so the released
checkpoints load straight into this module (see ``sybil.weights``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


# --------------------------------------------------------------------------- #
#  Pooling blocks
# --------------------------------------------------------------------------- #
class Simple_AttentionPool(nn.Module):
    """Attention-weighted average over the last dimension. x: (B, C, N)."""

    def __init__(self, num_chan: int = 512, **kwargs):
        super().__init__()
        self.attention_fc = nn.Linear(num_chan, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        B, C = x.shape[:2]
        x = x.view(B, C, -1)
        scores = self.attention_fc(x.transpose(1, 2)).transpose(1, 2)  # B, 1, N
        out = {"volume_attention": self.logsoftmax(scores).view(B, -1)}
        out["hidden"] = torch.sum(x * self.softmax(scores), dim=-1)
        return out


class Simple_AttentionPool_MultiImg(nn.Module):
    """Spatial attention inside every slice. x: (B, C, T, W, H) -> (B, C, T)."""

    def __init__(self, num_chan: int = 512, **kwargs):
        super().__init__()
        self.attention_fc = nn.Linear(num_chan, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        B, C, T, W, H = x.size()
        x = x.permute([0, 2, 1, 3, 4]).contiguous().view(B * T, C, W * H)
        scores = self.attention_fc(x.transpose(1, 2)).transpose(1, 2)  # B*T, 1, WH
        out = {"image_attention": self.logsoftmax(scores).view(B, T, -1)}
        x = torch.sum(x * self.softmax(scores), dim=-1)
        out["multi_image_hidden"] = x.view(B, T, C).permute([0, 2, 1]).contiguous()
        return out


class PerFrameMaxPool(nn.Module):
    """Max over the in-plane dims of every slice. (B, C, T, W, H) -> (B, C, T)."""

    def forward(self, x):
        B, C, T = x.shape[:3]
        return {"multi_image_hidden": x.view(B, C, T, -1).max(-1)[0]}


class Conv1d_AttnPool(nn.Module):
    """Conv1d along the slice axis followed by attention pooling."""

    def __init__(self, num_chan: int = 512, conv_pool_kernel_size: int = 11, stride: int = 1, **kwargs):
        super().__init__()
        self.conv1d = nn.Conv1d(
            num_chan, num_chan, kernel_size=conv_pool_kernel_size,
            stride=stride, padding=conv_pool_kernel_size // 2, bias=False,
        )
        self.aggregate = Simple_AttentionPool(num_chan=num_chan)

    def forward(self, x):
        return self.aggregate(self.conv1d(x))


class GlobalMaxPool(nn.Module):
    def forward(self, x):
        B, C = x.shape[:2]
        return {"hidden": x.view(B, C, -1).max(-1)[0]}


class MultiAttentionPool(nn.Module):
    def __init__(self, num_chan: int = 512):
        super().__init__()
        params = {"num_chan": num_chan, "conv_pool_kernel_size": 11, "stride": 1}
        self.image_pool1 = Simple_AttentionPool_MultiImg(**params)
        self.volume_pool1 = Simple_AttentionPool(**params)
        self.image_pool2 = PerFrameMaxPool()
        self.volume_pool2 = Conv1d_AttnPool(**params)
        self.global_max_pool = GlobalMaxPool()
        self.multi_img_hidden_fc = nn.Linear(2 * num_chan, num_chan)
        self.hidden_fc = nn.Linear(3 * num_chan, num_chan)

    def forward(self, x):
        out = {}
        img1 = self.image_pool1(x)
        vol1 = self.volume_pool1(img1["multi_image_hidden"])
        img2 = self.image_pool2(x)
        vol2 = self.volume_pool2(img2["multi_image_hidden"])
        for pool_out, n in [(img1, 1), (vol1, 1), (img2, 2), (vol2, 2)]:
            for k, v in pool_out.items():
                out[f"{k}_{n}"] = v

        out["maxpool_hidden"] = self.global_max_pool(x)["hidden"]
        mih = torch.cat([img1["multi_image_hidden"], img2["multi_image_hidden"]], dim=-2)
        out["multi_image_hidden"] = self.multi_img_hidden_fc(mih.permute([0, 2, 1])).permute([0, 2, 1]).contiguous()
        hidden = torch.cat([vol1["hidden"], vol2["hidden"], out["maxpool_hidden"]], dim=-1)
        out["hidden"] = self.hidden_fc(hidden)
        return out


# --------------------------------------------------------------------------- #
#  Survival head
# --------------------------------------------------------------------------- #
class Cumulative_Probability_Layer(nn.Module):
    """
    Predicts non-negative per-year hazards h_1..h_T plus a base hazard b and
    returns cumulative logits  z_t = b + sum_{i<=t} h_i, which are monotonically
    non-decreasing in t -- risk can never go *down* with longer follow-up.
    """

    def __init__(self, num_features: int = 512, max_followup: int = 6):
        super().__init__()
        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.t(torch.tril(torch.ones([max_followup, max_followup])))
        self.register_parameter("upper_triagular_mask", nn.Parameter(mask, requires_grad=False))

    def hazards(self, x):
        return self.relu(self.hazard_fc(x))

    def forward(self, x):
        h = self.hazards(x)
        B, T = h.size()
        masked = h.unsqueeze(-1).expand(B, T, T) * self.upper_triagular_mask
        return torch.sum(masked, dim=1) + self.base_hazard_fc(x)


# --------------------------------------------------------------------------- #
#  Full network
# --------------------------------------------------------------------------- #
class SybilNet(nn.Module):
    def __init__(self, max_followup: int = 6, dropout: float = 0.2, pretrained_encoder: bool = False):
        super().__init__()
        self.hidden_dim = 512
        weights = None
        if pretrained_encoder:
            weights = torchvision.models.video.R3D_18_Weights.KINETICS400_V1
        encoder = torchvision.models.video.r3d_18(weights=weights)
        self.image_encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.pool = MultiAttentionPool(self.hidden_dim)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout)
        self.prob_of_failure_layer = Cumulative_Probability_Layer(self.hidden_dim, max_followup=max_followup)

    def forward(self, x):
        """x: (B, 3, T, H, W) normalised CT. Returns dict with logits + attentions."""
        feats = self.image_encoder(x)
        out = self.pool(feats)
        out["hidden"] = self.dropout(self.relu(out["hidden"]))
        out["logit"] = self.prob_of_failure_layer(out["hidden"])
        out["activ"] = feats
        return out
