"""DaD-Ca model architecture (PyTorch).

This module contains only the neural network definition and its internal
loss computation helpers. No dataset loading, preprocessing, training loop,
evaluation, visualization, file paths, or experiment outputs are included.

Input: EEG differential-entropy features, [batch, 1, time=5, channels, bands=5].
Outputs: class logits and, in training mode, domain/decoupling losses.
The DSFE input is the residual (aligned incoming features - invariant features).
"""

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

class ECA_layer(nn.Module):

    def __init__(self, k_size=3):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv3 = nn.Conv1d(2, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y_avg = self.conv1(y_avg.squeeze(-1).transpose(-1, -2))
        y_max = self.conv2(y_max.squeeze(-1).transpose(-1, -2))
        y = torch.concat((y_max, y_avg), dim=1)
        y = self.conv3(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SSTAttentionBlock(nn.Module):

    def __init__(self, spaSize, speSize, temSize, embed_dim, num_heads=4, dropOut=0.1):
        super().__init__()
        self.spaSize = spaSize
        self.speSize = speSize
        self.temSize = temSize
        self.eca_layer = ECA_layer()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()
        if dropOut is not None:
            self.drop = nn.Dropout(dropOut)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropOut)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim))

    def forward(self, x):
        B, C, T, S, F = x.shape
        x_flat = rearrange(x, 'b c t s f -> (b c) t s f')
        temAtt = self.eca_layer(x_flat)
        AvgAtt = torch.mean(x_flat, dim=1, keepdim=True)
        MaxAtt, _ = torch.max(x_flat, dim=1, keepdim=True)
        spaFreAtt = torch.cat((AvgAtt, MaxAtt), dim=1)
        spaFreAtt = self.conv(spaFreAtt)
        spaFreAtt = self.activation(spaFreAtt)
        if hasattr(self, 'drop'):
            spaFreAtt = self.drop(spaFreAtt)
        if spaFreAtt.dim() == 4 and spaFreAtt.size(1) == 1:
            spaFreAtt = spaFreAtt.squeeze(1)
        tem_seq = rearrange(temAtt, '(b c) t s f -> b (t s f) c', b=B, c=C)
        spa_seq = rearrange(spaFreAtt, '(b c) s f -> b (s f) c', b=B, c=C)
        Q = self.q_proj(tem_seq)
        K = self.k_proj(spa_seq)
        V = self.v_proj(spa_seq)
        Q_norm = self.norm1(Q)
        K_norm = self.norm1(K)
        V_norm = self.norm1(V)
        attn_out, _ = self.cross_attn(Q_norm, K_norm, V_norm, need_weights=False)
        out_seq = tem_seq + attn_out
        out_seq = out_seq + self.ffn(self.norm2(out_seq))
        res = rearrange(out_seq, 'b (t s f) c -> b c t s f', b=B, t=T, s=S, f=F)
        return res


class LoRA(nn.Module):

    def __init__(self, inplanes, midplanes, outplanes, downsample):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, inplanes, kernel_size=1, bias=False)
        self.in1 = nn.BatchNorm3d(inplanes, affine=True)
        self.pool2 = nn.AvgPool3d(kernel_size=(1, 2, 2), ceil_mode=True)
        self.pool4 = nn.AvgPool3d(kernel_size=(1, 2, 2), ceil_mode=True)
        self.pool8 = nn.AvgPool3d(kernel_size=(1, 2, 2), ceil_mode=True)
        self.conv2_1 = nn.Conv3d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv2_3 = nn.Conv3d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv2_5 = nn.Conv3d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv2_7 = nn.Conv3d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv2_1_r = nn.Conv3d(midplanes, midplanes, kernel_size=1, bias=False)
        self.conv2_3_r = nn.Conv3d(midplanes, midplanes, kernel_size=1, bias=False)
        self.conv2_5_r = nn.Conv3d(midplanes, midplanes, kernel_size=1, bias=False)
        self.conv2_7_r = nn.Conv3d(midplanes, midplanes, kernel_size=1, bias=False)
        self.conv3 = nn.Conv3d(midplanes, outplanes, kernel_size=1, bias=False)
        self.in2 = nn.BatchNorm3d(midplanes, affine=True)
        self.in3 = nn.BatchNorm3d(outplanes, affine=True)
        self.in4 = nn.BatchNorm3d(outplanes, affine=True)
        self.avgpool = nn.AvgPool3d(kernel_size=(1, 2, 2)) if downsample else nn.Identity()
        self.zero_conv = nn.Conv3d(outplanes, outplanes, kernel_size=1, bias=False)
        nn.init.zeros_(self.zero_conv.weight)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.in1(self.conv1(x)))
        out_1 = self.conv2_1_r(self.relu(self.conv2_1(out)))
        _, C, D, H, W = out_1.shape
        out_2 = self.conv2_3_r(F.interpolate(self.relu(self.conv2_3(self.pool2(out))), size=(D, H, W), mode='trilinear', align_corners=False))
        out_3 = self.conv2_5_r(F.interpolate(self.relu(self.conv2_5(self.pool4(out))), size=(D, H, W), mode='trilinear', align_corners=False))
        out_4 = self.conv2_7_r(F.interpolate(self.relu(self.conv2_7(self.pool8(out))), size=(D, H, W), mode='trilinear', align_corners=False))
        out = self.relu(self.in2(out_1 + out_2 + out_3 + out_4))
        out = self.relu(self.in3(self.conv3(out)))
        out = self.avgpool(out)
        out = self.zero_conv(out)
        out = self.in4(out)
        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channels, bottleneck, dropOut, growth_rate, bn_size):
        super().__init__()
        self.bottleneck = bottleneck
        self.batchNorm = nn.BatchNorm3d(in_channels)
        self.activation = nn.ReLU()
        self.convSS = nn.Conv3d(in_channels, bn_size * growth_rate, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.convST = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.convSST = nn.Conv3d(in_channels, growth_rate, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.bottleConv = nn.Conv3d(in_channels, bn_size * growth_rate, kernel_size=(1, 1, 1))
        self.convSSBottle = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=(3, 3, 1), padding=(0, 1, 1))
        self.convSTBottle = nn.Conv3d(growth_rate, growth_rate, kernel_size=(1, 1, 3), padding=(0, 0, 1))

    def forward(self, x):
        x = self.batchNorm(x)
        x = self.activation(x)
        if self.bottleneck:
            x = self.bottleConv(x)
            x = self.activation(x)
            x = self.convSSBottle(x)
            x = self.convSTBottle(x)
        else:
            x = self.convSST(x)
        return x


class DenseBlock(nn.ModuleDict):

    def __init__(self, num_input_features, nb_layers, growth_rate, bn_size, bottleNeck=False, dropOut=None):
        super().__init__()
        for i in range(nb_layers):
            layer = ConvBlock(num_input_features + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size, bottleneck=bottleNeck, dropOut=dropOut)
            self.add_module(f'number {i} ConvBlock', layer)

    def forward(self, x):
        features = [x]
        for name, layer in self.items():
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.neg() * ctx.alpha, None)


class DA_ada_3d(nn.Module):
    """Feature backbone with residual-based DSFE and TSF cross-attention."""

    def __init__(self, num_classes=2, out_features=None, pool_vec=True, chaSize=64, spaSize=64, speSize=5, temSize=5, nb_layers_per_block=4, growth_rate=16, dropOut=0.5, bn_size=4, bottleNeck=False):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.Identity()
        self.chaSize = chaSize
        self.spaSize = spaSize
        self.speSize = speSize
        self.temSize = temSize
        self.bn_size = bn_size
        self.dropOut = dropOut
        self.bottleNeck = bottleNeck
        self.growth_rate = growth_rate
        self.nb_layers_per_block = nb_layers_per_block
        self.layer1 = self._make_sst_block(self.chaSize, 'res2')
        self.layer2 = self._make_sst_block(128, 'res3')
        self.layer3 = self._make_sst_block(192, 'res4')
        self.align_conv1 = nn.Conv3d(64, 128, kernel_size=1)
        self.align_conv2 = nn.Conv3d(128, 192, kernel_size=1)
        self.dia_1 = self._make_adapter(64, 32, 128)
        self.dia_2 = self._make_adapter(128, 64, 192)
        self.dsa_1 = self._make_adapter(128, 64, 128)
        self.dsa_2 = self._make_adapter(192, 96, 192)
        self.softmax = nn.Softmax(dim=1)
        self._out_features = out_features if out_features is not None else ['res2', 'res3', 'res4']
        self.pool_vec = pool_vec
        self.attnpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_sst_block(self, in_channels, name):
        layers = nn.Sequential()
        layers.add_module(f'{name}_attention', SSTAttentionBlock(spaSize=self.spaSize, speSize=self.speSize, temSize=self.temSize, embed_dim=self.chaSize, dropOut=self.dropOut))
        dense = DenseBlock(num_input_features=in_channels, nb_layers=self.nb_layers_per_block, growth_rate=self.growth_rate, bn_size=self.bn_size, bottleNeck=self.bottleNeck, dropOut=self.dropOut)
        layers.add_module(f'{name}_dense', dense)
        out_channels = in_channels + self.nb_layers_per_block * self.growth_rate
        self.chaSize = out_channels
        return layers

    def _make_adapter(self, in_c, mid_c, out_c):
        return nn.Sequential(LoRA(in_c, mid_c, out_c, downsample=False))

    def get_features(self, x):
        outputs = {}
        dis_feat = {}
        dia_feat = {}
        dsa_feat = {}
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        temp = self.align_conv1(x)
        x_ada = self.dia_1(x)
        for i, block in enumerate(self.layer1):
            x = block(x)
        x = x + x_ada
        dis_feat['res2'] = x
        diff = self.dsa_1(temp - x)
        dsa_feat['res2'] = x * diff
        x = x + x * diff
        dia_feat['res2'] = dis_feat['res2']
        outputs['res2'] = x if 'res2' in self._out_features else None
        temp = self.align_conv2(x)
        x_ada = self.dia_2(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        x = x + x_ada
        dis_feat['res3'] = x
        diff = self.dsa_2(temp - x)
        dsa_feat['res3'] = x * diff
        x = x + x * diff
        dia_feat['res3'] = dis_feat['res3']
        outputs['res3'] = x if 'res3' in self._out_features else None
        x = self.layer3(x) if 'res4' in self._out_features else x
        outputs['res4'] = x if 'res4' in self._out_features else None
        if self.pool_vec:
            x = self.attnpool(x)
            return x
        else:
            return (outputs, dia_feat, dsa_feat)

    def forward(self, x):
        return self.get_features(x)


def offdiag_cov_decouple_loss_v2(h_inv, h_sp, target_dim=128, standardize=True, eps=1e-06):
    B, D = h_inv.shape
    if D > target_dim:
        h_inv = F.adaptive_avg_pool1d(h_inv.unsqueeze(1), target_dim).squeeze(1)
        h_sp = F.adaptive_avg_pool1d(h_sp.unsqueeze(1), target_dim).squeeze(1)
        D = target_dim
    h_inv = h_inv - h_inv.mean(dim=0, keepdim=True)
    h_sp = h_sp - h_sp.mean(dim=0, keepdim=True)
    if standardize:
        h_inv = h_inv / (h_inv.std(dim=0, unbiased=True, keepdim=True) + eps)
        h_sp = h_sp / (h_sp.std(dim=0, unbiased=True, keepdim=True) + eps)
    HH = h_sp @ h_inv.T
    total_frob2 = HH.pow(2).sum() / ((B - 1) ** 2 + eps)
    diag_dot = (h_inv * h_sp).sum(dim=0) / (B - 1 + eps)
    diag_sq_sum = diag_dot.pow(2).sum()
    offdiag_frob2 = torch.max(total_frob2 - diag_sq_sum, torch.tensor(0.0, device=h_inv.device))
    loss = offdiag_frob2 / (D * D + eps)
    return loss


class DANNModel(nn.Module):
    """DaD-Ca model with classification and domain-discrimination heads."""

    def __init__(self, nb_classes):
        super().__init__()
        self.feature = DA_ada_3d(nb_classes, out_features=['res2', 'res3', 'res4'], pool_vec=False)
        self.class_classifier = nn.Sequential(nn.Conv3d(in_channels=256, out_channels=nb_classes, kernel_size=1), nn.AdaptiveAvgPool3d(1), nn.Flatten(start_dim=1))
        self.domain_classifiers = nn.ModuleDict({'res2': nn.Sequential(nn.Conv3d(128, 2, kernel_size=1), nn.AdaptiveAvgPool3d(1), nn.Flatten()), 'res3': nn.Sequential(nn.Conv3d(192, 2, kernel_size=1), nn.AdaptiveAvgPool3d(1), nn.Flatten())})

    def forward(self, x, alpha, is_source=True):
        outputs, dia_feat, dsa_feat = self.feature(x)
        class_output = self.class_classifier(outputs['res4'])
        if self.training:
            batch_size = x.size(0)
            device = x.device
            domain_loss = 0
            for layer in ['res2', 'res3']:
                reversed_feat = ReverseLayerF.apply(dia_feat[layer], alpha)
                domain_pred = self.domain_classifiers[layer](reversed_feat)
                target = torch.zeros(batch_size, dtype=torch.long).to(device) if is_source else torch.ones(batch_size, dtype=torch.long).to(device)
                domain_loss += F.cross_entropy(domain_pred, target)
            domain_loss /= len(['res2', 'res3'])
            decouple_loss = 0.0
            for layer in ['res2', 'res3']:
                h_i = dia_feat[layer].flatten(1)
                h_s = dsa_feat[layer].flatten(1)
                target_dim = 128
                if h_i.size(1) > target_dim:
                    h_i = F.adaptive_avg_pool1d(h_i, target_dim)
                    h_s = F.adaptive_avg_pool1d(h_s, target_dim)
                decouple_loss += offdiag_cov_decouple_loss_v2(h_i, h_s, standardize=True)
            return (class_output, domain_loss, decouple_loss)
        else:
            device = x.device
            return (class_output, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device))
