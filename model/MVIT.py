import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math

from model.ViT import Encoder, PatchEmbed
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class Detection_Header(nn.Module):

    def __init__(self, use_bn=True,reg_layer=2,input_angle_size=0):
        super(Detection_Header, self).__init__()

        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.input_angle_size = input_angle_size
        self.target_angle = 224
        bias = not use_bn

        if(self.input_angle_size==224):
            self.conv1 = conv3x3(256, 144, bias=bias)
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==448):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif(self.input_angle_size==896):
            self.conv1 = conv3x3(256, 144, bias=bias,stride=(1,2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias,stride=(1,2))
            self.bn2 = nn.BatchNorm2d(96)
        else:
            raise NameError('Wrong channel angle paraemter !')
            return

        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)

        self.clshead = conv3x3(96, 1, bias=True)
        self.reghead = conv3x3(96, reg_layer, bias=True)
            
    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)
    
        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)

        return torch.cat([cls, reg], dim=1)
    
class Detection_Header2(nn.Module):
    def __init__(self, D, Hp, Wp, use_bn=True, reg_layer=2):
        super(Detection_Header2, self).__init__()
        self.D = D
        self.Hp = Hp
        self.Wp = Wp
        self.use_bn = use_bn
        bias = not use_bn

        self.conv1 = conv3x3(D, 144, bias=bias)
        self.bn1 = nn.BatchNorm2d(144)
        self.conv2 = conv3x3(144, 96, bias=bias)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)

        self.clshead = conv3x3(96, 1, bias=True)
        self.reghead = conv3x3(96, reg_layer, bias=True)
    
    def forward(self, x):
        B, Np, D = x.shape
        x = x.permute(0,2,1).contiguous()
        x = x.view(B,D, self.Hp, self.Wp)
        x = F.interpolate(x, size=(128, 64), mode='bilinear', align_corners=False)  # (B, D, 128, 224)
        
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)
        return torch.cat([cls, reg], dim=1)

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out

class RangeAngle_Decoder(nn.Module):
    """
    Adapts the Swin transpose-trick for ViT features (B, C, H, W=16).
    L projections: C → 224, then transpose(1,3) swaps C↔W:
        (B, 224, H, 16) → (B, 16, H, 224)
    Three deconvs double Range (H) at each step: 16→32→64→128
    Output: (B, 256, 128, 224) → compatible with Detection_Header(input_angle_size=224)

    Input features:
      features[0]: (B, D,   128, 16)  feat0 — y upsampled x4 in Range
      features[1]: (B, 160,  64, 16)  feat1
      features[2]: (B, 192,  32, 16)  feat2
      features[3]: (B,  16,  16, 16)  feat3
    """
    def __init__(self, D=256):
        super(RangeAngle_Decoder, self).__init__()

        self.L4 = nn.Conv2d( 16, 224, kernel_size=1)   
        self.L3 = nn.Conv2d(192, 224, kernel_size=1)   
        self.L2 = nn.Conv2d(160, 224, kernel_size=1)   
        self.L1 = nn.Conv2d(  D, 224, kernel_size=1) 

        self.deconv4 = nn.ConvTranspose2d( 16,  16, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))  # H: 16→32
        self.deconv3 = nn.ConvTranspose2d( 64,  64, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))  # H: 32→64
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))  # H: 64→128

        self.conv_block4 = BasicBlock( 32,  64)   # 16+16=32
        self.conv_block3 = BasicBlock( 80, 128)   # 64+16=80
        self.conv_block2 = BasicBlock(144, 256)   # 128+16=144

    def forward(self, features):
        T4 = self.L4(features[3]).transpose(1, 3)  # (B, 16,  16, 224)
        T3 = self.L3(features[2]).transpose(1, 3)  # (B, 16,  32, 224)
        T2 = self.L2(features[1]).transpose(1, 3)  # (B, 16,  64, 224)
        T1 = self.L1(features[0]).transpose(1, 3)  # (B, 16, 128, 224)

        S4  = torch.cat((self.deconv4(T4), T3), dim=1)  # (B, 32,  32, 224)
        S4  = self.conv_block4(S4)                       # (B, 64,  32, 224)

        S43 = torch.cat((self.deconv3(S4), T2), dim=1)  # (B, 80,  64, 224)
        S43 = self.conv_block3(S43)                      # (B,128,  64, 224)

        out = torch.cat((self.deconv2(S43), T1), dim=1) # (B,144, 128, 224)
        out = self.conv_block2(out)                      # (B,256, 128, 224)

        return out

class AntennaEncoder(nn.Module):
    def __init__(self, D, p, H, W, neuron, mha, layer, dropout):
        super().__init__()
        self.Hp = H // p
        self.Wp = W // p

        self.patch_embed = PatchEmbed(patch_size=(p,p), in_chans=2, embed_dim=D)
        self.pe = PositionalEncoding(D)

        self.encoder = nn.ModuleList([Encoder(D, neuron * D, mha, dropout) for _ in range(layer)])

    def forward(self, re, im):
        x = torch.stack([re, im], dim=1) 
        x = self.patch_embed(x)
        x = self.pe(x.flatten(2).transpose(1, 2))
        for layer in self.encoder:
            x = layer(x, None)
        return x

class FusionTransformer(nn.Module):
    def __init__(self, D, mha, layer, dropout, Nrx):
        super().__init__()
        self.Nrx = Nrx
        self.antenna_pe = nn.Parameter(torch.randn(1, Nrx, 1, D))
        self.fusion_layers = nn.ModuleList([
            Encoder(D, D*4, mha, dropout) for _ in range(layer)
        ])
        self.fusion_proj = nn.Linear(Nrx * D, D)

    def forward(self, antenna_tokens):
        B, Np, D = antenna_tokens[0].shape

        x = torch.stack(antenna_tokens, dim=1)    # (B, Nrx, Np, D)
        x = x + self.antenna_pe                   # (B, Nrx, Np, D)

        x = x.permute(0, 2, 1, 3).contiguous()   # (B, Np, Nrx, D)
        x = x.view(B * Np, self.Nrx, D)          # (B*Np, Nrx, D)

        for layer in self.fusion_layers:
            x = layer(x, None)                    # (B*Np, Nrx, D)

        x = x.reshape(B * Np, self.Nrx * D)
        x = self.fusion_proj(x)                   # (B*Np, D)
        x = x.view(B, Np, D)                      # (B, Np, D)
        return x

class MViT(nn.Module):
    def __init__(self, D, p, H, W, neuron, mha, layer, dropout, Nrx=16):
        super().__init__()
        self.Nrx = Nrx
        self.Hp = H // p 
        self.Wp = W // p
        self.D = D

        self.antenna_encoder = AntennaEncoder(D, p, H, W, neuron, mha, layer, dropout)
        self.fusion = FusionTransformer(D, mha, layer, dropout, Nrx=Nrx)

        self.down  = nn.Conv2d(D,   16, kernel_size=3, stride=(2,1), padding=1)
        self.proj  = nn.Conv2d(D,  192, kernel_size=1)
        self.up2   = nn.ConvTranspose2d(D, 160, kernel_size=(2,1), stride=(2,1))
        self.up4   = nn.ConvTranspose2d(160,   D, kernel_size=(2,1), stride=(2,1))

        self.ra_decoder     = RangeAngle_Decoder(D=D)
        self.detection_head = Detection_Header(
            use_bn=True, reg_layer=2, input_angle_size=224
        )

    def forward(self, x, x_mask=None):
        # x: (B, 2*NRx, H, W) — re puis im pour chaque Rx
        B = x.shape[0]

        # ---- Encoder per antena ----
        antenna_tokens = []
        for i in range(self.Nrx):
            re = x[:, i, :, :]   # (B, H, W)
            im = x[:, i + self.Nrx, :, :]   # (B, H, W)
            tokens = self.antenna_encoder(re, im)  # (B, Np, D)
            antenna_tokens.append(tokens)

        # ---- Fusion inter-antennes ----
        fused = self.fusion(antenna_tokens)  # (B, Np, D)

        # ---- Reshape + multi-scale ----
        y = fused.permute(0,2,1).contiguous()
        y = y.view(B, self.D, self.Hp, self.Wp)

        feat3 = self.down(y)      # (B,  16, Hp/2, Wp)
        feat2 = self.proj(y)      # (B, 192, Hp,   Wp)
        feat1 = self.up2(y)       # (B, 160, Hp*2, Wp)
        feat0 = self.up4(feat1)   # (B,   D, Hp*4, Wp)

        out = self.ra_decoder([feat0, feat1, feat2, feat3])
        out = self.detection_head(out)
        return {"Detection": out}