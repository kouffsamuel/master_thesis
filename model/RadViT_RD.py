import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math

from utils.pos_embed import get_2d_sincos_pos_embed
from model.fourier_net import FFT_Net
from model.ViT import Encoder, PatchEmbed, PositionalEncoding

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)
class Detection_Header(nn.Module):

    def __init__(self, use_bn=True, reg_layer=2, num_classes=4):
        super().__init__()
        self.use_bn = use_bn
        bias = not use_bn

        self.conv1 = conv3x3(256, 144, bias=bias)
        self.bn1   = nn.BatchNorm2d(144)
        self.conv2 = conv3x3(144,  96, bias=bias)
        self.bn2   = nn.BatchNorm2d(96)
        self.conv3 = conv3x3( 96,  96, bias=bias)
        self.bn3   = nn.BatchNorm2d(96)
        self.conv4 = conv3x3( 96,  96, bias=bias)
        self.bn4   = nn.BatchNorm2d(96)

        self.clshead = conv3x3(96,          1, bias=True)
        self.reghead = conv3x3(96,  reg_layer, bias=True)
        self.cathead = conv3x3(96, num_classes, bias=True)

    def forward(self, x):
        # x: (B, 256, 256, 128)
        x = F.relu(self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x))
        x = F.relu(self.bn3(self.conv3(x)) if self.use_bn else self.conv3(x))
        x = F.relu(self.bn4(self.conv4(x)) if self.use_bn else self.conv4(x))

        cls = torch.sigmoid(self.clshead(x))   # (B, 1,   256, 128)
        reg = self.reghead(x)                   # (B, 2,   256, 128)
        cat = self.cathead(x)                   # (B, 4,   256, 128)

        return torch.cat([cls, reg, cat], dim=1)
    
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

class RangeDoppler_Decoder(nn.Module):
    def __init__(self, D=256):
        super().__init__()
        # Projections 1×1 vers 128 canaux
        self.L3 = nn.Conv2d( 16, 128, kernel_size=1)   # (16,  8)
        self.L2 = nn.Conv2d(192, 128, kernel_size=1)   # (32, 16)
        self.L1 = nn.Conv2d(160, 128, kernel_size=1)   # (64, 32)
        self.L0 = nn.Conv2d(  D, 128, kernel_size=1)   # (128,64)

        # Deconv ×2 sur range ET doppler à chaque étage
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # (16,8)  → (32,16)
        self.block3  = BasicBlock(256, 128)

        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # (32,16) → (64,32)
        self.block2  = BasicBlock(256, 128)

        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # (64,32) → (128,64)
        self.block1  = BasicBlock(256, 256)

        self.deconv0 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # (128,64)→ (256,128)
        self.block0  = BasicBlock(256, 256)

    def forward(self, features):
        f0, f1, f2, f3 = features  # (B,D,128,64), (B,160,64,32), (B,192,32,16), (B,16,16,8)

        t3 = self.L3(f3)   # (B,128, 16,  8)
        t2 = self.L2(f2)   # (B,128, 32, 16)
        t1 = self.L1(f1)   # (B,128, 64, 32)
        t0 = self.L0(f0)   # (B,128,128, 64)

        x = self.block3(torch.cat([self.deconv3(t3), t2], dim=1))  # (B,128, 32, 16)
        x = self.block2(torch.cat([self.deconv2(x),  t1], dim=1))  # (B,128, 64, 32)
        x = self.block1(torch.cat([self.deconv1(x),  t0], dim=1))  # (B,256,128, 64)
        x = self.block0(self.deconv0(x))                            # (B,256,256,128)

        return x


class RadViT_RD(nn.Module):
    def __init__(self, D, p, H, W, neuron, mha, layer, dropout, n_encoders=1, data_mode='Custom_RD'):
        super().__init__()
        # ViT network parameters
        self.D = D  # dimension embedding
        self.p = p   # image patch size
        self.H = H  # image height
        self.W = W  # image width
        self.neuron = neuron * self.D  # dimension of the MLP
        self.mha = mha  # dimension of the multi-head attention
        self.layer = layer   # number of layer in the encoder
        self.dropout = dropout  # dropout, if needed
        self.n_encoders = n_encoders
        self.Hp = self.H // self.p
        self.Wp = self.W // self.p
        self.data_mode = data_mode

        # Nombre total de patchs par image
        self.Np = self.Hp * self.Wp

        # Embedding linéaire de chaque patch
        self.patch_embed = nn.Linear(2 *self.p ** 2, self.D)
        
        # # Positional encodings séparés
        self.pe = PositionalEncoding(self.D)

        if self.data_mode == 'ADC':
            self.DFT = FFT_Net()


        self.encoder_layers = nn.ModuleList([
            Encoder(
                hidden_size=self.D,
                ff_size=self.neuron,
                multi_head=self.mha,
                dropout_rate=self.dropout
            )
            for _ in range(self.layer)])
        
        self.feat_3 = nn.Conv2d(self.D, 16, kernel_size=1)
        self.feat_2 = nn.Conv2d(self.D, 192, kernel_size=1)
        self.feat_1 = nn.Conv2d(self.D, 160, kernel_size=1)
        self.feat_0 = nn.Conv2d(self.D, self.D, kernel_size=1)  # feat0 projection (identity)

        self.rd_decoder = RangeDoppler_Decoder(D=self.D)
        self.detection_head = Detection_Header(use_bn=True, reg_layer=2)

    def forward(self, x, x_mask=None):
        B = x.shape[0]  # batch
        T = x.shape[1]  # number of antennas (or channels)
        
        if self.data_mode == 'ADC':
            x = torch.view_as_complex(x)
            x = self.DFT(x)
       
        antenna_outputs = []
        for i in range(T//2):
            re = x[:, i, :, :]
            im = x[:, i + T//2, :, :]   
            re = re.view(B, self.Np, self.p * self.p)  # (B, Np, p²)
            im = im.view(B, self.Np, self.p * self.p)  # (B, Np, p²)
            xi = torch.cat([re, im], dim=-1)           # (B, Np, 2*p²)
            xi = self.patch_embed(xi)  # (B, Np, D) # 256/26 = 16 tokens 
            xi = self.pe(xi)  # (B, Np, D)

            # Encoder 
            for layer in self.encoder_layers:
                xi = layer(xi, x_mask)
            antenna_outputs.append(xi)

        x = torch.stack(antenna_outputs, dim=1).mean(dim=1)
                
        x = x.permute(0, 2, 1).contiguous()  # (B, D, Np)
        y = x.view(B, self.D, self.Hp, self.Wp) # [4, 256, 32, 16]

        # ---- Multi-scale features ----
    
        # y: (B, D, 32, 16)
        feat3 = self.feat_3(F.avg_pool2d(y, 2))                                       # (B, 16,  16, 8)
        feat2 = self.feat_2(y)                                                        # (B, 192,  32, 16)
        feat1 = self.feat_1(F.interpolate(y, (64, 32),  mode='bilinear', align_corners=False))                                                 # (B, 160,  64, 16)
        feat0 = self.feat_0(F.interpolate(y, size=(128, 64), mode='bilinear', align_corners=False))                                                 # (B,   D, 128, 16)

        out = self.rd_decoder([feat0, feat1, feat2, feat3])  # Should BE : (B, 256, 256, 128)
        out = self.detection_head(out)                      # Should BE : (B,   3, 256, 128)

        return {"Detection": out}