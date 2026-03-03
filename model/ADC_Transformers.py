import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from torchvision.transforms.transforms import Sequence
from timm.models.layers import to_2tuple

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
    def __init__(self, ):
        super(RangeAngle_Decoder, self).__init__()
        
        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        
        self.conv_block4 = BasicBlock(48,128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block3 = BasicBlock(192,256)

        self.L3  = nn.Conv2d(192, 224, kernel_size=1, stride=1,padding=0)
        self.L2  = nn.Conv2d(160, 224, kernel_size=1, stride=1,padding=0)
        
        
    def forward(self,features):

        T4 = features[3].transpose(1, 3) 
        T3 = self.L3(features[2]).transpose(1, 3)
        T2 = self.L2(features[1]).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S4 = self.conv_block4(S4)
        
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        out = self.conv_block3(S43)
        
        return out

class AntennaEncoder(nn.Module):
    def __init__(self, n_chirps, n_samples, d_model=512, nhead=4, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(n_samples * 2, d_model) # Map the inputs into the d_model (dimensional embedding space) that the Transformer expects. 
        self.pos_embed = nn.Embedding(n_chirps, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, Chirps, Samples]
        B, C, S = x.shape
        x = self.input_proj(x)                        
        positions = torch.arange(C, device=x.device) # Generates 1-D tensor of sequential integer position indices used for positional encoding. [0, ..., 255]
        x = x + self.pos_embed(positions) # Injects positional information into the input embeddings by adding learned positional encodings to the projected input tensor x
        x = self.transformer(x)                       
        return x                                      


class FusionBridge(nn.Module):
    def __init__(self, n_antennas, n_chirps, d_model=256, H=256):
        super().__init__()

        self.H = H  # Range bins (256)

        # Transformer de fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        seq_len = n_antennas * n_chirps

        self.proj3 = nn.Linear(d_model, 224)  # features[3] : C=224
        self.proj2 = nn.Linear(d_model, 192)  # features[2] : C=192
        self.proj1 = nn.Linear(d_model, 160)  # features[1] : C=160

        # Projections vers les tailles spatiales H×W attendues
        # features[3] : [B, 224, H/4, 16]  → H/4*16 éléments
        # features[2] : [B, 192, H/2, 32]  → H/2*32 éléments
        # features[1] : [B, 160, H,   64]  → H*64   éléments
        self.seq_to_s3 = nn.Linear(seq_len, (H // 4) * 16)
        self.seq_to_s2 = nn.Linear(seq_len, (H // 2) * 32)
        self.seq_to_s1 = nn.Linear(seq_len,  H       * 64)

    def to_feature_map(self, x, proj, seq_to_spatial, h, w):
        """x: [B, N*Chirps, d_model] → [B, C, h, w]"""
        B = x.shape[0]
        x = proj(x)                    # [B, N*Chirps, C]
        x = x.permute(0, 2, 1)        # [B, C, N*Chirps]
        x = seq_to_spatial(x)         # [B, C, h*w]
        return x.view(B, -1, h, w)    # [B, C, h, w]

    def forward(self, antenna_features):
        # antenna_features : liste de N tenseurs [B, Chirps, d_model]
        B = antenna_features[0].shape[0]

        x = torch.stack(antenna_features, dim=1)   # [B, N, Chirps, d_model]
        N, C_dim, d = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(B, N * C_dim, d)                # [B, N*Chirps, d_model]

        x = self.fusion_transformer(x)             # [B, N*Chirps, d_model]

        # Shapes exactes pour RangeAngle_Decoder
        f3 = self.to_feature_map(x, self.proj3, self.seq_to_s3, self.H // 4, 16)

        f2 = self.to_feature_map(x, self.proj2, self.seq_to_s2, self.H // 2, 32)

        f1 = self.to_feature_map(x, self.proj1, self.seq_to_s1, self.H, 64)

        return [None, f1, f2, f3]
    
class ADC_Transformers(nn.Module):
    def __init__(
        self,
        n_antennas=16,
        n_chirps=256,
        n_samples=512,
        d_model=256,
        ra_h=128, 
        ra_w=224,
    ):
        super().__init__()

        self.n_antennas = n_antennas

        self.encoders = nn.ModuleList([
            AntennaEncoder(n_chirps, n_samples, d_model)
            for _ in range(n_antennas)
        ])

        self.fusion_bridge = FusionBridge(
            n_antennas, n_chirps, d_model, ra_h
        )

        self.ra_decoder = RangeAngle_Decoder()
        self.detection_head = Detection_Header(input_angle_size=ra_w, reg_layer=2)

    def forward(self, x):
        # Encodage par antenne
        out = {'Detection':[]}
        antenna_features = []

        for i in range(self.n_antennas):
            ant = x[:, i, :, :].permute(0, 2, 1)   
            ant_real = ant.real
            ant_imag = ant.imag
            ant = torch.cat([ant_real, ant_imag], dim=-1)
            feat = self.encoders[i](ant)
            antenna_features.append(feat)

        feature_map = self.fusion_bridge(antenna_features) 

        ra_features = self.ra_decoder(feature_map)    

        out['Detection'] = self.detection_head(ra_features)

        return out