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

def make_conv_block(in_channels, out_channels):
    return nn.Sequential(
        conv3x3(in_channels, out_channels, stride=2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def make_proj(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
def make_res_block(in_channels, out_channels):
    return nn.Sequential(
        conv3x3(in_channels, out_channels, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        conv3x3(out_channels, out_channels, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

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

        T4 = features['x4'].transpose(1, 3) 
        T3 = self.L3(features['x3']).transpose(1, 3)
        T2 = self.L2(features['x2']).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S4 = self.conv_block4(S4)
        
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        out = self.conv_block3(S43)
        
        return out

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=256, embed_dim=256):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class ADC_ViT_PerAntenna(nn.Module):
    def __init__(self, n_antennas):
        super(ADC_ViT_PerAntenna, self).__init__()

        self.n_antennas = n_antennas

        self.stem = nn.ModuleList([
            make_conv_block(2, 32),
            make_conv_block(32,  64), 
            make_conv_block(64,  128),
            make_conv_block(128, 256),
        ])

        self.proj_x2 = make_proj(self.n_antennas*64, 160)
        self.proj_x3 = make_proj(self.n_antennas*128, 192)

        self.RA_decoder = RangeAngle_Decoder()
        self.patch_embed = PatchEmbed(in_chans=256, embed_dim=256)
        self.detection_header = Detection_Header(input_angle_size=4*4,reg_layer=2)
    
    def iq_split(self,x):
        B = x.shape[0]
        if x.is_complex():
            x = torch.view_as_real(x) # (B, 512, 256, 16, 2)
            x = x.permute(0, 3, 4, 1, 2) # (B, 16, 2, 512, 256)
        return x.reshape(B * self.n_antennas, 2, x.shape[3], x.shape[4]), B
    
    def fuse_concat(self, features, B, proj):
        B, C, H, W = features.shape
        return proj(features.reshape(B, self.n_antennas *C, H, W))

    def forward(self,x):
                       
        out = {'Detection':[]}
        x,B = self.iq_split(x) # (B*16, 2, 512, 256)
        x = self.stem[0](x)
        x = self.stem[1](x)
        fx2 = x
        x = self.stem[2](x)
        fx3 = x
        x = self.stem[3](x)
        fx4 = x

        x2 = self.fuse_concat(fx2, B, self.proj_x2)
        x3 = self.fuse_concat(fx3, B, self.proj_x3)

        tok = self.patch_embed(fx4)

        RA = self.RA_decoder({'x2':x2,'x3':x3,'x4':fx4})
        out['Detection'] = self.detection_header(RA)
        
        return out