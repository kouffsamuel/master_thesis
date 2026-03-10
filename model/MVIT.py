import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math


# Projection des vecteurs de dimensions inp_size vers la dimension d_model
class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.len = max_len
        # Créer une matrice de taille (max_len, d_model) avec des valeurs de zéro
        pe = torch.zeros(max_len, d_model)

        # Créer un vecteur représentant les positions (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Calculer les angles pour chaque position et chaque dimension
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # Appliquer les formules de sinus et cosinus aux positions pour chaque dimension
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Ajouter une dimension supplémentaire pour l'utiliser avec les batchs (batch_size, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Enregistre `pe` comme Variable pour les anciennes versions de PyTorch
        self.register_buffer('pe', Variable(pe, requires_grad=False))  # permet de gerer le device

    def forward(self, x):
        # Ajouter l'encodage positionnel aux embeddings des tokens en entrée
        x = x + self.pe[:, :x.shape[1], :]
        return x


class LayerNorm(nn.Module):  # couche Add and norm
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(size, eps=eps)

    def forward(self, x):
        return self.norm(x)


class FC(nn.Module):  # fully connected layer du MLP
    def __init__(self, in_size, out_size, dropout_r, use_relu):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU(inplace=True)  # inplace permet de gagner de la memoire en "overwrite" l input
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r, use_relu):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dimension, multi_head, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.mha = multi_head
        self.hidden_dimension = hidden_dimension

        self.linear_v = nn.Linear(hidden_dimension, hidden_dimension)
        self.linear_k = nn.Linear(hidden_dimension, hidden_dimension)
        self.linear_q = nn.Linear(hidden_dimension, hidden_dimension)
        self.linear_merge = nn.Linear(hidden_dimension, hidden_dimension)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)  # Added

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        d_k = self.hidden_dimension // self.mha

        # 1: Split V, K, Q into B x N_head x Seq length x D / N_head
        v = self.linear_v(v).view(n_batches, -1, self.mha, d_k).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.mha, d_k).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.mha, d_k).transpose(1, 2)

        # 2: Compute the attention
        atted = self.att(v, k, q, mask)

        # 3: Concat each head to form B x Seq length x D
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, d_k * self.mha)

        # 4: Apply a final layer
        atted = self.linear_merge(atted)
        atted = self.dropout2(atted)  # Added
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, hidden_dimension, ff_dimension, dropout_rate):
        super(FFN, self).__init__()
        self.mlp = MLP(
            in_size=hidden_dimension,
            mid_size=ff_dimension,
            out_size=hidden_dimension,
            dropout_r=dropout_rate,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, hidden_size, ff_size, multi_head, dropout_rate):
        super(Encoder, self).__init__()
        self.mhatt = MultiHeadAttention(hidden_dimension=hidden_size,
                                        multi_head=multi_head,
                                        dropout_rate=dropout_rate)
        self.ffn = FFN(hidden_dimension=hidden_size,
                       ff_dimension=ff_size,
                       dropout_rate=dropout_rate)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = LayerNorm(hidden_size)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm(hidden_size)

    def forward(self, x, x_mask):
        y = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        y = self.norm2(y + self.dropout2(self.ffn(y)))
        return y



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
    
# -------------------------
# ---- Main Net Model ----
# -------------------------


class MViT(nn.Module):
    def __init__(self, D, p, H, W, neuron, mha, layer, dropout):
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
        self.n_antennas = 16
        self.Hp = self.H // self.p
        self.Wp = self.W // self.p

        # Nombre total de patchs par image
        self.Np = self.Hp * self.Wp

        # Embedding linéaire de chaque patch
        self.patch_embed = nn.Linear(2 *self.p ** 2, self.D)

        # Positional encodings séparés
        self.pe = PositionalEncoding(d_model=self.D)  # nn.Parameter(torch.randn(1, self.Np, self.D))

        # self.encoders = nn.ModuleList([nn.ModuleList([
        #     Encoder(
        #         hidden_size=self.D,
        #         ff_size=self.neuron,
        #         multi_head=self.mha,
        #         dropout_rate=self.dropout
        #     )
        #     for _ in range(self.layer)
        # ]) for _ in range(self.n_antennas)])

        self.encoder_layers = nn.ModuleList([
            Encoder(
                hidden_size=self.D,
                ff_size=self.neuron,
                multi_head=self.mha,
                dropout_rate=self.dropout
            )  for _ in range(self.layer)
        ])

        self.feat_3 = nn.Conv2d(self.D, 16, kernel_size=1)
        self.feat_2 = nn.Conv2d(self.D, 192, kernel_size=1)
        self.feat_1 = nn.Conv2d(self.D, 160, kernel_size=1)
        self.feat_0 = nn.Conv2d(self.D, self.D, kernel_size=1)  # feat0 projection (identity)

        self.ra_decoder = RangeAngle_Decoder(D=self.D)
        self.detection_head = Detection_Header(use_bn=True, reg_layer=2, input_angle_size=224)
        #self.detection_head = Detection_Header2(self.D, self.Hp, self.Wp, True, 2)

    def forward(self, x, x_mask=None):
        B, T, H, W = x.shape  # batch

        antenna_outputs = []
        for i in range(T//2):
            re = x[:, i, :, :]
            im = x[:, i + T//2, :, :]   
            re = re.view(B, self.Np, self.p * self.p)  # (B, Np, p²)
            im = im.view(B, self.Np, self.p * self.p)  # (B, Np, p²)
            xi = torch.cat([re, im], dim=-1)           # (B, Np, 2*p²)
            xi = self.patch_embed(xi)  # (B, Np, D) # 256/26 = 16 tokens 
            xi = self.pe(xi)  # (B, Np, D)

            # ---- Encoder ----
            for layer in self.encoder_layers:
                xi = layer(xi, x_mask)
            
            antenna_outputs.append(xi)

        x = torch.stack(antenna_outputs, dim=1).mean(dim=1) # (B, 512, 256), (B, Np, D)
        
        x = x.permute(0, 2, 1).contiguous()  # (B, D, Np)
        y = x.view(B, self.D, self.Hp, self.Wp) # [4, 256, 32, 16]

        # ---- Multi-scale features ----

        # y: (B, D, 32, 16)
        feat2 = self.feat_2(y)                                                        # (B, 192,  32, 16)
        base_half = F.avg_pool2d(y, kernel_size=(2, 1))
        feat3 = self.feat_3(base_half)                                                # (B,  16,  16, 16)
        base_up2 = F.interpolate(y, size=(64,  16), mode='bilinear', align_corners=False)
        feat1 = self.feat_1(base_up2)                                                 # (B, 160,  64, 16)
        base_up4 = F.interpolate(y, size=(128, 16), mode='bilinear', align_corners=False)
        feat0 = self.feat_0(base_up4)                                                 # (B,   D, 128, 16)

        out = self.ra_decoder([feat0, feat1, feat2, feat3])  # (B, 256, 128, 224)
        out = self.detection_head(out)                      # (B,   3, 128, 224)

        return {"Detection": out}





