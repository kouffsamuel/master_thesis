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

    
# -------------------------
# ---- Main Net Model ----
# -------------------------


class MViT_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # ViT network parameters
        self.D = 256  # dimension embedding
        self.p = 16   # image patch size
        self.H = 512  # image height
        self.W = 256  # image width
        self.neuron = 4 * self.D  # dimension of the MLP
        self.mha = 8  # dimension of the multi-head attention
        self.layer = 6   # number of layer in the encoder
        self.dropout = 0.1  # dropout, if needed
        self.n_antennas = 16
        self.Hp = self.H // self.p
        self.Wp = self.W // self.p

        # Nombre total de patchs par image
        self.Np = self.Hp * self.Wp

        # Embedding linéaire de chaque patch
        self.patch_embed = nn.Linear(2 *self.p ** 2, self.D)

        # Positional encodings séparés
        self.pe = PositionalEncoding(d_model=self.D)  # nn.Parameter(torch.randn(1, self.Np, self.D))

        self.encoders = nn.ModuleList([nn.ModuleList([
            Encoder(
                hidden_size=self.D,
                ff_size=self.neuron,
                multi_head=self.mha,
                dropout_rate=self.dropout
            )
            for _ in range(self.layer)
        ]) for _ in range(self.n_antennas)])

    
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.D))  # Required ?


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
            for layer in self.encoders[i]:
                xi = layer(xi, x_mask)
            
            antenna_outputs.append(xi)

        x = torch.stack(antenna_outputs, dim=1).mean(dim=1)
        
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
        out = self.detection_head(out)                       # (B,   3, 128, 224)

        return {"Detection": out}





