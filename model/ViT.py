import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math

"""
Vision Transformer backbone
"""

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

    def forward(self, x, x_mask=None):
        # y = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        # y = self.norm2(y + self.dropout2(self.ffn(y)))
        y = x + self.dropout1(self.mhatt(self.norm1(x), self.norm1(x), self.norm1(x), x_mask))
        y = y + self.dropout2(self.ffn(self.norm2(y)))
        return y

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(16, 16), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        #x = torch.fft.fft(x,dim=2)
        #x = torch.fft.fft(x,dim=3).real
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x