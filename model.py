import torch
import torch.nn as nn
from math import sqrt

def scaled_dot_product_attention(query, key, value, dropout=None):
    """
    Scaled Dot-Product Attention
    """
    d_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(d_k)
    attention_weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    return torch.bmm(attention_weights, value)

class AttentionHead(nn.Module):
    """
    Attention Head
    """
    def __init__(self, embed_dim, head_dim, dropout=0.0):
        super().__init__()
        self.proj_q = nn.Linear(embed_dim, head_dim)
        self.proj_k = nn.Linear(embed_dim, head_dim)
        self.proj_v = nn.Linear(embed_dim, head_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_output = scaled_dot_product_attention(self.proj_q(x), self.proj_k(x), self.proj_v(x), dropout=self.dropout)
        return attn_output
    
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        head_dim = hidden_dim // num_heads

        self.attention_heads = nn.ModuleList([
            AttentionHead(hidden_dim, head_dim, dropout=dropout) for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        x = torch.cat([attn_head(x) for attn_head in self.attention_heads], dim=2)
        x = self.out_proj(x)
        return x

class MLP(nn.Module):
    """
    Position-wise FEED-FORWARD Network
    """
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x
    
class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    """
    def __init__(self, embed_dim, hidden_dim, num_heads=12, attn_dropout=0.0, ff_dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, attn_dropout)
        self.feed_forward = MLP(embed_dim, hidden_dim, dropout=ff_dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        hidden_state = self.layer_norm(x)
        x = self.attention(hidden_state) + x
        x = self.layer_norm(x)
        x = self.feed_forward(x) + x
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        return x
    

class ViT(nn.Module):
    def __init__(self,
                 image_size:int=224, 
                 patch_size:int=16, 
                 in_channels:int=3, 
                 embed_dim:int=768,
                 num_transformer_layer:int=12,
                 num_heads:int=12,
                 hidden_units:int=3072,
                 attn_dropout:float=0.0,
                 ff_dropout:float=0.1,
                 embed_dropout:float=0.1, 
                 num_classes:int=10):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        self.num_patches = int(image_size * image_size) // patch_size ** 2

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim),
                                               requires_grad=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(embed_dropout)
        # for now I will use the PyTorch built-in modules. but I will implement most of it from scratch
        # self.transformer_layers = nn.Sequential(*[
        #     nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_units, dropout=attn_dropout)
        #     for _ in range(num_transformer_layer)
        # ])
        self.transformer_layers = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim=embed_dim, hidden_dim=hidden_units, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            for _ in range(num_transformer_layer)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        # create patch embeddings
        x = self.patch_embed(x)
        # prepend class token to the patch embeddings
        x = torch.cat((cls_token, x), dim=1)
        # add position embedding to the patch embeddings
        x = x + self.position_embedding
        x = self.embedding_dropout(x)
        # run patch, position and class token embeddings through transformer encoder
        x = self.transformer_layers(x)
        # project back to the embedding space
        x = self.classifier(x[:, 0])
        # print(x.shape)
        return x



if __name__ == '__main__':

    # Testing self attention
    # emb = nn.Embedding(5, 10)
    # input_emb = emb(torch.tensor([[0, 1, 2, 3, 4]]))
    # query = key = value = input_emb
    # attn = scaled_dot_product_attention(query, key, value)
    # print(attn, attn.shape)

    # attn_head = AttentionHead(embed_dim=768, head_dim=64)
    # x = torch.randn(1, 5, 768)
    # print(attn_head(x).shape)

    # multihead_attn = MultiHeadAttention(hidden_dim=768, num_heads=12)
    # print(multihead_attn(x).shape)

    # mlp = MLP(embed_dim=768, hidden_dim=3072)
    # print(mlp(x).shape)

    # transoformer_encoder = TransformerEncoderLayer(embed_dim=768, hidden_dim=3072, num_heads=12, attn_dropout=0.0, ff_dropout=0.1)
    # print(transoformer_encoder(x).shape)

    # patchify = PatchEmbedding(patch_size=14, in_channels=3, embed_dim=384)
    # x = torch.rand(1, 3, 28, 28)
    # x = patchify(x)
    # print(x.shape)
    

    model = ViT()
    
    x = torch.randn(1, 3, 224, 224)
    img = model(x)    
    print(img.shape)


