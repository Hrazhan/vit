import torch
import torch.nn as nn


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
                 ffn_dropout:float=0.1,
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
        self.transformer_layers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_units, dropout=attn_dropout)
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
    # patchify = PatchEmbedding(patch_size=14, in_channels=3, embed_dim=384)
    # x = torch.rand(1, 3, 28, 28)
    # x = patchify(x)
    # print(x.shape)
    
    model = ViT()
    
    x = torch.randn(1, 3, 224, 224)
    img = model(x)    
    print(img.shape)

    


