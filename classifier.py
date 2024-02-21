from models.f_transformers import Encoder
from fast_transformers.attention import FullAttention,LinearAttention,CausalLinearAttention
import torch
class Amazon_Classifier(torch.nn.Module):
    def __init__(self,layer=FullAttention,dim=256,n_layers=6,n_heads=8,dim_feedfwd=1024,causal=False):
        super().__init__()
        self.enc = Encoder(layer,n_layers=n_layers,n_dim=dim,n_heads=n_heads,dim_feedfwd=dim_feedfwd,causal=causal)
        self.emb = torch.nn.Embedding(50002,dim)
        self.class_head = torch.nn.Linear(dim,2)
    def forward(self,x):
        x = self.emb(x)
        x = self.enc(x)
        x =self.class_head(x.max(1).values)
        
        return x