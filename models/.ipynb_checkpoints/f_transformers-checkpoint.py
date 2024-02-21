import torch
from fast_transformers.transformers import TransformerEncoderLayer,TransformerEncoder,TransformerDecoder,TransformerDecoderLayer
from fast_transformers.attention import FullAttention,LinearAttention,CausalLinearAttention
import math

class Encoder(torch.nn.Module):
    def __init__(self,attn_layer,n_layers=6,n_dim=256,n_heads=8,dim_feedfwd=1024,causal=False):
        super().__init__()
        
        d_keys = (n_dim//n_heads)
        # d_values = (n_dim//n_heads)
        L = attn_layer() if (attn_layer!=(CausalLinearAttention) and attn_layer!=(LinearAttention))  else attn_layer(d_keys)
        self.enc = TransformerEncoder([TransformerEncoderLayer(L,d_model=d_keys,d_ff=dim_feedfwd) for _ in range(n_layers)])
        self.proj = torch.nn.Linear(n_dim,n_dim)
        self.causal=causal
        self.n_heads=n_heads
        
    @staticmethod    
    def positionalencoding1d(length,d_model,temp=10000.0):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(temp) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe
        
    def forward(self,x,attn_mask=None,length_mask=None):
        N,L,D = x.shape
        x = x + self.positionalencoding1d(L,D).to(x.device)
        x = self.proj(x).view(N,L,self.n_heads,-1)
        x = self.enc(x,attn_mask=attn_mask,length_mask=length_mask)
        
        return x.flatten(2,3)
    
class Decoder(torch.nn.Module):
    def __init__(self,self_attn,cross_attn,n_layers=6,n_dim=256,n_heads=8,dim_feedfwd=1024,causal=False):
        super().__init__()
        d_keys = (n_dim//n_heads)
        L1 = self_attn() if self_attn!=CausalLinearAttention else self_attn(d_keys)
        L2 = cross_attn() if (cross_attn!=(CausalLinearAttention) and cross_attn!=(LinearAttention)) else cross_attn(d_keys)
        self.dec = TransformerDecoder([TransformerDecoderLayer(L1,L2,d_model=d_keys,d_ff=dim_feedfwd) for _ in range(n_layers)])
        self.projQ = torch.nn.Linear(n_dim,n_dim)
        self.projK = torch.nn.Linear(n_dim,n_dim)
        self.causal=causal
        self.n_heads=n_heads
        
    def forward(self,Q,K,Qattn_mask=None,Qlength_mask=None,Kattn_mask=None,Klength_mask=None):
        N,L,D = Q.shape
        _,T,_ = K.shape
        Q = self.projQ(Q).view(N,L,self.n_heads,-1)
        K = self.projQ(K).view(N,T,self.n_heads,-1)
        x = self.dec(Q,K,x_mask=Qattn_mask,x_length_mask=Qlength_mask,memory_mask=Kattn_mask,memory_length_mask=Klength_mask)
        
        return x.flatten(2,3)
    
class Transformer(torch.nn.Module):
    def __init__(self,enc_layer = FullAttention, dec_layer1=FullAttention, dec_layer2=FullAttention,n_enc_layers=6,n_dec_layers=6,dim=256,n_heads=8,ffn_dim=1024,dropout=0.2,return_inter = False,causal=False):
        super().__init__()
        self.enc = Encoder(enc_layer,n_layers=n_enc_layers,n_dim=dim,n_heads=n_heads,dim_feedfwd=ffn_dim,causal=causal)
        self.dec = Decoder(dec_layer1,dec_layer2,n_layers=n_dec_layers,n_dim=dim,n_heads=n_heads,dim_feedfwd=ffn_dim,causal=causal)
        self.return_inter = return_inter
        
    @staticmethod    
    def positionalencoding1d(d_model, length,temp=10000.0):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(temp) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe
    def forward(self,Q,K,Qattn_mask=None,Qlength_mask=None,Kattn_mask=None,Klength_mask=None):

        K= self.enc(K,attn_mask=Kattn_mask,length_mask=Klength_mask)
        Q= self.dec(Q,K,Qattn_mask=Qattn_mask,Qlength_mask=Qlength_mask)#,Kattn_mask=Kattn_mask,Klength_mask=Klength_mask
        
        return Q   
    