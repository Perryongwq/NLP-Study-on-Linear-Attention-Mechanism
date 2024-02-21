from .attention import LinearAttentionLayer,FullAttentionLayer
import torch
import math
from torch import nn

class Transformer(torch.nn.Module):
    def __init__(self,dec_layer=FullAttentionLayer,n_enc_layers=6,n_dec_layers=6,dim=256,n_heads=8,ffn_dim=1024,dropout=0.2,return_inter = False,causal=False):
        super().__init__()
        self.enc = Transformer_Encoder(n_enc_layers=n_enc_layers,dim=dim,n_heads=n_heads,ffn_dim=ffn_dim,dropout=dropout)
        self.dec = Transformer_Decoder(dec_layer,n_dec_layers=n_dec_layers,dim=dim,n_heads=n_heads,ffn_dim=ffn_dim,dropout=dropout,return_inter=return_inter,causal=causal)
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
    def forward(self,Q,K,return_QK=False,src_mask=None,tgt_mask=None,src_pad_mask=None,tgt_pad_mask=None):
        Q = Q+self.positionalencoding1d(Q.shape[-1],Q.shape[1]).to(Q.device)
        K = K+self.positionalencoding1d(K.shape[-1],K.shape[1]).to(Q.device)
        # print(K.shape,Q.shape)
        K,qk_enc = self.enc(K,return_QK=return_QK,key_padding_mask = src_pad_mask,attn_mask=src_mask)
        x,qk  = self.dec(Q,K,return_QK=return_QK,key_padding_mask = tgt_pad_mask,attn_mask=tgt_mask)
        
        return x,(qk_enc,qk)
    
class Transformer_Encoder(torch.nn.Module):
    
    def __init__(self,n_enc_layers=6,dim=256,n_heads=8,ffn_dim=1024,dropout=0.2,return_inter = False,causal=False):
        super().__init__()
        self.return_intermediate = return_inter
        # self.MHA = [FullAttentionLayer(d_model=dim, n_heads=n_heads , d_keys=dim,d_values=dim,d_model_keys=dim,causal=causal) for _ in range(n_enc_layers)] #18GB
        self.MHA = nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads,batch_first=True) for _ in range(n_enc_layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_enc_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_enc_layers)])
        self.do1 = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_enc_layers)])
        self.FFN = nn.ModuleList([nn.Sequential(*[   nn.Linear(dim,ffn_dim),\
                                       nn.ReLU(inplace=True),\
                                       nn.Dropout(dropout),\
                                       nn.Linear(ffn_dim,dim),\
                                       nn.ReLU(inplace=True)]) for _ in range(n_enc_layers)])
        # self.pos_enc = keras_nlp.layers.SinePositionEncoding()
        self.causal = causal
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
    def forward(self,x,attn_mask=None,key_padding_mask =None,return_QK=False,):
        # positional_encoding = self.positionalencoding1d(x.shape[-1],x.shape[1])
        # x = x+positional_encoding.to(x.device)
        # print(x.shape)
        returns = []
        for mha,do,norm,ffn,norm2 in zip(self.MHA,self.do1,self.norm1,self.FFN,self.norm2):
            
            # attn1,attn_map = mha(x,x,x,return_QK)

            attn1,attn_map = mha(x,x,x,key_padding_mask =key_padding_mask,attn_mask=attn_mask,need_weights=return_QK)

            attn1 = do(attn1)
            x = norm(attn1+x)
            f = ffn(x)
            x = norm2(f+x)
            if self.return_intermediate: 
                returns.append(x)
            
        if self.return_intermediate:
            return returns,attn_map
        else:
            return x,attn_map
        
class Transformer_Decoder(torch.nn.Module):
    
    def __init__(self,dec_layer,n_dec_layers=6,dim=256,n_heads=8,ffn_dim=1024,dropout=0.2,return_inter = False,causal=False):
        super().__init__()
        # self.MHA1 = [FullAttentionLayer(d_model=dim, n_heads=n_heads , d_keys=dim,d_values=dim,d_model_keys=dim,causal=causal) for _ in range(n_dec_layers)]
        # self.MHA2 = [FullAttentionLayer(d_model=dim, n_heads=n_heads , d_keys=dim,d_values=dim,d_model_keys=dim) for _ in range(n_dec_layers)]
        # self.MHA1 = nn.ModuleList([dec_layer(d_model=dim, n_heads=n_heads,causal=causal) for _ in range(n_dec_layers)])
        self.MHA1 = nn.ModuleList([torch.nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True) for _ in range(n_dec_layers)])
        self.MHA2 = nn.ModuleList([dec_layer(d_model=dim, n_heads=n_heads) for _ in range(n_dec_layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_dec_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_dec_layers)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_dec_layers)])
        self.do1 = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_dec_layers)])
        self.do2 = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_dec_layers)])
        self.FFN = nn.ModuleList([nn.Sequential(*[ nn.Linear(dim,ffn_dim),\
                                                   nn.ReLU(inplace=True),\
                                                   nn.Dropout(dropout),\
                                                   nn.Linear(ffn_dim,dim),\
                                                   nn.ReLU(inplace=True)]) for _ in range(n_dec_layers)])
        # self.pos_enc = keras_nlp.layers.SinePositionEncoding()
        self.return_intermediate = return_inter
        self.causal=causal
        
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
    def forward(self,Q,K,return_QK=False,attn_mask=None,key_padding_mask =None):
        # positional_encoding = self.positionalencoding1d(Q.shape[-1],Q.shape[1])
        # x1 = Q+positional_encoding.to(Q.device)
        x1=Q
        if self.return_intermediate:
            feat_list = []
        for mha1,do1,norm1,mha2,do2,norm2,ffn,norm3 \
            in zip(self.MHA1,self.do1,self.norm1,self.MHA2,self.do2,self.norm2,self.FFN,self.norm3):
            

            x,_ = mha1(x1,x1,x1,key_padding_mask =key_padding_mask ,attn_mask=attn_mask)
            # x,_ = mha1(x1,x1,x1)
            x = do1(x)
            x2 = norm1(x+x1)
            
            x,qk2 = mha2(x2,K,K,return_QK)
            x = do2(x)
            x3 = norm2(x+x2)
            
            x = ffn(x3)
            x1 = norm3(x+x3)

            if self.return_intermediate:
                feat_list.append(x1)

        if self.return_intermediate:
            return feat_list,qk2
        else:
            return x1,qk2