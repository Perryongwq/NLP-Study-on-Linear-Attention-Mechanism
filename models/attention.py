import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
def causal_linear_attn(qs, ks, vs):
    """Computes not-normalized FAVOR causal attention A_{masked}V.
    Args:
    qs: query_prime tensor of the shape [B,L,H,D].
    ks: key_prime tensor of the shape [B,L,H,D].
    vs: value tensor of the shape [B,L,H,K].
    Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
    """

    result = []
    
    Si = 0

    Z = torch.cumsum(ks,dim=1)
    for i in range(qs.shape[1]):
        Si = Si + torch.einsum("bhd,bhk->bhdk", ks[:,i], vs[:,i])
        Vi = torch.einsum("bhd,bhkd->bhk",qs[:,i],Si)/(torch.einsum("bhd,bhd->bhd",qs[:,i],Z[:,i])+1e-06)
        result.append(Vi[:,None,...])

    result = torch.cat(result, dim=1) #concat in time axis
    return result

def causal_linear_attn2(qs, ks, vs):
    """Computes not-normalized FAVOR causal attention A_{masked}V.
    Args:
    qs: query_prime tensor of the shape [B,L,H,D].
    ks: key_prime tensor of the shape [B,L,H,D].
    vs: value tensor of the shape [B,L,H,K].
    Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
    """


    result = []
    
    Si = 0
    Zi = 0
    # print(qs.shape)
    n = np.maximum(ks.shape[1],qs.shape[1])
    for i in range(n):
        Si = Si + torch.einsum("bhd,bhk->bhdk", ks[:,i], vs[:,i])
        Zi = Zi + ks[:,i]
        Vi = torch.einsum("bhd,bhkd->bhk",qs[:,i],Si)/(torch.einsum("bhd,bhd->bhd",qs[:,i],Zi)+1e-06)
        result.append(Vi[:,None,...])

    result = torch.cat(result, dim=1) #concat in time axis
    return result

class LinearAttentionLayer(torch.nn.Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.
    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.
    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality for the queries source
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        d_model_keys: The input feature dimensionality for keys source
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, d_model, n_heads, d_keys=None,
                 d_values=None, d_model_keys=None, causal=False):
        super().__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        d_model_keys = d_model_keys or d_model

        self.query_projection = nn.Linear(d_model,d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,d_values * n_heads)
        self.out_projection = nn.Linear(d_model,d_model)
        self.n_heads = n_heads
        self.feat_proj = lambda x,W,is_query=False:F.elu(x,inplace=True)+1 #From "Transformers are RNNs"
        self.W=None
        # self.feat_proj = project_feat_FAVOR
        # self.W = get_orth_random_mat(d_model,rand_feats=128)
        self.causal = causal

    def forward(self, queries, keys, values,return_QK=False):

        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        Q_lin = self.feat_proj(queries,self.W,is_query=True)
        K_lin = self.feat_proj(keys,self.W)

        if self.causal:
            V_lin = causal_linear_attn2(Q_lin, K_lin, values)
            if return_QK:
                Z = 1/(torch.einsum("nlhd,nhd->nlh", Q_lin, K_lin.sum(dim=1))+1e-10)  
        else:
            Z = 1/(torch.einsum("nlhd,nhd->nlh", Q_lin, K_lin.sum(dim=1))+1e-10)

            KV = torch.einsum("nshd,nshm->nhmd", K_lin, values)

            V_lin = torch.einsum("nlhd,nhmd,nlh->nlhm", Q_lin, KV, Z)

        new_values = V_lin.contiguous().view(N, L, -1)
        QK = None if not return_QK else torch.einsum("nlhd,nthd,nlh->nlht",Q_lin,K_lin,Z).mean(dim=2)
        # QK = [Q_lin,K_lin]
        # Project the output and return
        return self.out_projection(new_values),QK
    
class FullAttentionLayer(torch.nn.Module):
    """Implement the attention layer. Namely project the inputs to multi-head
    queries, keys and values, call the attention implementation and then
    reproject the output.
    It can be thought of as a decorator (see decorator design patter) of an
    attention layer.
    Arguments
    ---------
        attention: Specific inner attention implementation that just computes a
                   weighted average of values given a similarity of queries and
                   keys.
        d_model: The input feature dimensionality for the queries source
        n_heads: The number of heads for the multi head attention
        d_keys: The dimensionality of the keys/queries
                (default: d_model/n_heads)
        d_values: The dimensionality of the values (default: d_model/n_heads)
        d_model_keys: The input feature dimensionality for keys source
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, d_model, n_heads, d_keys=None,
                 d_values=None, d_model_keys=None, causal=False):
        super().__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        d_model_keys = d_model_keys or d_model

        self.query_projection = nn.Linear(d_model,d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,d_values * n_heads)
        self.out_projection = nn.Linear(d_model,d_model)
        self.n_heads = n_heads
        self.causal = causal
    def forward(self, queries, keys, values,return_QK=False,length_mask=None):

        # Extract the dimensions into local variables

        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = torch.reshape(self.query_projection(queries),(N, L, H, -1))
        keys = torch.reshape(self.key_projection(keys),(N, S, H, -1))
        values = torch.reshape(self.value_projection(values),(N, S, H, -1))
        D = keys.shape[-1]
        sm_map = torch.einsum("nlhd,nshd->nlhs",queries,keys)
        sm_map = sm_map/math.sqrt(D)
        if self.causal:
            mask = torch.triu(torch.ones(sm_map.shape[1], sm_map.shape[1],dtype=torch.bool),diagonal=1).to(queries.device)
            # print(mask)
            mask = mask.unsqueeze(0).unsqueeze(2).repeat(N,1,H,1)
            sm_map[mask] = -torch.inf
            
            if length_mask:
                sm_map[length_mask]=-torch.inf
            
            

        sm_map = sm_map.softmax(dim=-1)
        attn = torch.einsum("nlhs,nshd->nlhd",sm_map,values)

        new_values = attn.flatten(2,3)

        QK = None if not return_QK else sm_map.mean(dim=2)
        # Project the output and return
        return self.out_projection(new_values),QK