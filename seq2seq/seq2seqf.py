import torch
from torch import nn
from torch import Tensor
from transformers import Transformer
from attention import LinearAttentionLayer,FullAttentionLayer
import math
from f_transformers import Transformer
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 layers,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(enc_layer = layers[0], dec_layer1=layers[1], dec_layer2=layers[2],n_enc_layers=num_encoder_layers,\
                                       n_dec_layers=num_decoder_layers,dim=emb_size,n_heads=nhead,ffn_dim=dim_feedforward,dropout=dropout,return_inter = False,causal=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
                    emb_size, dropout=dropout)
        # print(emb_size)
    def forward(self,
                src: Tensor,
                trg: Tensor,src_mask=None,tgt_mask=None,src_pad_mask=None,tgt_pad_mask=None):
        

        src = self.positional_encoding(self.src_tok_emb(src))
        trg = self.positional_encoding(self.tgt_tok_emb(trg))
        src = src.transpose(1,0)
        trg = trg.transpose(1,0)

        # print(src.shape)
    
        outs = self.transformer(trg,src,Qattn_mask=tgt_mask,Qlength_mask=tgt_pad_mask,Kattn_mask=src_mask,Klength_mask=src_pad_mask)

        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)