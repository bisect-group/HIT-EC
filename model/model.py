import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import math
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class TransformerEmbedding(nn.Module): 

    def __init__(self, vocab_size, embed_size, dropout):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=embed_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)
    
class MultiHeadedAttention(nn.Module):

    def __init__(self, h=8, d_model=256, dropout=0.1, sdpa=False):
        super().__init__()
        assert d_model % h == 0
        
        self.sdpa = sdpa
        self.d_k = d_model // h
        self.h = h
        self.dim = d_model
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model*h)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def transpose_for_scores_value(self, x):
        new_x_shape = x.size()[:-1] + (self.h, self.dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def summed_value(self, x, rows):
        """Value projection of the first `rows` summed columns, summing every h adjacent columns.

        The attention output is viewed as (length, batch, dim, h) and summed over h, which sums h adjacent
        value columns of each head. Summing the weights first gives the same output from a dim-wide value.
        """
        weight = self.value.weight.view(self.dim, self.h, -1)[:rows].sum(1)
        bias = self.value.bias.view(self.dim, self.h)[:rows].sum(1)
        return F.linear(x, weight, bias)

    def forward(self, x, mask=None):
        x = self.norm(x)
        batch_size = x.size(0)
        query = self.transpose_for_scores(self.query(x))
        key = self.transpose_for_scores(self.key(x))
        value = self.summed_value(x, self.dim).view(batch_size, -1, self.h, self.dim // self.h).permute(0, 2, 1, 3)

        if self.sdpa:
            # Default scale 1/sqrt(d_k), no mask, no attention dropout.
            out = F.scaled_dot_product_attention(query, key, value)
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
            out = torch.matmul(self.softmax(scores), value)
        return out.reshape(batch_size, -1, self.dim)

    def forward_cls(self, x):
        """Output row 0 only. It reads only the first h query rows of head 0, so only that head is computed."""
        x = self.norm(x)
        batch_size = x.size(0)
        query = F.linear(x[:, : self.h], self.query.weight[: self.d_k], self.query.bias[: self.d_k])
        key = F.linear(x, self.key.weight[: self.d_k], self.key.bias[: self.d_k])
        value = self.summed_value(x, self.dim // self.h)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        return torch.matmul(self.softmax(scores), value).reshape(batch_size, self.dim)


class InterAttention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    
class InterMultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = InterAttention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        self.save_attn(attn)
        attn.register_hook(self.save_attn_gradients)

        return self.output_linear(x)
    
    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
class SublayerConnection(nn.Module):
    
    def __init__(self, size, feed_forward_hidden, dropout):
        super(SublayerConnection, self).__init__()
        self.feed_forward = PositionwiseFeedForward(d_model=size, d_ff=feed_forward_hidden, dropout=dropout)
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        return x1 + self.dropout(self.feed_forward(self.norm(x2)))
    
class TransformerBlock(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, sdpa=False):
        super().__init__()
        self.attention = MultiHeadedAttention(attn_heads, hidden, dropout, sdpa)
        self.inter_attention = InterMultiHeadedAttention(attn_heads, hidden, dropout)
        self.output_sublayer = SublayerConnection(hidden, feed_forward_hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def infer_cls(self, x):
        """Row 0 of the infer output."""
        x_infer = x[:, 0] + self.attention.forward_cls(x)
        return self.dropout(self.output_sublayer(x_infer, x_infer))

    def forward(self, x, mask, mode):
        if mode == 'infer' : 
            x1_infer = self.attention(x, mask)
            x_infer = x + x1_infer
            x_infer = self.output_sublayer(x_infer, x_infer)
            x_infer = self.dropout(x_infer)
            return x_infer
        
        elif mode == 'inter' : 
            x1_inter = self.inter_attention(x, x, x, mask)
            x_inter = x + x1_inter
            x_inter = self.output_sublayer(x_inter, x_inter)
            x_inter = self.dropout(x_inter)
            return x_inter
    
class Transformer(nn.Module) :
    
    def __init__(self, vocab_size, dimension, attn_heads, output_dims, dropout, beta, sdpa=False) :
        super().__init__()
        self.embeddings = TransformerEmbedding(vocab_size, dimension, dropout)
        self.enc_1 = TransformerBlock(dimension, attn_heads, dimension*4, dropout, sdpa)
        self.enc_2 = TransformerBlock(dimension, attn_heads, dimension*4, dropout, sdpa)
        self.enc_3 = TransformerBlock(dimension, attn_heads, dimension*4, dropout, sdpa)
        self.enc_4 = TransformerBlock(dimension, attn_heads, dimension*4, dropout, sdpa)
        self.linear1 = nn.Linear(dimension, output_dims[0])
        self.linear2 = nn.Linear(dimension, output_dims[1])
        self.linear3 = nn.Linear(dimension, output_dims[2])
        self.linear4 = nn.Linear(dimension, output_dims[3])
        self.linear5 = nn.Linear(dimension, np.sum(output_dims))
        self.beta = beta
        
    def forward(self, x, mode):
        local = []
        mask = None
        
        if mode == 'infer': 
            emb = self.embeddings(x)
            x = self.enc_1(emb, mask, mode='infer')
            local.append(self.linear1(x[:,0,:]))
            x = self.enc_2(x + emb, mask, mode='infer')
            local.append(self.linear2(x[:,0,:]))
            x = self.enc_3(x + emb, mask, mode='infer')
            local.append(self.linear3(x[:,0,:]))
            x = self.enc_4.infer_cls(x + emb)
            local.append(self.linear4(x))
            global_p = self.linear5(x)
            final = self.beta * global_p + (1 - self.beta) * torch.cat(local, dim=1)
            return final
        
        if mode == 'inter': 
            emb = self.embeddings(x)
            x = self.enc_1(emb, mask, mode='inter')
            local.append(self.linear1(x[:,0,:]))
            x = self.enc_2(x + emb, mask, mode='inter')
            local.append(self.linear2(x[:,0,:]))
            x = self.enc_3(x + emb, mask, mode='inter')
            local.append(self.linear3(x[:,0,:]))
            x = self.enc_4(x + emb, mask, mode='inter')
            local.append(self.linear4(x[:,0,:]))
            global_p = self.linear5(x[:,0,:])
            final = self.beta * global_p + (1 - self.beta) * torch.cat(local, dim=1)
            return final
        
class Model(pl.LightningModule):

    def __init__(self, config, dimension=1024, vocab_size=23):
        
        super().__init__()
        self.model = Transformer(vocab_size, dimension, config["ah"], config["output_dims"], config["dr"], config["beta"])
    
    def forward(self, x, mode='infer'):
        return self.model(x, mode)

    def configure_optimizers(self):
        pass
            

    def training_step(self, batch, batch_idx):
        pass
    
    def validation_step(self, batch, batch_idx):
        pass
    
    def test_step(self, batch, batch_idx):
        pass