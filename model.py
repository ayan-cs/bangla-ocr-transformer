import numpy as np
import math, os, torch, torchvision
from torch import nn
import torchvision.transforms as T
from torchvision.models import resnet18

from utils import load_function

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BanglaOCR(nn.Module):
    def __init__(self, vocab_len : int, max_text_length : int, hidden_dim : int, nheads : int, num_decoder_layers : int):
        super().__init__()

        # create ResNet-18 encoder; You can use other standard CNN architectures such as ResNet-50 as well
        self.encoder = resnet18()
        del self.encoder.fc

        # create conversion layer
        self.conv = nn.Conv2d(512, hidden_dim, 1)

        # create a default PyTorch transformer
        self.tf_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, nheads), num_layers = num_decoder_layers
        )

        # prediction heads with length of vocab
        # DETR used basic 3 layer MLP for output
        self.vocab = nn.Linear(hidden_dim, vocab_len)

        # output positional encodings (object queries)
        self.decoder = nn.Embedding(vocab_len, hidden_dim)
        self.query_pos = PositionalEncoding(d_model = hidden_dim, dropout = .2, max_len = max_text_length)

        # Sine positional encodings, spatial positional encoding can be used.
        # Detr baseline uses sine positional encoding.
        self.row_embed = torch.zeros((1000, hidden_dim // 2), requires_grad=False)
        self.col_embed = torch.zeros((1000, hidden_dim // 2), requires_grad=False)
        div_term = torch.exp(-math.log(10000) * torch.arange(0, hidden_dim // 2, 2) / hidden_dim)
        pe = torch.arange(0, 1000).unsqueeze(1)
        self.row_embed[:, 0::2] = torch.sin(pe * div_term)
        self.row_embed[:, 1::2] = torch.cos(pe * div_term)
        self.col_embed[:, 0::2] = torch.sin(pe * div_term)
        self.col_embed[:, 1::2] = torch.cos(pe * div_term)
        self.row_embed = nn.Parameter(self.row_embed)
        self.col_embed = nn.Parameter(self.col_embed)
        self.trg_mask = None
  
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def get_feature(self,x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)   
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        return x


    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, inputs, trg):
        # propagate inputs through ResNet-101 up to avg-pool layer
        x = self.get_feature(inputs)

        # convert from backbone output dimension to Transformer Decoder input dimension feature planes
        h = self.conv(x)

        # construct positional encodings
        bs,_,H, W = h.shape
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # generating subsequent mask for target
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(trg.shape[1]).to(trg.device)

        # Padding mask
        trg_pad_mask = self.make_len_mask(trg)

        # Getting postional encoding for target
        trg = self.decoder(trg)
        trg = self.query_pos(trg)
        output = self.tf_decoder(memory = pos + 0.1 * h.flatten(2).permute(2, 0, 1), tgt = trg.permute(1,0,2), tgt_mask=self.trg_mask, tgt_key_padding_mask=trg_pad_mask.permute(1,0))
        return self.vocab(output.transpose(0,1))