"""
Author: Stankevich Andrey, MIPT <stankevich.as@phystech.edu>
"""


import numpy as np
import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantizer


class VQVAE(nn.Module):

	def __init__(
		self,
		device,
		n_hid,
		embedding_dim,
		n_embed
	):

		super().__init__()

		self.device        = device
		self.n_hid         = n_hid 
		self.embedding_dim = embedding_dim
		self.n_embed       = n_embed

		self.encoder       = Encoder(input_channels=3, n_hid=self.n_hid, n_out=4*self.n_hid, n_groups=2)
		self.quantize_conv = nn.Conv2d(4*self.n_hid, self.embedding_dim, 1) 
		self.quantizer     = VectorQuantizer(self.embedding_dim, self.n_embed)
		self.decoder       = Decoder(input_channels=self.embedding_dim, n_hid=self.n_hid, n_groups=2)
	
	def encode(self, x):

		enc   = self.encoder(x)
		quant = self.quantize_conv(enc).permute(0, 2, 3, 1)
		quant, diff, id_ = self.quantizer(quant)
		quant = quant.permute(0, 3, 1, 2)
		diff  = diff.unsqueeze(0) 

		return quant, diff, id_

	def decode(self, quant):
		return torch.sigmoid(self.decoder(quant))

	def decode_code(self, code):

		quant = self.quantizer.embed_code(code)
		quant = quant.permute(0, 3, 1, 2)
		dec   = self.decode(quant)

		return dec

	def forward(self, x):
		quant, diff, _ = self.encode(x)
		return self.decode(quant), diff