import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput


def _get_clone(module):
    return copy.deepcopy(module)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def lengths_to_mask(lengths: List[int], device: torch.device, max_len: int = None) -> Tensor:
    """Convert lengths to boolean mask tensor."""
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


@dataclass
class MldVaeEncoderOutput(ModelOutput):
    """
    Output type of MldVae encoder.
    
    Args:
        latent: Latent representation [latent_size, batch_size, latent_dim]
        dist_params: Distribution parameters (mu, logvar) [latent_size, batch_size, latent_dim]
    """
    latent: Optional[torch.FloatTensor] = None
    dist_params: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None


@dataclass
class MldVaeDecoderOutput(ModelOutput):
    """
    Output type of MldVae decoder.
    
    Args:
        features: Reconstructed motion features [seq_len, batch_size, dims]
    """
    features: Optional[torch.FloatTensor] = None


@dataclass
class MldVaeOutput(ModelOutput):
    """
    Output type of MldVae.
    
    Args:
        features: Reconstructed motion features [seq_len, batch_size, dims]
        latent: Latent representation [latent_size, batch_size, latent_dim]
        dist_params: Distribution parameters (mu, logvar) [latent_size, batch_size, latent_dim]
    """
    features: Optional[torch.FloatTensor] = None
    latent: Optional[torch.FloatTensor] = None
    dist_params: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None


class PositionEmbeddingSine1D(nn.Module):

    def __init__(self, d_model, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            pos = self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            pos = self.pe[:x.shape[0], :]
        return pos


class PositionEmbeddingLearned1D(nn.Module):

    def __init__(self, d_model, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        # self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.zeros(max_len, 1, d_model))
        # self.pe = pe.unsqueeze(0).transpose(0, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            pos = self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return x
        # return self.dropout(x)


class SkipTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = encoder_layer.d_model

        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        x = src

        xs = []
        for module in self.input_blocks:
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            xs.append(x)

        x = self.middle_block(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            x = self.norm(x)
        return x


class SkipTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.d_model = decoder_layer.d_model
        
        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(decoder_layer, num_block)
        self.middle_block = _get_clone(decoder_layer)
        self.output_blocks = _get_clones(decoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        x = tgt

        xs = []
        for module in self.input_blocks:
            x = module(x, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            xs.append(x)

        x = self.middle_block(x, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            x = self.norm(x)

        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
                     
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class MldVaeConfig(PretrainedConfig):
    """
    Configuration class for MldVae model.
    
    Args:
        nfeats (int): Number of input features (e.g., 263 for HumanML3D)
        latent_dim (List[int]): Latent dimension [num_tokens, dim] (default: [1, 256])
        ff_size (int): Feed-forward network hidden size (default: 1024)
        num_layers (int): Number of transformer layers (default: 9)
        num_heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout probability (default: 0.1)
        arch (str): Architecture type, 'all_encoder' or 'encoder_decoder' (default: 'encoder_decoder')
        normalize_before (bool): Whether to normalize before attention (default: False)
        activation (str): Activation function type (default: 'gelu')
        position_embedding (str): Position embedding type (default: 'learned')
        datatype (str): Dataset type, 'humanml' or 'motionx' (default: 'humanml')
        mlp_dist (bool): Whether to use linear to expand mean and std rather expand token nums (default: False)
        skip_connect (bool): Whether to use skip connections (default: True)
    """
    
    model_type = "mld_vae"
    
    def __init__(
        self,
        nfeats: int = 263,
        latent_dim: List[int] = [1, 256],
        ff_size: int = 1024,
        num_layers: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1,
        arch: str = "encoder_decoder",
        normalize_before: bool = False,
        activation: str = "gelu",
        position_embedding: str = "learned",
        datatype: str = "humanml",
        mlp_dist: bool = False,
        skip_connect: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if latent_dim is None:
            latent_dim = [1, 256]
            
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.arch = arch
        self.normalize_before = normalize_before
        self.activation = activation
        self.position_embedding = position_embedding
        self.datatype = datatype
        self.mlp_dist = mlp_dist
        self.skip_connect = skip_connect
        
        # Dataset-specific normalization parameters
        if 'motionx' in datatype.lower():
            self.mean_std_inv = 0.7281
            self.mean_mean = 0.0636
        else:  # humanml3d
            self.mean_std_inv = 0.8457
            self.mean_mean = -0.1379


class MldVaePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and pretrained model loading.
    """
    config_class = MldVaeConfig
    base_model_prefix = "mld_vae"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerEncoderLayer", "TransformerDecoderLayer"]
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=0.02)


class MldVaeModel(MldVaePreTrainedModel):
    """
    Motion Latent Diffusion VAE Model for motion generation.
    
    This model encodes motion sequences into a latent space and decodes them back.
    """
    
    def __init__(self, config: MldVaeConfig):
        super().__init__(config)
        self.config = config
        
        self.latent_size = config.latent_dim[0]
        self.latent_dim = config.latent_dim[-1]
        
        # Position encoders
        if config.position_embedding == "learned":
            self.query_pos_encoder = PositionEmbeddingLearned1D(self.latent_dim)
            self.query_pos_decoder = PositionEmbeddingLearned1D(self.latent_dim)
        elif config.position_embedding == "sine":
            self.query_pos_encoder = PositionEmbeddingSine1D(self.latent_dim)
            self.query_pos_decoder = PositionEmbeddingSine1D(self.latent_dim)
        else:
            raise ValueError(f"Not supported position embedding: {config.position_embedding}")
        
        # Encoder
        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            config.num_heads,
            config.ff_size,
            config.dropout,
            config.activation,
            config.normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, config.num_layers, encoder_norm)
        
        # Decoder
        if config.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, config.num_layers, decoder_norm)
        elif config.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                config.num_heads,
                config.ff_size,
                config.dropout,
                config.activation,
                config.normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, config.num_layers, decoder_norm)
        else:
            raise ValueError(f"Not supported architecture: {config.arch}")
        
        if config.mlp_dist:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size, self.latent_dim)
            )
            self.dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)
        else:
            self.global_motion_token = nn.Parameter(
                torch.randn(self.latent_size * 2, self.latent_dim)
            )
        
        self.skel_embedding = nn.Linear(config.nfeats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, config.nfeats)
        
        # Initialize weights
        self.post_init()
    
    def encode_dist(self, features: Tensor, lengths: Optional[List[int]] = None):
        """Encode features to distribution parameters"""
        if lengths is None:
            lengths = [len(feature) for feature in features]
        
        device = features.device
        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device, max_len=nframes)
        
        # Embed skeleton features
        x = self.skel_embedding(features)
        x = x.permute(1, 0, 2)  # [nframes, bs, latent_dim]
        
        # Global motion tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
        
        # Create augmented mask
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=bool, device=device)
        aug_mask = torch.cat((dist_masks, mask), 1)
        
        # Concatenate tokens and features
        xseq = torch.cat((dist, x), 0)
        
        # Encode with position encoding
        xseq = self.query_pos_encoder(xseq)
        dist = self.encoder(xseq, src_key_padding_mask=~aug_mask)[:dist.shape[0]]
        
        return dist
    
    def encode_dist2z(self, dist):
        """Convert distribution to latent code"""
        if self.config.mlp_dist:
            tokens_dist = self.dist_layer(dist)
            mu = tokens_dist[:, :, :self.latent_dim]
            logvar = tokens_dist[:, :, self.latent_dim:]
        else:
            mu = dist[0:self.latent_size, ...] # 1, bs, 256
            logvar = dist[self.latent_size:, ...] # 1, bs, 256
        
        # Reparameterization trick
        std = logvar.exp().pow(0.5)
        dist_params = torch.distributions.Normal(mu, std)
        latent = dist_params.rsample() # [1, bs, 256]
        
        return latent, (mu, logvar)
    
    def encode(
        self, 
        features: Tensor, 
        lengths: Optional[List[int]] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, MldVaeEncoderOutput]:
        """Encode features to latent representation"""
        dist = self.encode_dist(features, lengths)
        latent, dist_params = self.encode_dist2z(dist)
        output = MldVaeEncoderOutput(latent=latent, dist_params=dist_params)
        return output if return_dict else output.to_tuple()
    
    def decode(
        self, 
        z: Tensor, 
        lengths: List[int],
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MldVaeDecoderOutput]:
        """Decode latent representation to features"""
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape
        
        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        queries = self.query_pos_decoder(queries)
        
        output = self.decoder(
            tgt=queries,
            memory=z,
            tgt_key_padding_mask=~mask,
        ).squeeze(0)
        
        output = self.final_layer(output)
        output[~mask.T] = 0
        
        feats = output.permute(1, 0, 2)
        output = MldVaeDecoderOutput(features=feats)
        return output if return_dict else output.to_tuple()
    
    def forward(
        self,
        features: Tensor,
        lengths: Optional[List[int]] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, MldVaeOutput]:
        """
        Forward pass of MldVae.
        
        Args:
            features: Input motion features [batch_size, seq_len, nfeats]
            lengths: Length of each sequence in the batch [batch_size]
            return_dict: Whether to return ModelOutput
            
        Returns:
            MldVaeOutput or tuple of (reconstructed_features, latent, distribution)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        lengths = lengths if lengths is not None else [len(feature) for feature in features]
        
        # Encode
        latent, dist_params = self.encode(features, lengths, return_dict=False)
        
        # Decode
        reconstructed, = self.decode(latent, lengths, return_dict=False)
        
        if not return_dict:
            return (reconstructed, latent, dist_params)
        
        output = MldVaeOutput(
            features=reconstructed,
            latent=latent,
            dist_params=dist_params,
        )
        return output if return_dict else output.to_tuple()
    
    @classmethod
    def from_pretrained_ckpt(cls, ckpt_path: str, config: Union[str, MldVaeConfig] = None, **kwargs):
        """
        Load model from original checkpoint file.
        
        Args:
            config: Model configuration or path to config file, if None, use default config
            ckpt_path: Path to checkpoint file (.ckpt)
            
        Returns:
            MldVaeModel instance with loaded weights
        """
        # Load config
        if config is None:
            config = MldVaeConfig() # default config
        elif isinstance(config, str):
            config = MldVaeConfig.from_pretrained(config)
        
        # Initialize model
        print(f"Loading MldVae model with config: {config}")
        model = cls(config)
        
        # Load checkpoint
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Extract VAE weights (remove 'vae.' prefix)
        vae_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('vae.'):
                new_key = key.replace('vae.', '')
                vae_state_dict[new_key] = value
        
        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(vae_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        return model

