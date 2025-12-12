# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed


class Mlp(nn.Sequential):
  def __init__(
      self,
      embed_dim: int,
      mlp_dim: int,
      dropout: float = 0.0,
      input_dim: Optional[int] = None,
      output_dim: Optional[int] = None
  ):
    if input_dim is None:
      input_dim = embed_dim
    if output_dim is None:
      output_dim = embed_dim

    super().__init__(
        nn.Linear(input_dim, mlp_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_dim, output_dim),
        nn.Dropout(dropout),
    )


class LearnableFourierFeaturesEncoding(nn.Module):
  def __init__(
      self,
      pos_dim: int,
      embed_dim: int,
      mlp_dim: int,
      gamma=1.0
  ):
    super(LearnableFourierFeaturesEncoding, self).__init__()
    assert embed_dim % 2 == 0, 'number of fourier feature dimensions must be divisible by 2.'
    half_embed_dim = int(embed_dim / 2)
    self.w = nn.Parameter(
        torch.randn([half_embed_dim, pos_dim]) * (gamma ** 2)
    )
    self.mlp = Mlp(
        embed_dim=embed_dim,
        mlp_dim=mlp_dim,
        dropout=0.0
    )
    self.register_buffer(
        "scale",
        torch.tensor(1.0 / np.sqrt(embed_dim))
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [b, n, pos_dim]
    # w: [half_embed_dim, pos_dim]
    # w.T: [pos_dim, half_embed_dim]
    x = x @ self.w.T  # [b, n, half_embed_dim]
    f = self.scale * torch.cat(
        [torch.cos(x), torch.sin(x)],
        dim=-1
    )  # [b, n, embed_dim]
    y = self.mlp(f)  # [b, n, embed_dim]
    return y


def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################


class Attention(nn.Module):
  def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False):
    super().__init__()
    assert dim % num_heads == 0, 'dim should be divisible by num_heads'
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = head_dim ** -0.5
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C)  # require pytorch 2.0
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class CrossAttention(nn.Module):
  def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., qk_norm: bool = True):
    super().__init__()
    assert dim % num_heads == 0, 'dim should be divisible by num_heads'
    self.num_heads = num_heads
    head_dim = dim // num_heads
    self.scale = head_dim ** -0.5

    self.q = nn.Linear(dim, dim, bias=qkv_bias)
    self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)

    if not qk_norm:
      self.q_norm = nn.Identity()
      self.k_norm = nn.Identity()
    else:
      self.q_norm = nn.RMSNorm(head_dim * num_heads, eps=1e-5)
      self.k_norm = nn.RMSNorm(head_dim * num_heads, eps=1e-5)

  def forward(self, q, c):
    B, N, C = q.shape
    _, M, _ = c.shape  # Context length M (different from N)

    q = self.q(q)  # q: B, N, C
    kv = self.kv(c).view(B, M, 2, C).permute(2, 0, 1, 3)  # Flatten heads
    k, v = kv.unbind(0)  # k, v: B, M, C
    # Apply query-key normalization
    q = self.q_norm(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    k = self.k_norm(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    v = v.reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    # Attention
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # Requires PyTorch 2.0+
    x = x.transpose(1, 2).reshape(B, N, C)  # Merge head and feature dims
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class SelfAttention(CrossAttention):
  def __init__(self, *args, **kwargs):
    super(SelfAttention, self).__init__(*args, **kwargs)

  def forward(self, x):
    return super(SelfAttention, self).forward(q=x, c=x)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """

  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(frequency_embedding_size, hidden_size, bias=True),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size, bias=True),
    )
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t, use_fp16=False):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    if use_fp16:
      t_freq = t_freq.to(dtype=torch.float16)
    t_emb = self.mlp(t_freq)
    return t_emb


#################################################################################
#                                 Core Latte Model                                #
#################################################################################

class TransformerBlock(nn.Module):
  """
  A Latte tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
  """

  def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
    super().__init__()
    self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
    self.xattn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
    self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    mlp_hidden_dim = int(hidden_size * mlp_ratio)
    def approx_gelu(): return nn.GELU(approximate="tanh")
    from timm.models.vision_transformer import Mlp
    self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
    self.adaLN_modulation = nn.Sequential(
        nn.SiLU(),
        nn.Linear(hidden_size, 7 * hidden_size, bias=True)
    )

  def forward(self, x, t, c):
    shift_msa, scale_msa, gate_msa, gate_mxa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(7, dim=1)
    x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
    x = x + gate_mxa.unsqueeze(1) * self.xattn(x, c)
    x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
    return x


class FinalLayer(nn.Module):
  """
  The final layer of Latte.
  """

  def __init__(self, hidden_size, patch_size, out_channels):
    super().__init__()
    self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
    self.adaLN_modulation = nn.Sequential(
        nn.SiLU(),
        nn.Linear(hidden_size, 2 * hidden_size, bias=True)
    )

  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
    x = modulate(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x


class TrackPosEncoder(nn.Module):
  def __init__(self, embed_dim: int, pos_embed_dim: int):
    super(TrackPosEncoder, self).__init__()
    self.lff = LearnableFourierFeaturesEncoding(
        pos_dim=2,
        embed_dim=pos_embed_dim,
        mlp_dim=4*embed_dim
    )

  def forward(self, tracks: torch.Tensor):
    # tracks:  (N, T, 2, H, W)
    x = tracks
    x = rearrange(x, "b t d h w -> b t h w d")
    x = torch.cat(
        [x, self.lff(x)],
        dim=-1
    )
    x = rearrange(x, "b t h w d -> b t d h w ")
    return x


# class PatchEmbed(nn.Module):
#     def __init__(
#             self,
#             img_size: int = 224,
#             patch_size: int = 14,
#             in_chans: int = 3,
#             embed_dim: int = 768,
#             bias: bool = True,
#     ):
#         super().__init__()
#         self.proj = nn.Conv2d(
#           in_chans,
#           embed_dim,
#           kernel_size=patch_size,
#           stride=patch_size,
#           bias=bias
#         )
#         self.norm = nn.Identity()
#         self.num_patches = (
#            img_size // patch_size *
#            img_size // patch_size
#         )
#         self.patch_size = (
#           patch_size,
#           patch_size
#         )

#     def forward(self, x):
#         x = self.proj(x)
#         b, d, h, w = x.shape
#         x = x.view(b, d, h * w).transpose(1, 2).contiguous()
#         # x = rearrange(
#         #   x,
#         #   "b d h w -> b (h w) d"
#         # ).contiguous()
#         x = self.norm(x)
#         return x


class Latte(nn.Module):
  """
  Diffusion model with a Transformer backbone.
  """

  def __init__(
      self,
      input_size=(32, 32),
      patch_size=2,
      in_channels=4,
      hidden_size=1152,
      depth=28,
      num_heads=16,
      mlp_ratio=4.0,
      num_frames=16,
      learn_sigma=True,
      frame_features_dim=384,
      num_horizontal_frame_tokens=33,
      num_vertical_frame_tokens=18,
      extras=1,
      latent=True,
      pos_embed_dim=128,
      text_embed_dim=None,
      num_frame_tokens=256
  ):
    super().__init__()
    self.learn_sigma = learn_sigma
    self.in_channels = in_channels
    self.out_channels = in_channels * 2 if learn_sigma else in_channels
    self.patch_size = patch_size
    self.num_heads = num_heads
    self.extras = extras
    self.num_frames = num_frames
    self.num_frame_tokens = num_frame_tokens
    self.latent = latent
    if isinstance(input_size, int):
      input_size = (input_size, input_size)
    if not latent:
      self.x_encoder = TrackPosEncoder(
          embed_dim=hidden_size,
          pos_embed_dim=pos_embed_dim
      )
      self.x_embedder = PatchEmbed(
          input_size,
          patch_size,
          2 + pos_embed_dim,
          hidden_size,
          bias=True
      )
    else:
      self.x_embedder = PatchEmbed(
          input_size,
          patch_size,
          in_channels,
          hidden_size,
          bias=True
      )
    self.t_embedder = TimestepEmbedder(hidden_size)
    self.frame_proj = nn.Linear(
        frame_features_dim,
        hidden_size
    )
    if text_embed_dim is not None:
      mlp_hidden_dim = int(hidden_size * mlp_ratio)
      def approx_gelu(): return nn.GELU(approximate="tanh")
      from timm.models.vision_transformer import Mlp
      self.text_embeds_proj = Mlp(
          in_features=text_embed_dim,
          hidden_features=mlp_hidden_dim,
          out_features=hidden_size,
          act_layer=approx_gelu,
          drop=0
      )
      # self.text_tokens_proj = nn.Linear(text_embed_dim, hidden_size)
    self.text_conditional = text_embed_dim is not None

    num_patches_v = input_size[0] // patch_size
    num_patches_h = input_size[1] // patch_size
    # Will use fixed sin-cos embedding:
    # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
    # self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
    # self.frame_embed = nn.Parameter(torch.zeros(1, num_frame_tokens, hidden_size), requires_grad=False)
    self.register_buffer(
        "pos_embed",
        torch.zeros(1, num_patches_v * num_patches_h, hidden_size)
    )
    self.register_buffer(
        "temp_embed",
        torch.zeros(1, num_frames, hidden_size)
    )
    self.register_buffer(
        "frame_embed",
        torch.zeros(
            1,
            num_horizontal_frame_tokens * num_vertical_frame_tokens,
            hidden_size
        )
    )
    self.hidden_size = hidden_size
    self.num_frame_tokens = (
        num_vertical_frame_tokens,
        num_horizontal_frame_tokens
    )
    self.num_patches = (
        num_patches_v,
        num_patches_h
    )
    self.blocks = nn.ModuleList([
        TransformerBlock(
            hidden_size,
            num_heads,
            mlp_ratio=mlp_ratio
        ) for _ in range(depth)
    ])
    self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
    self.initialize_weights()

  def initialize_weights(self):
    # Initialize transformer layers:
    def _basic_init(module):
      if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)
    self.apply(_basic_init)

    # Initialize (and freeze) pos_embed by sin-cos embedding:

    pos_embed = get_2d_sincos_pos_embed(
        self.pos_embed.shape[-1],
        self.num_patches[0],
        self.num_patches[1],
    )
    self.pos_embed.data.copy_(
        torch.from_numpy(pos_embed)
        .float()
        .unsqueeze(0)
    )
    temp_embed = get_1d_sincos_temp_embed(
        self.temp_embed.shape[-1],
        self.temp_embed.shape[-2]
    )
    self.temp_embed.data.copy_(
        torch.from_numpy(temp_embed)
        .float()
        .unsqueeze(0)
    )
    frame_embed = get_2d_sincos_pos_embed(
        self.temp_embed.shape[-1],
        self.num_frame_tokens[0],
        self.num_frame_tokens[1],
    )
    self.frame_embed.data.copy_(
        torch.from_numpy(frame_embed)
        .float()
        .unsqueeze(0)
    )
    # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
    w = self.x_embedder.proj.weight.data
    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    nn.init.constant_(self.x_embedder.proj.bias, 0)

    if self.extras == 2:
      # Initialize label embedding table:
      nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

    # Initialize timestep embedding MLP:
    nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
    nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    # Zero-out adaLN modulation layers in Latte blocks:
    for block in self.blocks:
      nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
      nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    # Zero-out output layers:
    nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
    nn.init.constant_(self.final_layer.linear.weight, 0)
    nn.init.constant_(self.final_layer.linear.bias, 0)

  def unpatchify(self, x):
    """
    x: (N, T, patch_size**2 * C)
    imgs: (N, H, W, C)
    """
    c = self.out_channels
    p = self.x_embedder.patch_size[0]
    (h, w) = self.num_patches

    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
    return imgs
  # @torch.cuda.amp.autocast()
  # @torch.compile

  def forward(self,
              x,
              t,
              f,
              text_embeds=None,
              text_tokens=None,
              use_fp16=False):
    """
    Forward pass of Latte.
    x: (N, F, C, H, W) tensor of video inputs
    t: (N,) tensor of diffusion timesteps
    f: (N, P*P, D) tensor of initial frame features
    text_embeds: (N, D) tensor of instruction pooled features
    text_tokens: (N, 25, D) tensor of instruction token embeddings
    """
    if use_fp16:
      x = x.to(dtype=torch.float16)
      f = f.to(dtype=torch.float16)
      if self.text_conditional:
        text_embeds = text_embeds.to(dtype=torch.float16)
        # text_tokens = text_tokens.to(dtype=torch.float16)
    batch_size, *_ = x.shape
    if not self.latent:
      x = self.x_encoder(x)
    if self.text_conditional:
      # Project to model dim
      text_embeds = self.text_embeds_proj(text_embeds)
      # text_tokens = self.text_tokens_proj(text_tokens)
    x = rearrange(x, 'b f c h w -> (b f) c h w')
    x = self.x_embedder(x) + self.pos_embed
    t = self.t_embedder(t, use_fp16=use_fp16)
    if self.text_conditional:
      t += text_embeds
    timestep_spatial = repeat(t, 'n d -> (n c) d', c=self.temp_embed.shape[1])
    timestep_temp = repeat(t, 'n d -> (n c) d', c=self.pos_embed.shape[1])
    c = (
        self.frame_proj(f) +
        self.frame_embed
    )
    c_spatial = repeat(c, 'b n d -> (b c) n d', c=self.temp_embed.shape[1])
    c_temp = repeat(c, 'b n d -> (b c) n d', c=self.pos_embed.shape[1])

    for i in range(0, len(self.blocks), 2):
      spatial_block, temp_block = self.blocks[i:i+2]
      x = spatial_block(x, timestep_spatial, c_spatial)
      x = rearrange(x, '(b f) t d -> (b t) f d', b=batch_size)
      # Add Time Embedding
      if i == 0:
        x = x + self.temp_embed
      x = temp_block(x, timestep_temp, c_temp)
      x = rearrange(x, '(b t) f d -> (b f) t d', b=batch_size)

    x = self.final_layer(x, timestep_spatial)
    x = self.unpatchify(x)
    x = rearrange(x, '(b f) c h w -> b f c h w', b=batch_size)
    return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_1d_sincos_temp_embed(embed_dim, length):
  pos = torch.arange(0, length).unsqueeze(1)
  return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_2d_sincos_pos_embed(embed_dim, h, w, cls_token=False, extra_tokens=0):
  """
  return:
  pos_embed: [grid_h*grid_w, embed_dim] or [1+grid_h*grid_w, embed_dim] (w/ or w/o cls_token)
  """
  grid_h = np.arange(h, dtype=np.float32)
  grid_w = np.arange(w, dtype=np.float32)
  grid = np.meshgrid(grid_w, grid_h)  # here w goes first
  grid = np.stack(grid, axis=0)

  grid = grid.reshape([2, 1, h, w])
  pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
  if cls_token and extra_tokens > 0:
    pos_embed = np.concatenate(
        [np.zeros([extra_tokens, embed_dim]), pos_embed],
        axis=0
    )
  return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
  assert embed_dim % 2 == 0

  # use half of dimensions to encode grid_h
  emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
  emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

  emb = np.concatenate([emb_h, emb_w], axis=1)
  return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
  """
  embed_dim: output dimension for each position
  pos: a list of positions to be encoded: size (M,)
  out: (M, D)
  """
  assert embed_dim % 2 == 0
  omega = np.arange(embed_dim // 2, dtype=np.float64)
  omega /= embed_dim / 2.
  omega = 1. / 10000**omega

  pos = pos.reshape(-1)
  out = np.einsum('m,d->md', pos, omega)

  emb_sin = np.sin(out)
  emb_cos = np.cos(out)

  emb = np.concatenate([emb_sin, emb_cos], axis=1)
  return emb


#################################################################################
#                                   Latte Configs                                  #
#################################################################################

def Latte_XL_2(**kwargs):
  return Latte(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def Latte_XL_4(**kwargs):
  return Latte(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def Latte_XL_8(**kwargs):
  return Latte(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def Latte_L_2(**kwargs):
  return Latte(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def Latte_L_4(**kwargs):
  return Latte(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def Latte_L_8(**kwargs):
  return Latte(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def Latte_B_2(**kwargs):
  return Latte(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def Latte_B_4(**kwargs):
  return Latte(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def Latte_B_8(**kwargs):
  return Latte(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def Latte_S_2(**kwargs):
  return Latte(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def Latte_S_4(**kwargs):
  return Latte(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def Latte_S_8(**kwargs):
  return Latte(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


Latte_models = {
    'Latte-XL/2': Latte_XL_2,  'Latte-XL/4': Latte_XL_4,  'Latte-XL/8': Latte_XL_8,
    'Latte-L/2':  Latte_L_2,   'Latte-L/4':  Latte_L_4,   'Latte-L/8':  Latte_L_8,
    'Latte-B/2':  Latte_B_2,   'Latte-B/4':  Latte_B_4,   'Latte-B/8':  Latte_B_8,
    'Latte-S/2':  Latte_S_2,   'Latte-S/4':  Latte_S_4,   'Latte-S/8':  Latte_S_8,
}


#################################################################################
#                           Timestep Sampling Helpers                           #
#################################################################################

def log_normal_timestep(batch_size: int, device=torch.device("cpu")):
  """Samples timesteps from a logit-normal distribution with N(0,1) before sigmoid."""
  z = torch.randn(batch_size, device=device)  # Sample from N(0,1)
  return torch.sigmoid(z)      # Apply sigmoid to get (0,1) range