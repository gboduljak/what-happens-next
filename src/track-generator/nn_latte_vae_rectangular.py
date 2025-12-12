# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed
from torch.distributions import Normal, kl_divergence

from nn_latte_vae import LearnableFourierFeaturesEncoding, VAEOutput




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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v).reshape(B, N, C)  # require pytorch 2.0
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
        q = self.q_norm(q).reshape(B, N, self.num_heads, C //
                                   self.num_heads).permute(0, 2, 1, 3).contiguous()
        k = self.k_norm(k).reshape(B, M, self.num_heads, C //
                                   self.num_heads).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, M, self.num_heads, C //
                      self.num_heads).permute(0, 2, 1, 3).contiguous()
        # Attention
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v)  # Requires PyTorch 2.0+
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
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
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
        self.norm1 = nn.RMSNorm(
            hidden_size,
            elementwise_affine=True,
            eps=1e-6
        )
        self.attn = SelfAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs
        )
        self.norm2 = nn.RMSNorm(
            hidden_size,
            elementwise_affine=True,
            eps=1e-6
        )
        self.xattn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs
        )
        self.norm3 = nn.RMSNorm(
            hidden_size,
            elementwise_affine=True,
            eps=1e-6
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0
        )
        self.gate_msa = nn.Parameter(torch.zeros((1, 1, hidden_size)))
        self.gate_mxa = nn.Parameter(torch.zeros((1, 1, hidden_size)))
        self.gate_mlp = nn.Parameter(torch.zeros((1, 1, hidden_size)))

    def forward(self, x, c):
        x = x + (
            self.gate_msa *
            self.attn(self.norm1(x))
        )
        x = x + (
            self.gate_mxa *
            self.xattn(self.norm2(x), c)
        )
        x = x + (
            self.gate_mlp *
            self.mlp(self.norm3(x))
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of Latte.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True
        )

    def forward(self, x):
        return self.linear(x)


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
        x = rearrange(x, "b t h w d -> b t d h w")
        return x


class LatteVAE(nn.Module):
    """
    Spatio-Temporal VAE with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=(64, 116),
        patch_size=8,
        in_channels=4,
        latent_channels=8,
        hidden_size=384,
        encoder_depth=6,
        decoder_depth=6,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=24,
        learn_sigma=True,
        frame_features_dim=1024,
        num_horizontal_frame_tokens=33,
        num_vertical_frame_tokens=18,
        extras=1,
        pos_embed_dim=128,
        beta=1.0
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.extras = extras
        self.num_frames = num_frames
        self.beta = beta

        self.latent_channels = latent_channels
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
        if encoder_depth:
            self.frame_proj = nn.Linear(frame_features_dim, hidden_size)
        self.to_latents = nn.Linear(hidden_size, 2*latent_channels)
        self.from_latents = nn.Linear(latent_channels, hidden_size)

        num_patches_v = input_size[0] // patch_size
        num_patches_h = input_size[1] // patch_size
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_v * num_patches_h, hidden_size),
            requires_grad=False
        )
        self.temp_embed = nn.Parameter(
            torch.zeros(1, num_frames, hidden_size),
            requires_grad=False
        )
        if encoder_depth:
            self.frame_embed = nn.Parameter(
                torch.zeros(
                    1,
                    num_horizontal_frame_tokens * num_vertical_frame_tokens,
                    hidden_size
                ),
                requires_grad=False
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
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(encoder_depth)
        ])
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(decoder_depth)
        ])
        self.final_layer = FinalLayer(
            hidden_size,
            patch_size,
            self.out_channels
        )
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
            self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(
            torch.from_numpy(temp_embed).float().unsqueeze(0))
        if self.encoder_blocks:
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

        # Zero-out output layers:
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

    def encode(self, x: torch.Tensor, f: torch.Tensor, use_fp16=False) -> Normal:
        # x: [b, t, c, h, w]
        # f: [b, n, d]
        if use_fp16:
            x = x.to(dtype=torch.float16)
            f = f.to(dtype=torch.float16)

        batch_size, *_ = x.shape
        x = self.x_encoder(x)
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed
        if self.encoder_blocks:
            f = (
                self.frame_proj(f) +
                self.frame_embed
            )
            f_spatial = repeat(
                f,
                'b n d -> (b c) n d',
                c=self.temp_embed.shape[1]
            )
            f_temp = repeat(f, 'b n d -> (b c) n d', c=self.pos_embed.shape[1])

        for i in range(0, len(self.encoder_blocks), 2):
            spatial_block, temp_block = self.encoder_blocks[i:i+2]
            x = spatial_block(x, f_spatial)
            x = rearrange(x, '(b f) t d -> (b t) f d', b=batch_size)
            # Add Time Embedding
            if i == 0:
                x = x + self.temp_embed
            x = temp_block(x, f_temp)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batch_size)

        # x : [(b f), t, d]
        x = rearrange(x, "(b f) t d -> b f t d", b=batch_size)
        [mean, logvar] = self.to_latents(x).chunk(2, dim=-1)
        return Normal(
            loc=mean,
            scale=F.softplus(0.5*logvar) + 1e-8
        )

    def decode(self, z: torch.Tensor, f: torch.Tensor, use_fp16=False) -> torch.Tensor:
        if use_fp16:
            z = z.to(dtype=torch.float16)
            f = f.to(dtype=torch.float16)

        x = self.from_latents(z)  # [b, f, t, d]
        x = rearrange(x, "b f t d -> (b f) t d")
        batch_size, *_ = f.shape

        if self.encoder_blocks:
            f = (
                self.frame_proj(f) +
                self.frame_embed
            )
            batch_size, *_ = f.shape
            f_spatial = repeat(f, 'b n d -> (b c) n d',
                               c=self.temp_embed.shape[1])
            f_temp = repeat(f, 'b n d -> (b c) n d', c=self.pos_embed.shape[1])

        for i in range(0, len(self.decoder_blocks), 2):
            spatial_block, temp_block = self.decoder_blocks[i:i+2]
            x = spatial_block(x, f_spatial)
            x = rearrange(
                x,
                '(b f) n d -> (b n) f d',
                f=self.num_frames
            )
            x = temp_block(x, f_temp)
            x = rearrange(
                x,
                '(b n) f d -> (b f) n d',
                f=self.num_frames,
                b=batch_size
            )
        x = self.final_layer(x)
        x = self.unpatchify(x)
        x = rearrange(
            x, '(b f) c h w -> b f c h w',
            f=self.num_frames,
            b=batch_size
        )
        return x

    # @torch.cuda.amp.autocast()
    # @torch.compile

    def forward(self,
                x,
                f,
                use_fp16=False,
                deterministic=False):
        """
        Forward pass of Latte.
        x: (N, F, C, H, W) tensor of video inputs
        f: (N, P*P, D) tensor of initial frame features
        """
        if use_fp16:
            x = x.to(dtype=torch.float16)
            f = f.to(dtype=torch.float16)

        q_z_given_x = self.encode(x, f, use_fp16)
        p_z = Normal(
            loc=torch.zeros_like(q_z_given_x.loc),
            scale=torch.ones_like(q_z_given_x.scale)
        )

        if deterministic:
            z = q_z_given_x.mode
        else:
            z = q_z_given_x.rsample()

        x_hat = self.decode(z, f, use_fp16)
        # mathematically, it is correct to use sum instead of mean for both kl and ll
        kl = kl_divergence(q_z_given_x, p_z).mean(dim=[1, 2, 3])  # [b, ]
        ll = -F.huber_loss(
            x_hat,
            x,
            reduction="none"
        ).mean(dim=[1, 2, 3, 4])  # [b, ]

        if deterministic:
            elbo = ll
        else:
            elbo = ll - self.beta*kl

        return VAEOutput(
            sample=x_hat,
            latents=z,
            elbo=elbo,
            ll=-F.huber_loss(x_hat, x, reduction="none"),
            kl=kl_divergence(q_z_given_x, p_z),
            latent_dist=q_z_given_x
        )

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