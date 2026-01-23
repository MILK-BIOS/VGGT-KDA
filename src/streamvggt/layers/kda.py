import logging
import os
from typing import Callable, List, Any, Tuple, Dict, Union
import warnings

import torch
from torch import nn, Tensor
from fla.layers import KimiDeltaAttention
from fla.models.utils import Cache

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


XFORMERS_AVAILABLE = False


class KDABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        head_dim: int = 64,
        expand_v: float = 1.0,
        num_v_heads: int = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        mlp_ratio: float = 4.0,
        ffn_bias: bool = True,
        drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm, # Use RMSNorm or LayerNorm (NEED EXPERIMENT)
        attn_class: Callable[..., nn.Module] = KimiDeltaAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        **kwargs,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            hidden_size=dim,
            expand_v=expand_v,
            num_heads=num_heads,
            head_dim=head_dim,
            num_v_heads=num_v_heads,
            mode=mode,
            use_short_conv=use_short_conv,
            allow_neg_eigval=allow_neg_eigval,
            conv_size=conv_size,
            conv_bias=conv_bias,
            norm_eps=norm_eps,
            layer_idx=layer_idx,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.proj = nn.Linear(dim, dim, bias=ffn_bias)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(
        self, 
        x: Tensor, 
        attn_mask=None, 
        past_key_values=None, 
        output_attentions=False, 
        use_cache=False, 
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Cache]]: 
        
        if output_attentions:
            warnings.warn("`KDAModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        residual = x
        x = self.norm1(x)
        x, attentions, past_key_values = self.attn(
                x,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
        if self.training and self.sample_drop_ratio > 0.0:
            x = residual + self.drop_path1(self.ls1(self.proj(x)))
            residual = x
            x = self.norm2(x)
            x = residual + self.drop_path2(self.ls2(self.mlp(x)))
        else:
            x = residual + self.ls1(self.proj(x))
            residual = x
            x = self.norm2(x)
            x = self.ls2(self.mlp(x))
            x = residual + x

        if use_cache:
            return x, past_key_values
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
    pos=None,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def get_branges_scales(x, sample_drop_ratio=0.0):
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    residual_scale_factor = b / sample_subset_size
    return brange, residual_scale_factor


def add_residual(x, brange, residual, residual_scale_factor, scaling_vector=None):
    if scaling_vector is None:
        x_flat = x.flatten(1)
        residual = residual.flatten(1)
        x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    else:
        x_plus_residual = scaled_index_add(
            x, brange, residual.to(dtype=x.dtype), scaling=scaling_vector, alpha=residual_scale_factor
        )
    return x_plus_residual


attn_bias_cache: Dict[Tuple, Any] = {}


def get_attn_bias_and_cat(x_list, branges=None):
    """
    this will perform the index select, cat the tensors, and provide the attn_bias from cache
    """
    batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
    all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
    if all_shapes not in attn_bias_cache.keys():
        seqlens = []
        for b, x in zip(batch_sizes, x_list):
            for _ in range(b):
                seqlens.append(x.shape[1])
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        attn_bias._batch_sizes = batch_sizes
        attn_bias_cache[all_shapes] = attn_bias

    if branges is not None:
        cat_tensors = index_select_cat([x.flatten(1) for x in x_list], branges).view(1, -1, x_list[0].shape[-1])
    else:
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in x_list)
        cat_tensors = torch.cat(tensors_bs1, dim=1)

    return attn_bias_cache[all_shapes], cat_tensors


def drop_add_residual_stochastic_depth_list(
    x_list: List[Tensor],
    residual_func: Callable[[Tensor, Any], Tensor],
    sample_drop_ratio: float = 0.0,
    scaling_vector=None,
) -> Tensor:
    # 1) generate random set of indices for dropping samples in the batch
    branges_scales = [get_branges_scales(x, sample_drop_ratio=sample_drop_ratio) for x in x_list]
    branges = [s[0] for s in branges_scales]
    residual_scale_factors = [s[1] for s in branges_scales]

    # 2) get attention bias and index+concat the tensors
    attn_bias, x_cat = get_attn_bias_and_cat(x_list, branges)

    # 3) apply residual_func to get residual, and split the result
    residual_list = attn_bias.split(residual_func(x_cat, attn_bias=attn_bias))  # type: ignore

    outputs = []
    for x, brange, residual, residual_scale_factor in zip(x_list, branges, residual_list, residual_scale_factors):
        outputs.append(add_residual(x, brange, residual, residual_scale_factor, scaling_vector).view_as(x))
    return outputs


