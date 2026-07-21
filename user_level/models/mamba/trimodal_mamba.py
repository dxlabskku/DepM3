'''
TriModal Mamba for Audio, Visual, and Text modalities
Extended from mm_bimamba.py to support 3 modalities
'''

# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

from models.mamba.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class TriModalMamba(nn.Module):
    """
    Mamba module for 3 modalities: audio, visual, and text.
    Shares the state transition matrix A across all modalities.
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="v2",
        if_devide_out=True,
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        assert bimamba_type == 'v2', "Only v2 is supported for TriModalMamba"

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.a_gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)
            self.v_gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)
            self.t_gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        # Input projections for each modality
        self.a_in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.v_in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.t_in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Conv1d for each modality (forward direction)
        self.a_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.v_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.t_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # SSM projections for each modality
        self.a_x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.a_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.v_x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.v_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.t_x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.t_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt projections
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.a_dt_proj.weight, dt_init_std)
            nn.init.constant_(self.v_dt_proj.weight, dt_init_std)
            nn.init.constant_(self.t_dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.a_dt_proj.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.v_dt_proj.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.t_dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias for each modality
        for dt_proj in [self.a_dt_proj, self.v_dt_proj, self.t_dt_proj]:
            dt = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True

        # SHARED State transition matrix A (this is the key for collaborative SSM)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter for each modality
        self.a_D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.a_D._no_weight_decay = True
        self.v_D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.v_D._no_weight_decay = True
        self.t_D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.t_D._no_weight_decay = True

        # Bidirectional - backward direction
        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        # Backward conv1d for each modality
        self.a_conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.v_conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.t_conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # Backward SSM projections
        self.a_x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.a_dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.v_x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.v_dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.t_x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.t_dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Backward D parameters
        self.a_D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.a_D_b._no_weight_decay = True
        self.v_D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.v_D_b._no_weight_decay = True
        self.t_D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.t_D_b._no_weight_decay = True

        # Output projections for each modality
        self.a_out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.v_out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.t_out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, a_hidden_states, v_hidden_states, t_hidden_states, 
                a_inference_params=None, v_inference_params=None, t_inference_params=None):
        """
        Forward pass for 3 modalities.
        
        Args:
            a_hidden_states: (B, L, D) audio features
            v_hidden_states: (B, L, D) visual features
            t_hidden_states: (B, L, D) text features
            
        Returns:
            Tuple of (a_out, v_out, t_out) each of shape (B, L, D)
        """
        batch, seqlen, dim = a_hidden_states.shape
        
        # Input projections
        a_xz = rearrange(
            self.a_in_proj.weight @ rearrange(a_hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.a_in_proj.bias is not None:
            a_xz = a_xz + rearrange(self.a_in_proj.bias.to(dtype=a_xz.dtype), "d -> d 1")

        v_xz = rearrange(
            self.v_in_proj.weight @ rearrange(v_hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.v_in_proj.bias is not None:
            v_xz = v_xz + rearrange(self.v_in_proj.bias.to(dtype=v_xz.dtype), "d -> d 1")

        t_xz = rearrange(
            self.t_in_proj.weight @ rearrange(t_hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.t_in_proj.bias is not None:
            t_xz = t_xz + rearrange(self.t_in_proj.bias.to(dtype=t_xz.dtype), "d -> d 1")

        # Shared state transition matrix A
        A = -torch.exp(self.A_log.float())
        A_b = -torch.exp(self.A_b_log.float())

        # Fast path with fused kernels
        if self.use_fast_path and a_inference_params is None and v_inference_params is None and t_inference_params is None:
            # Forward direction
            a_out = mamba_inner_fn_no_out_proj(
                a_xz,
                self.a_conv1d.weight,
                self.a_conv1d.bias,
                self.a_x_proj.weight,
                self.a_dt_proj.weight,
                A,
                None,
                None,
                self.a_D.float(),
                delta_bias=self.a_dt_proj.bias.float(),
                delta_softplus=True,
            )

            v_out = mamba_inner_fn_no_out_proj(
                v_xz,
                self.v_conv1d.weight,
                self.v_conv1d.bias,
                self.v_x_proj.weight,
                self.v_dt_proj.weight,
                A,
                None,
                None,
                self.v_D.float(),
                delta_bias=self.v_dt_proj.bias.float(),
                delta_softplus=True,
            )

            t_out = mamba_inner_fn_no_out_proj(
                t_xz,
                self.t_conv1d.weight,
                self.t_conv1d.bias,
                self.t_x_proj.weight,
                self.t_dt_proj.weight,
                A,
                None,
                None,
                self.t_D.float(),
                delta_bias=self.t_dt_proj.bias.float(),
                delta_softplus=True,
            )

            # Backward direction
            a_out_b = mamba_inner_fn_no_out_proj(
                a_xz.flip([-1]),
                self.a_conv1d_b.weight,
                self.a_conv1d_b.bias,
                self.a_x_proj_b.weight,
                self.a_dt_proj_b.weight,
                A_b,
                None,
                None,
                self.a_D_b.float(),
                delta_bias=self.a_dt_proj_b.bias.float(),
                delta_softplus=True,
            )

            v_out_b = mamba_inner_fn_no_out_proj(
                v_xz.flip([-1]),
                self.v_conv1d_b.weight,
                self.v_conv1d_b.bias,
                self.v_x_proj_b.weight,
                self.v_dt_proj_b.weight,
                A_b,
                None,
                None,
                self.v_D_b.float(),
                delta_bias=self.v_dt_proj_b.bias.float(),
                delta_softplus=True,
            )

            t_out_b = mamba_inner_fn_no_out_proj(
                t_xz.flip([-1]),
                self.t_conv1d_b.weight,
                self.t_conv1d_b.bias,
                self.t_x_proj_b.weight,
                self.t_dt_proj_b.weight,
                A_b,
                None,
                None,
                self.t_D_b.float(),
                delta_bias=self.t_dt_proj_b.bias.float(),
                delta_softplus=True,
            )

            # Combine forward and backward
            if not self.if_devide_out:
                a_out = F.linear(rearrange(a_out + a_out_b.flip([-1]), "b d l -> b l d"), 
                                self.a_out_proj.weight, self.a_out_proj.bias)
                v_out = F.linear(rearrange(v_out + v_out_b.flip([-1]), "b d l -> b l d"), 
                                self.v_out_proj.weight, self.v_out_proj.bias)
                t_out = F.linear(rearrange(t_out + t_out_b.flip([-1]), "b d l -> b l d"), 
                                self.t_out_proj.weight, self.t_out_proj.bias)
            else:
                a_out = F.linear(rearrange(0.5*a_out + 0.5*a_out_b.flip([-1]), "b d l -> b l d"), 
                                self.a_out_proj.weight, self.a_out_proj.bias)
                v_out = F.linear(rearrange(0.5*v_out + 0.5*v_out_b.flip([-1]), "b d l -> b l d"), 
                                self.v_out_proj.weight, self.v_out_proj.bias)
                t_out = F.linear(rearrange(0.5*t_out + 0.5*t_out_b.flip([-1]), "b d l -> b l d"), 
                                self.t_out_proj.weight, self.t_out_proj.bias)
        else:
            raise NotImplementedError("Slow path not implemented for TriModalMamba")

        if self.init_layer_scale is not None:
            a_out = a_out * self.a_gamma
            v_out = v_out * self.v_gamma
            t_out = t_out * self.t_gamma

        return a_out, v_out, t_out

