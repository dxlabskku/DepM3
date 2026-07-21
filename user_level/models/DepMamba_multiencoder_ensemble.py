import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from speechbrain.nnet.activations import Swish
from speechbrain.nnet.normalization import LayerNorm

from mamba_ssm import Mamba
from .mamba.bimamba import Mamba as BiMamba
from .mamba.trimodal_mamba import TriModalMamba
from .moe import MultimodalMoEBlock, FiLMModalityMoE
from .base import BaseNet


class TriModalMambaEncoderLayer(nn.Module):
    """Mamba encoder layer for 3 modalities (audio, visual, text)."""
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config is not None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            raise NotImplementedError("Only bidirectional mode supported for TriModalMamba")
        else:
            self.mamba = TriModalMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.norm3 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, a_x, v_x, t_x, a_inference_params=None, v_inference_params=None, t_inference_params=None):
        a_out1, v_out1, t_out1 = self.mamba(a_x, v_x, t_x, a_inference_params, v_inference_params, t_inference_params)
        a_out = a_x + self.norm1(a_out1)
        v_out = v_x + self.norm2(v_out1)
        t_out = t_x + self.norm3(t_out1)
        return a_out, v_out, t_out


class TriModalCNNEncoderLayer(nn.Module):
    """CNN encoder layer for 3 modalities (audio, visual, text)."""
    def __init__(self, input_size, output_size, dropout=0.0, dilation=1):
        super().__init__()

        # Audio CNN
        self.a_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.a_bn = nn.BatchNorm1d(output_size)
        self.a_drop = nn.Dropout(dropout)
        self.a_net = nn.Sequential(self.a_conv, self.a_bn, nn.ReLU(), self.a_drop)

        # Visual CNN
        self.v_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.v_bn = nn.BatchNorm1d(output_size)
        self.v_drop = nn.Dropout(dropout)
        self.v_net = nn.Sequential(self.v_conv, self.v_bn, nn.ReLU(), self.v_drop)

        # Text CNN
        self.t_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.t_bn = nn.BatchNorm1d(output_size)
        self.t_drop = nn.Dropout(dropout)
        self.t_net = nn.Sequential(self.t_conv, self.t_bn, nn.ReLU(), self.t_drop)

        # Skip connections
        if input_size != output_size:
            self.a_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, bias=False)
            self.v_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, bias=False)
            self.t_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, bias=False)
        else:
            self.a_skipconv = None
            self.v_skipconv = None
            self.t_skipconv = None

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.a_conv.weight.data)
        nn.init.xavier_uniform_(self.v_conv.weight.data)
        nn.init.xavier_uniform_(self.t_conv.weight.data)

    def forward(self, xa, xv, xt):
        a_out = self.a_net(xa)
        v_out = self.v_net(xv)
        t_out = self.t_net(xt)
        
        if self.a_skipconv is not None:
            xa = self.a_skipconv(xa)
        if self.v_skipconv is not None:
            xv = self.v_skipconv(xv)
        if self.t_skipconv is not None:
            xt = self.t_skipconv(xt)
            
        return a_out + xa, v_out + xv, t_out + xt


class MambaEncoderLayer(nn.Module):
    """Single modality Mamba encoder layer."""
    def __init__(self, d_model, d_ffn, activation='Swish', dropout=0.0, causal=False, mamba_config=None):
        super().__init__()
        assert mamba_config is not None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(d_model=d_model, **mamba_config)
        else:
            self.mamba = BiMamba(d_model=d_model, bimamba_type='v2', **mamba_config)
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, inference_params=None):
        out = x + self.norm1(self.mamba(x, inference_params))
        return out


class CNNEncoderLayer(nn.Module):
    """Single modality CNN encoder layer."""
    def __init__(self, input_size, output_size, dropout=0.0, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)

        if input_size != output_size:
            self.conv = nn.Conv1d(input_size, output_size, 1, padding=0, bias=False)
        else:
            self.conv = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data)

    def forward(self, x):
        out = self.net(x)
        if self.conv is not None:
            x = self.conv(x)
        return out + x


class TriModalCoSSM(nn.Module):
    """Collaborative SSM for 3 modalities (audio, visual, text)."""
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256, 512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        
        cnn_list = []
        mamba_list = []
        
        for i in range(len(output_sizes)):
            cnn_list.append(TriModalCNNEncoderLayer(
                input_size=input_size if i < 1 else output_sizes[i-1],
                output_size=output_sizes[i],
                dropout=dropout
            ))
            mamba_list.append(TriModalMambaEncoderLayer(
                d_model=output_sizes[i],
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)

    def forward(self, a_x, v_x, t_x, a_inference_params=None, v_inference_params=None, t_inference_params=None):
        a_out, v_out, t_out = a_x, v_x, t_x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            a_out, v_out, t_out = cnn_layer(a_out.permute(0,2,1), v_out.permute(0,2,1), t_out.permute(0,2,1))
            a_out = a_out.permute(0,2,1)
            v_out = v_out.permute(0,2,1)
            t_out = t_out.permute(0,2,1)
            a_out, v_out, t_out = mamba_layer(a_out, v_out, t_out, a_inference_params, v_inference_params, t_inference_params)
            
        return a_out, v_out, t_out


class EnSSM(nn.Module):
    """Enhanced SSM encoder for fused features (without MoE)."""
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256, 512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None,
    ):
        super().__init__()
        
        cnn_list = []
        mamba_list = []
        
        for i in range(len(output_sizes)):
            cnn_list.append(CNNEncoderLayer(
                input_size=input_size if i < 1 else output_sizes[i-1],
                output_size=output_sizes[i],
                dropout=dropout
            ))
            mamba_list.append(MambaEncoderLayer(
                d_model=output_sizes[i],
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)

    def forward(self, x, inference_params=None):
        out = x
        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            out = cnn_layer(out.permute(0, 2, 1))
            out = out.permute(0, 2, 1)
            out = mamba_layer(out, inference_params=inference_params)
        return out


class EnSSM_MoE(nn.Module):
    """Enhanced SSM encoder with MoE for fused features."""
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256, 512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None,
        # MoE parameters
        use_moe=True,
        num_experts=6,
        top_k_experts=2,
        modality_dims=None,
        moe_loss_weight=0.1
    ):
        super().__init__()
        self.use_moe = use_moe
        
        cnn_list = []
        mamba_list = []
        moe_list = []
        
        for i in range(len(output_sizes)):
            cnn_list.append(CNNEncoderLayer(
                input_size=input_size if i < 1 else output_sizes[i-1],
                output_size=output_sizes[i],
                dropout=dropout
            ))
            mamba_list.append(MambaEncoderLayer(
                d_model=output_sizes[i],
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))
            
            # Add MoE layer after each Mamba layer
            if use_moe:
                moe_list.append(MultimodalMoEBlock(
                    input_dim=output_sizes[i],
                    hidden_dim=d_ffn,
                    output_dim=output_sizes[i],
                    num_experts=num_experts,
                    modality_dims=modality_dims,
                    top_k=top_k_experts,
                    dropout=dropout,
                    load_balance_loss_weight=moe_loss_weight,
                    load_balance_loss_type='combined'
                ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)
        if use_moe:
            self.moe_layers = torch.nn.ModuleList(moe_list)

    def forward(
        self,
        x,
        modality_features=None,
        inference_params=None,
        return_moe_loss=True
    ):
        out = x
        total_moe_loss = 0.0
        moe_info_list = []

        for i, (cnn_layer, mamba_layer) in enumerate(zip(self.cnn_layers, self.mamba_layers)):
            out = cnn_layer(out.permute(0, 2, 1))
            out = out.permute(0, 2, 1)
            out = mamba_layer(out, inference_params=inference_params)
            
            # Apply MoE if enabled
            if self.use_moe:
                out, moe_loss, moe_info = self.moe_layers[i](
                    out,
                    modality_features,
                    return_aux_loss=return_moe_loss
                )
                if return_moe_loss and moe_loss is not None:
                    total_moe_loss += moe_loss
                moe_info_list.append(moe_info)

        if return_moe_loss and self.use_moe:
            return out, total_moe_loss, moe_info_list
        else:
            return out, None, moe_info_list


class MultimodalEncoder(nn.Module):
    """Single multimodal encoder (TriModalCoSSM + EnSSM) without MoE.
    
    This is the base encoder that processes Audio + Visual + Text.
    Also includes learnable embedding for missing text modality.
    """
    def __init__(
        self,
        audio_input_size=25,
        video_input_size=136,
        text_input_size=768,
        mm_input_size=256,
        mm_output_sizes=[256, 64],
        d_ffn=1024,
        num_layers=2,
        dropout=0.1,
        activation='GELU',
        causal=False,
        mamba_config=None,
    ):
        super().__init__()
        
        self.audio_input_size = audio_input_size
        self.video_input_size = video_input_size
        self.text_input_size = text_input_size
        
        # Learnable embedding for missing text modality
        # This embedding is learned to complement audio and visual modalities
        # when text features are not available (instead of using zero-vectors)
        self.missing_text_embedding = nn.Parameter(
            torch.randn(text_input_size) * 0.02  # Small initialization
        )
        
        # Input projections
        self.conv_audio = nn.Conv1d(audio_input_size, mm_input_size, 1, bias=False)
        self.conv_video = nn.Conv1d(video_input_size, mm_input_size, 1, bias=False)
        self.conv_text = nn.Conv1d(text_input_size, mm_input_size, 1, bias=False)
        
        nn.init.xavier_uniform_(self.conv_audio.weight.data)
        nn.init.xavier_uniform_(self.conv_video.weight.data)
        nn.init.xavier_uniform_(self.conv_text.weight.data)
        
        # Collaborative SSM for 3 modalities
        self.cossm_encoder = TriModalCoSSM(
            num_layers,
            mm_input_size,
            mm_output_sizes,
            d_ffn,
            activation=activation,
            dropout=dropout,
            causal=causal,
            mamba_config=mamba_config
        )
        
        # Enhanced SSM for fused features (3 modalities)
        self.enssm_encoder = EnSSM(
            num_layers,
            mm_output_sizes[-1] * 3,  # 3 modalities concatenated
            [mm_output_sizes[-1] * 3],
            d_ffn,
            activation=activation,
            dropout=dropout,
            causal=causal,
            mamba_config=mamba_config,
        )
        
        self.output_dim = mm_output_sizes[-1] * 3
        
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: Input tensor (B, T, F) where F = visual_dim + audio_dim + text_dim
            padding_mask: Padding mask (B, T)
        Returns:
            features: (B, T, output_dim)
        """
        # Split modalities
        xv = x[:, :, :self.video_input_size]
        xa = x[:, :, self.video_input_size:self.video_input_size + self.audio_input_size]
        xt = x[:, :, self.video_input_size + self.audio_input_size:]
        
        # Replace missing text (zero-vectors) with learnable embedding
        # Check if text features are all zeros for each time step
        # Shape: (B, T, 1) - True where text is missing
        text_is_missing = (xt.abs().sum(dim=-1, keepdim=True) == 0)
        
        if text_is_missing.any():
            # Expand learnable embedding to match shape: (1, 1, text_dim) -> (B, T, text_dim)
            missing_embed = self.missing_text_embedding.view(1, 1, -1).expand_as(xt)
            # Replace zero-vectors with learnable embedding
            xt = torch.where(text_is_missing, missing_embed, xt)
        
        # Project to common dimension
        xa = self.conv_audio(xa.permute(0, 2, 1)).permute(0, 2, 1)
        xv = self.conv_video(xv.permute(0, 2, 1)).permute(0, 2, 1)
        xt = self.conv_text(xt.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Collaborative SSM processing
        xa, xv, xt = self.cossm_encoder(xa, xv, xt)
        
        # Concatenate all 3 modalities
        x = torch.cat([xa, xv, xt], dim=-1)
        
        # Enhanced SSM processing
        x = self.enssm_encoder(x)
        
        return x


class MultimodalMoEEncoder(nn.Module):
    """Single multimodal encoder with MoE (TriModalCoSSM + EnSSM_MoE).
    
    This encoder includes Mixture of Experts for dynamic expert routing.
    Also includes learnable embedding for missing text modality.
    
    NEW: Pre-CoSSM FiLM-based Modality-aware MoE
    - Applied before CoSSM to process each modality with modality-aware experts
    - Uses FiLM conditioning so experts know which modality they're processing
    """
    def __init__(
        self,
        audio_input_size=25,
        video_input_size=136,
        text_input_size=768,
        mm_input_size=256,
        mm_output_sizes=[256, 64],
        d_ffn=1024,
        num_layers=2,
        dropout=0.1,
        activation='GELU',
        causal=False,
        mamba_config=None,
        # Post-CoSSM MoE parameters (EnSSM_MoE)
        use_moe=True,
        num_experts=6,
        top_k_experts=2,
        moe_loss_weight=0.01,
        # Pre-CoSSM FiLM MoE parameters
        use_pre_cossm_moe=True,
        pre_cossm_num_experts=5,
        pre_cossm_top_k=2,
        pre_cossm_moe_loss_weight=0.01,
        # Modality dropout for robustness to missing modalities
        modality_dropout_rate=0.0,
    ):
        super().__init__()
        
        self.audio_input_size = audio_input_size
        self.video_input_size = video_input_size
        self.text_input_size = text_input_size
        self.use_moe = use_moe
        self.use_pre_cossm_moe = use_pre_cossm_moe
        self.modality_dropout_rate = modality_dropout_rate
        
        # Learnable embeddings for missing modalities (instead of zero-vectors)
        self.missing_audio_embedding = nn.Parameter(
            torch.randn(audio_input_size) * 0.02
        )
        self.missing_visual_embedding = nn.Parameter(
            torch.randn(video_input_size) * 0.02
        )
        self.missing_text_embedding = nn.Parameter(
            torch.randn(text_input_size) * 0.02
        )
        
        # Modality dimensions for MoE gating
        modality_dims = {
            'audio': audio_input_size,
            'visual': video_input_size,
            'text': text_input_size
        }
        
        # Input projections
        self.conv_audio = nn.Conv1d(audio_input_size, mm_input_size, 1, bias=False)
        self.conv_video = nn.Conv1d(video_input_size, mm_input_size, 1, bias=False)
        self.conv_text = nn.Conv1d(text_input_size, mm_input_size, 1, bias=False)
        
        nn.init.xavier_uniform_(self.conv_audio.weight.data)
        nn.init.xavier_uniform_(self.conv_video.weight.data)
        nn.init.xavier_uniform_(self.conv_text.weight.data)
        
        # NEW: Pre-CoSSM FiLM-based Modality-aware MoE
        # Applied after projection, before CoSSM
        # Each expert uses FiLM to be aware of which modality it's processing
        self.pre_cossm_moe = None
        if use_pre_cossm_moe:
            self.pre_cossm_moe = FiLMModalityMoE(
                d_model=mm_input_size,
                hidden_dim=d_ffn,
                num_experts=pre_cossm_num_experts,
                top_k=pre_cossm_top_k,
                dropout=dropout,
                load_balance_weight=pre_cossm_moe_loss_weight,
            )
        
        # Collaborative SSM for 3 modalities
        self.cossm_encoder = TriModalCoSSM(
            num_layers,
            mm_input_size,
            mm_output_sizes,
            d_ffn,
            activation=activation,
            dropout=dropout,
            causal=causal,
            mamba_config=mamba_config
        )
        
        # Enhanced SSM with MoE for fused features (3 modalities)
        self.enssm_encoder = EnSSM_MoE(
            num_layers=num_layers,
            input_size=mm_output_sizes[-1] * 3,  # 3 modalities concatenated
            output_sizes=[mm_output_sizes[-1] * 3],
            d_ffn=d_ffn,
            activation=activation,
            dropout=dropout,
            causal=causal,
            mamba_config=mamba_config,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            modality_dims=modality_dims,
            moe_loss_weight=moe_loss_weight,
        )
        
        self.output_dim = mm_output_sizes[-1] * 3
        
        # Store MoE losses
        self.last_moe_loss = None
        self.last_moe_info = None
        self.last_pre_moe_loss = None
        self.last_pre_moe_info = None
        
    def split_modalities(self, x):
        """Split input tensor into 3 modalities."""
        xv = x[:, :, :self.video_input_size]
        xa = x[:, :, self.video_input_size:self.video_input_size + self.audio_input_size]
        xt = x[:, :, self.video_input_size + self.audio_input_size:]
        return {
            'visual': xv,
            'audio': xa,
            'text': xt
        }
        
    def forward(self, x, padding_mask=None, return_moe_loss=True):
        """
        Args:
            x: Input tensor (B, T, F) where F = visual_dim + audio_dim + text_dim
            padding_mask: Padding mask (B, T)
            return_moe_loss: Whether to compute MoE loss
        Returns:
            features: (B, T, output_dim)
        """
        # Split and store modality features for MoE gating
        modality_features = self.split_modalities(x)
        xv = modality_features['visual']
        xa = modality_features['audio']
        xt = modality_features['text']
        
        # --- Detect naturally missing modalities (zero-vectors) ---
        audio_is_missing = (xa.abs().sum(dim=-1, keepdim=True) == 0)
        visual_is_missing = (xv.abs().sum(dim=-1, keepdim=True) == 0)
        text_is_missing = (xt.abs().sum(dim=-1, keepdim=True) == 0)
        
        # Per-sample missing flags (True if ALL time-steps are missing)
        audio_missing_flag = audio_is_missing.all(dim=1).any(dim=-1)  # (B,)
        visual_missing_flag = visual_is_missing.all(dim=1).any(dim=-1)  # (B,)
        text_missing_flag = text_is_missing.all(dim=1).any(dim=-1)  # (B,)
        
        # Replace naturally missing modalities with learnable embeddings
        if audio_is_missing.any():
            xa = torch.where(audio_is_missing, self.missing_audio_embedding.view(1, 1, -1).expand_as(xa), xa)
            modality_features['audio'] = xa
        if visual_is_missing.any():
            xv = torch.where(visual_is_missing, self.missing_visual_embedding.view(1, 1, -1).expand_as(xv), xv)
            modality_features['visual'] = xv
        if text_is_missing.any():
            xt = torch.where(text_is_missing, self.missing_text_embedding.view(1, 1, -1).expand_as(xt), xt)
            modality_features['text'] = xt
        
        # --- Modality dropout (training only) ---
        # Randomly drop modalities to simulate missing data, keeping at least 1
        if self.training and self.modality_dropout_rate > 0:
            drop_audio = torch.rand(1).item() < self.modality_dropout_rate
            drop_visual = torch.rand(1).item() < self.modality_dropout_rate
            drop_text = torch.rand(1).item() < self.modality_dropout_rate
            
            # Ensure at least one modality survives
            if drop_audio and drop_visual and drop_text:
                restore = random.choice(['audio', 'visual', 'text'])
                if restore == 'audio':
                    drop_audio = False
                elif restore == 'visual':
                    drop_visual = False
                else:
                    drop_text = False
            
            if drop_audio:
                xa = self.missing_audio_embedding.view(1, 1, -1).expand_as(xa)
                modality_features['audio'] = xa
                audio_missing_flag = torch.ones(xa.shape[0], dtype=torch.bool, device=xa.device)
            if drop_visual:
                xv = self.missing_visual_embedding.view(1, 1, -1).expand_as(xv)
                modality_features['visual'] = xv
                visual_missing_flag = torch.ones(xv.shape[0], dtype=torch.bool, device=xv.device)
            if drop_text:
                xt = self.missing_text_embedding.view(1, 1, -1).expand_as(xt)
                modality_features['text'] = xt
                text_missing_flag = torch.ones(xt.shape[0], dtype=torch.bool, device=xt.device)
        
        # Aggregate per-sample flags to batch-level (any sample missing → True)
        a_is_missing = audio_missing_flag.any().item()
        v_is_missing = visual_missing_flag.any().item()
        t_is_missing = text_missing_flag.any().item()
        
        # Project to common dimension
        xa = self.conv_audio(xa.permute(0, 2, 1)).permute(0, 2, 1)
        xv = self.conv_video(xv.permute(0, 2, 1)).permute(0, 2, 1)
        xt = self.conv_text(xt.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Pre-CoSSM FiLM-based Modality-aware MoE
        # Each modality is processed by experts that know which modality
        # they're handling via FiLM conditioning + missing flag
        pre_moe_loss = 0.0
        pre_moe_info = {}
        if self.use_pre_cossm_moe and self.pre_cossm_moe is not None:
            xa, a_loss, a_info = self.pre_cossm_moe(xa, 'audio', is_missing=a_is_missing, return_aux_loss=return_moe_loss)
            xv, v_loss, v_info = self.pre_cossm_moe(xv, 'visual', is_missing=v_is_missing, return_aux_loss=return_moe_loss)
            xt, t_loss, t_info = self.pre_cossm_moe(xt, 'text', is_missing=t_is_missing, return_aux_loss=return_moe_loss)
            
            if return_moe_loss:
                pre_moe_loss = a_loss + v_loss + t_loss
                pre_moe_info = {'audio': a_info, 'visual': v_info, 'text': t_info}
        
        # Store Pre-CoSSM MoE loss and info
        self.last_pre_moe_loss = pre_moe_loss
        self.last_pre_moe_info = pre_moe_info
        
        # Collaborative SSM processing
        xa, xv, xt = self.cossm_encoder(xa, xv, xt)
        
        # Concatenate all 3 modalities
        x = torch.cat([xa, xv, xt], dim=-1)
        
        # Enhanced SSM processing with MoE (Post-CoSSM)
        x, moe_loss, moe_info = self.enssm_encoder(
            x,
            modality_features=modality_features,
            return_moe_loss=return_moe_loss and self.use_moe
        )
        
        # Store Post-CoSSM MoE loss and info
        if return_moe_loss and self.use_moe:
            self.last_moe_loss = moe_loss
            self.last_moe_info = moe_info
        
        return x
    
    def get_moe_loss(self):
        """Get Post-CoSSM MoE loss from last forward pass."""
        return self.last_moe_loss if self.last_moe_loss is not None else torch.tensor(0.0)
    
    def get_moe_info(self):
        """Get Post-CoSSM MoE routing info from last forward pass."""
        return self.last_moe_info
    
    def get_pre_moe_loss(self):
        """Get Pre-CoSSM FiLM MoE loss from last forward pass."""
        if self.last_pre_moe_loss is not None:
            if isinstance(self.last_pre_moe_loss, torch.Tensor):
                return self.last_pre_moe_loss
            return torch.tensor(self.last_pre_moe_loss)
        return torch.tensor(0.0)
    
    def get_pre_moe_info(self):
        """Get Pre-CoSSM FiLM MoE routing info from last forward pass."""
        return self.last_pre_moe_info
    
    def get_total_moe_loss(self):
        """Get total MoE loss (Pre-CoSSM + Post-CoSSM) from last forward pass."""
        pre_loss = self.get_pre_moe_loss()
        post_loss = self.get_moe_loss()
        
        # Ensure both are tensors
        if not isinstance(pre_loss, torch.Tensor):
            pre_loss = torch.tensor(pre_loss)
        if not isinstance(post_loss, torch.Tensor):
            post_loss = torch.tensor(post_loss)
            
        # Move to same device if needed
        if pre_loss.device != post_loss.device:
            pre_loss = pre_loss.to(post_loss.device)
            
        return pre_loss + post_loss


class EnsembleHead(nn.Module):
    """Ensemble fusion head that combines multiple encoder outputs (Early Fusion).
    
    Early Fusion: fuse at the feature level, then use a single classifier
    
    Supports multiple fusion strategies:
    - average: Simple averaging of predictions
    - weighted: Learnable weighted average
    - attention: Attention-based dynamic weighting
    - concat: Concatenate features and use MLP
    """
    def __init__(
        self,
        num_encoders: int = 3,
        feature_dim: int = 192,
        fusion_type: str = 'weighted',
        num_classes: int = 1,
    ):
        super().__init__()
        
        self.num_encoders = num_encoders
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        self.num_classes = num_classes
        
        if fusion_type == 'weighted':
            # Learnable weights for each encoder
            self.ensemble_weights = nn.Parameter(torch.ones(num_encoders) / num_encoders)
            self.classifier = nn.Linear(feature_dim, num_classes)
            
        elif fusion_type == 'attention':
            # Attention-based weighting
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            self.classifier = nn.Linear(feature_dim, num_classes)
            
        elif fusion_type == 'concat':
            # Concatenate all encoder features
            self.fusion_layer = nn.Sequential(
                nn.Linear(feature_dim * num_encoders, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            self.classifier = nn.Linear(feature_dim, num_classes)
            
        else:  # average
            self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid() if num_classes == 1 else None
        
    def forward(self, encoder_features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            encoder_features: List of (B, feature_dim) tensors from each encoder
        Returns:
            logits: (B, num_classes)
            info: Dictionary with ensemble information
        """
        info = {}
        
        if self.fusion_type == 'weighted':
            # Normalized weights
            weights = F.softmax(self.ensemble_weights, dim=0)
            info['ensemble_weights'] = weights.detach()
            
            # Weighted average of features
            stacked = torch.stack(encoder_features, dim=0)  # (num_encoders, B, feature_dim)
            fused = (stacked * weights.view(-1, 1, 1)).sum(dim=0)  # (B, feature_dim)
            
        elif self.fusion_type == 'attention':
            # Compute attention scores for each encoder
            stacked = torch.stack(encoder_features, dim=1)  # (B, num_encoders, feature_dim)
            attn_scores = self.attention(stacked).squeeze(-1)  # (B, num_encoders)
            attn_weights = F.softmax(attn_scores, dim=1)  # (B, num_encoders)
            info['attention_weights'] = attn_weights.detach()
            
            # Weighted combination
            fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, feature_dim)
            
        elif self.fusion_type == 'concat':
            # Concatenate all features
            concat = torch.cat(encoder_features, dim=1)  # (B, feature_dim * num_encoders)
            fused = self.fusion_layer(concat)  # (B, feature_dim)
            
        else:  # average
            stacked = torch.stack(encoder_features, dim=0)
            fused = stacked.mean(dim=0)
        
        logits = self.classifier(fused)
        return logits, info
    
    def get_ensemble_weights(self) -> Optional[torch.Tensor]:
        """Get current ensemble weights (for weighted fusion)."""
        if self.fusion_type == 'weighted':
            return F.softmax(self.ensemble_weights, dim=0)
        return None


class LateEnsembleHead(nn.Module):
    """Late Ensemble Head - per-encoder classifiers just before classification, then ensemble.
    
    Late Fusion: each encoder has its own classifier; logits/predictions are ensembled
    
    Advantages of this design:
    - each encoder learns independently, encouraging diversity
    - easy to interpret as a classical ensemble
    - per-encoder performance can be analyzed
    
    Supports multiple ensemble strategies:
    - average: Simple averaging of logits
    - weighted: Learnable weighted average of logits
    - attention: Attention-based dynamic weighting of logits
    - voting: Soft voting (probability averaging)
    - stacking: Meta-learner for combining predictions
    """
    def __init__(
        self,
        num_encoders: int = 3,
        feature_dim: int = 192,
        ensemble_type: str = 'weighted',
        num_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_encoders = num_encoders
        self.feature_dim = feature_dim
        self.ensemble_type = ensemble_type
        self.num_classes = num_classes
        
        # build one classifier per encoder
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes)
            )
            for _ in range(num_encoders)
        ])
        
        if ensemble_type == 'weighted':
            # Learnable weights for each encoder's logits
            self.ensemble_weights = nn.Parameter(torch.ones(num_encoders) / num_encoders)
            
        elif ensemble_type == 'attention':
            # Attention-based weighting using features
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            
        elif ensemble_type == 'stacking':
            # Meta-learner: takes all encoders' logits as input for the final prediction
            self.meta_learner = nn.Sequential(
                nn.Linear(num_encoders * num_classes, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, num_classes)
            )
            
        elif ensemble_type == 'gating':
            # Gating network: feature-based weights for each classifier
            self.gating_network = nn.Sequential(
                nn.Linear(feature_dim * num_encoders, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_encoders),
            )
        
    def forward(self, encoder_features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            encoder_features: List of (B, feature_dim) tensors from each encoder
        Returns:
            logits: (B, num_classes)
            info: Dictionary with ensemble information
        """
        info = {}
        batch_size = encoder_features[0].shape[0]
        device = encoder_features[0].device
        
        # per-encoder classification
        individual_logits = []
        for i, (feat, classifier) in enumerate(zip(encoder_features, self.classifiers)):
            logit = classifier(feat)  # (B, num_classes)
            individual_logits.append(logit)
        
        # Stack logits: (num_encoders, B, num_classes)
        stacked_logits = torch.stack(individual_logits, dim=0)
        
        # store individual predictions (for analysis)
        info['individual_logits'] = [l.detach() for l in individual_logits]
        info['individual_probs'] = [torch.sigmoid(l).detach() for l in individual_logits]
        
        if self.ensemble_type == 'weighted':
            # Learnable weighted average of logits
            weights = F.softmax(self.ensemble_weights, dim=0)  # (num_encoders,)
            info['ensemble_weights'] = weights.detach()
            
            # Weighted sum of logits
            final_logits = (stacked_logits * weights.view(-1, 1, 1)).sum(dim=0)  # (B, num_classes)
            
        elif self.ensemble_type == 'attention':
            # Attention-based weighting using features
            stacked_features = torch.stack(encoder_features, dim=1)  # (B, num_encoders, feature_dim)
            attn_scores = self.attention(stacked_features).squeeze(-1)  # (B, num_encoders)
            attn_weights = F.softmax(attn_scores, dim=1)  # (B, num_encoders)
            info['attention_weights'] = attn_weights.detach()
            
            # Weighted combination of logits
            # stacked_logits: (num_encoders, B, num_classes) -> (B, num_encoders, num_classes)
            stacked_logits_t = stacked_logits.permute(1, 0, 2)
            final_logits = (stacked_logits_t * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, num_classes)
            
        elif self.ensemble_type == 'voting':
            # Soft voting: average probabilities then convert back to logits
            probs = torch.sigmoid(stacked_logits)  # (num_encoders, B, num_classes)
            avg_prob = probs.mean(dim=0)  # (B, num_classes)
            # Convert back to logits (inverse sigmoid = logit function)
            final_logits = torch.log(avg_prob / (1 - avg_prob + 1e-8) + 1e-8)
            info['average_probability'] = avg_prob.detach()
            
        elif self.ensemble_type == 'stacking':
            # Meta-learner: concatenate all logits and learn to combine
            concat_logits = torch.cat(individual_logits, dim=1)  # (B, num_encoders * num_classes)
            final_logits = self.meta_learner(concat_logits)  # (B, num_classes)
            info['meta_input'] = concat_logits.detach()
            
        elif self.ensemble_type == 'gating':
            # Gating network: feature-based gating
            concat_features = torch.cat(encoder_features, dim=1)  # (B, feature_dim * num_encoders)
            gate_weights = F.softmax(self.gating_network(concat_features), dim=1)  # (B, num_encoders)
            info['gate_weights'] = gate_weights.detach()
            
            # Apply gating weights to logits
            stacked_logits_t = stacked_logits.permute(1, 0, 2)  # (B, num_encoders, num_classes)
            final_logits = (stacked_logits_t * gate_weights.unsqueeze(-1)).sum(dim=1)  # (B, num_classes)
            
        else:  # average
            final_logits = stacked_logits.mean(dim=0)  # (B, num_classes)
        
        return final_logits, info
    
    def get_ensemble_weights(self) -> Optional[torch.Tensor]:
        """Get current ensemble weights (for weighted ensemble)."""
        if self.ensemble_type == 'weighted':
            return F.softmax(self.ensemble_weights, dim=0)
        return None
    
    def get_individual_predictions(self, encoder_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Get individual predictions from each classifier."""
        predictions = []
        for feat, classifier in zip(encoder_features, self.classifiers):
            logit = classifier(feat)
            predictions.append(logit)
        return predictions


class DepMambaMultiEncoderEnsemble(BaseNet):
    """DepMamba with Multiple Multimodal Encoders + MoE and Ensemble Fusion.
    
    This model contains N multimodal encoders (TriModalCoSSM + EnSSM_MoE) that process
    the same input. Each encoder includes Mixture of Experts (MoE) for dynamic routing.
    The outputs are combined using an ensemble fusion head.
    
    All encoders are trained end-to-end together.
    
    Architecture:
    - Multiple MultimodalMoEEncoders (each with its own MoE experts)
    - Ensemble fusion of encoder outputs
    
    Loss components:
    - Classification loss (BCE)
    - Diversity loss (encourages different encoder behaviors)
    - MoE load balancing loss (encourages uniform expert usage)
    
    Supports two ensemble strategies:
    - Early Ensemble: feature fusion followed by a single classifier (original scheme)
    - Late Ensemble: per-encoder classifiers followed by logits ensembling (new scheme)
    
    Args:
        num_encoders: Number of multimodal encoders
        audio_input_size: Audio feature dimension
        video_input_size: Visual feature dimension
        text_input_size: Text feature dimension
        mm_input_size: Common dimension after projection
        mm_output_sizes: Output sizes for encoder layers
        d_ffn: Feed-forward network dimension
        num_layers: Number of encoder layers
        dropout: Dropout rate
        activation: Activation function
        causal: Whether to use causal mode
        mamba_config: Mamba configuration
        ensemble_stage: 'early' (feature fusion) or 'late' (logits ensemble)
        fusion_type: Type of ensemble fusion 
            - For early: 'average', 'weighted', 'attention', 'concat'
            - For late: 'average', 'weighted', 'attention', 'voting', 'stacking', 'gating'
        num_classes: Number of output classes (1 for binary)
        diversity_loss_weight: Weight for diversity loss to encourage different encoder behaviors
        use_moe: Whether to use MoE in encoders
        num_experts: Number of experts per MoE layer
        top_k_experts: Number of top experts to activate
        moe_loss_weight: Weight for MoE load balancing loss
    """
    
    def __init__(
        self,
        num_encoders: int = 3,
        audio_input_size: int = 25,
        video_input_size: int = 136,
        text_input_size: int = 768,
        mm_input_size: int = 256,
        mm_output_sizes: List[int] = [256, 64],
        d_ffn: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'GELU',
        causal: bool = False,
        mamba_config: dict = None,
        ensemble_stage: str = 'late',  # 'early' or 'late'
        fusion_type: str = 'weighted',
        num_classes: int = 1,
        diversity_loss_weight: float = 0.01,
        # Post-CoSSM MoE parameters (EnSSM_MoE)
        use_moe: bool = True,
        num_experts: int = 6,
        top_k_experts: int = 2,
        moe_loss_weight: float = 0.01,
        # Pre-CoSSM FiLM MoE parameters
        use_pre_cossm_moe: bool = True,
        pre_cossm_num_experts: int = 5,
        pre_cossm_top_k: int = 2,
        pre_cossm_moe_loss_weight: float = 0.01,
        # Modality dropout for robustness to missing modalities
        modality_dropout_rate: float = 0.0,
    ):
        super().__init__()
        
        self.num_encoders = num_encoders
        self.ensemble_stage = ensemble_stage
        self.fusion_type = fusion_type
        self.diversity_loss_weight = diversity_loss_weight
        self.num_classes = num_classes
        self.use_moe = use_moe
        self.moe_loss_weight = moe_loss_weight
        self.use_pre_cossm_moe = use_pre_cossm_moe
        self.modality_dropout_rate = modality_dropout_rate
        
        # Create multiple multimodal encoders with MoE
        self.encoders = nn.ModuleList()
        for i in range(num_encoders):
            if use_moe or use_pre_cossm_moe:
                encoder = MultimodalMoEEncoder(
                    audio_input_size=audio_input_size,
                    video_input_size=video_input_size,
                    text_input_size=text_input_size,
                    mm_input_size=mm_input_size,
                    mm_output_sizes=mm_output_sizes,
                    d_ffn=d_ffn,
                    num_layers=num_layers,
                    dropout=dropout,
                    activation=activation,
                    causal=causal,
                    mamba_config=mamba_config.copy() if mamba_config else None,
                    # Post-CoSSM MoE
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k_experts=top_k_experts,
                    moe_loss_weight=moe_loss_weight,
                    # Pre-CoSSM FiLM MoE
                    use_pre_cossm_moe=use_pre_cossm_moe,
                    pre_cossm_num_experts=pre_cossm_num_experts,
                    pre_cossm_top_k=pre_cossm_top_k,
                    pre_cossm_moe_loss_weight=pre_cossm_moe_loss_weight,
                    # Modality dropout
                    modality_dropout_rate=modality_dropout_rate,
                )
            else:
                encoder = MultimodalEncoder(
                    audio_input_size=audio_input_size,
                    video_input_size=video_input_size,
                    text_input_size=text_input_size,
                    mm_input_size=mm_input_size,
                    mm_output_sizes=mm_output_sizes,
                    d_ffn=d_ffn,
                    num_layers=num_layers,
                    dropout=dropout,
                    activation=activation,
                    causal=causal,
                    mamba_config=mamba_config.copy() if mamba_config else None,
                )
            self.encoders.append(encoder)
        
        # Pooling layer
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Ensemble fusion head
        feature_dim = mm_output_sizes[-1] * 3  # 3 modalities
        
        if ensemble_stage == 'late':
            # Late Ensemble: per-encoder classifiers, then logits ensembling
            self.ensemble_head = LateEnsembleHead(
                num_encoders=num_encoders,
                feature_dim=feature_dim,
                ensemble_type=fusion_type,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            # Early Ensemble: feature fusion followed by a single classifier (original scheme)
            self.ensemble_head = EnsembleHead(
                num_encoders=num_encoders,
                feature_dim=feature_dim,
                fusion_type=fusion_type,
                num_classes=num_classes,
            )
        
        # Store losses
        self.last_diversity_loss = None
        self.last_moe_loss = None  # Post-CoSSM MoE
        self.last_pre_moe_loss = None  # Pre-CoSSM FiLM MoE
        self.last_ensemble_info = None
        self.last_moe_info = None
        self.last_pre_moe_info = None
        
    def feature_extractor(self, x, padding_mask=None, return_moe_loss=True):
        """Extract features from all encoders.
        
        Args:
            x: Input tensor (B, T, F)
            padding_mask: Padding mask (B, T)
            return_moe_loss: Whether to compute MoE loss
            
        Returns:
            List of pooled features from each encoder
        """
        encoder_features = []
        total_moe_loss = 0.0
        total_pre_moe_loss = 0.0
        moe_info_list = []
        pre_moe_info_list = []
        
        for encoder in self.encoders:
            # Get encoder output
            if self.use_moe or self.use_pre_cossm_moe:
                enc_out = encoder(x, padding_mask, return_moe_loss=return_moe_loss)
                if return_moe_loss:
                    # Post-CoSSM MoE loss
                    if self.use_moe:
                        moe_loss = encoder.get_moe_loss()
                        if moe_loss is not None and isinstance(moe_loss, torch.Tensor):
                            total_moe_loss = total_moe_loss + moe_loss
                        moe_info_list.append(encoder.get_moe_info())
                    
                    # Pre-CoSSM FiLM MoE loss
                    if self.use_pre_cossm_moe:
                        pre_moe_loss = encoder.get_pre_moe_loss()
                        if pre_moe_loss is not None and isinstance(pre_moe_loss, torch.Tensor):
                            total_pre_moe_loss = total_pre_moe_loss + pre_moe_loss
                        pre_moe_info_list.append(encoder.get_pre_moe_info())
            else:
                enc_out = encoder(x, padding_mask)
            
            # Pooling
            if padding_mask is not None:
                enc_out = enc_out * (padding_mask.unsqueeze(-1).float())
                pooled = enc_out.sum(dim=1) / (padding_mask.unsqueeze(-1).float().sum(dim=1) + 1e-8)
            else:
                pooled = self.pool(enc_out.permute(0, 2, 1)).squeeze(-1)
            
            encoder_features.append(pooled)
        
        # Store MoE loss and info
        if return_moe_loss:
            if self.use_moe:
                self.last_moe_loss = total_moe_loss
                self.last_moe_info = moe_info_list
            if self.use_pre_cossm_moe:
                self.last_pre_moe_loss = total_pre_moe_loss
                self.last_pre_moe_info = pre_moe_info_list
        
        return encoder_features
    
    def compute_diversity_loss(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """Compute diversity loss to encourage different encoder behaviors.
        
        Uses negative correlation learning to promote diversity.
        """
        if len(encoder_features) < 2:
            return torch.tensor(0.0, device=encoder_features[0].device)
        
        # Stack features
        stacked = torch.stack(encoder_features, dim=0)  # (num_encoders, B, feature_dim)
        
        # Compute pairwise cosine similarity
        diversity_loss = 0.0
        count = 0
        for i in range(len(encoder_features)):
            for j in range(i + 1, len(encoder_features)):
                # Cosine similarity between encoder i and j
                sim = F.cosine_similarity(encoder_features[i], encoder_features[j], dim=1)
                diversity_loss += sim.mean()
                count += 1
        
        # We want to minimize similarity (maximize diversity)
        diversity_loss = diversity_loss / count if count > 0 else 0.0
        
        return diversity_loss * self.diversity_loss_weight
    
    def classifier(self, encoder_features: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Classification with ensemble fusion."""
        return self.ensemble_head(encoder_features)
    
    def forward(self, x, padding_mask=None, return_diversity_loss=True, return_moe_loss=True):
        """
        Forward pass through the multi-encoder ensemble with MoE.
        
        Args:
            x: Input tensor (B, T, F) where F = visual_dim + audio_dim + text_dim
            padding_mask: Padding mask (B, T)
            return_diversity_loss: Whether to compute diversity loss
            return_moe_loss: Whether to compute MoE load balancing loss
            
        Returns:
            logits: (B, num_classes)
        """
        # Extract features from all encoders (with MoE)
        encoder_features = self.feature_extractor(x, padding_mask, return_moe_loss=return_moe_loss)
        
        # Compute diversity loss
        if return_diversity_loss:
            self.last_diversity_loss = self.compute_diversity_loss(encoder_features)
        
        # Ensemble fusion and classification
        logits, info = self.classifier(encoder_features)
        self.last_ensemble_info = info
        
        return logits
    
    def get_diversity_loss(self) -> torch.Tensor:
        """Get diversity loss from last forward pass."""
        return self.last_diversity_loss if self.last_diversity_loss is not None else torch.tensor(0.0)
    
    def get_moe_loss(self) -> torch.Tensor:
        """Get Post-CoSSM MoE load balancing loss from last forward pass."""
        if self.last_moe_loss is not None:
            if isinstance(self.last_moe_loss, torch.Tensor):
                return self.last_moe_loss
            return torch.tensor(self.last_moe_loss)
        return torch.tensor(0.0)
    
    def get_pre_moe_loss(self) -> torch.Tensor:
        """Get Pre-CoSSM FiLM MoE load balancing loss from last forward pass."""
        if self.last_pre_moe_loss is not None:
            if isinstance(self.last_pre_moe_loss, torch.Tensor):
                return self.last_pre_moe_loss
            return torch.tensor(self.last_pre_moe_loss)
        return torch.tensor(0.0)
    
    def get_total_moe_loss(self) -> torch.Tensor:
        """Get total MoE loss (Pre-CoSSM + Post-CoSSM) from last forward pass."""
        pre_loss = self.get_pre_moe_loss()
        post_loss = self.get_moe_loss()
        
        # Ensure both are tensors on the same device
        if isinstance(pre_loss, torch.Tensor) and isinstance(post_loss, torch.Tensor):
            if pre_loss.device != post_loss.device:
                pre_loss = pre_loss.to(post_loss.device)
        
        return pre_loss + post_loss
    
    def get_total_aux_loss(self) -> torch.Tensor:
        """Get total auxiliary loss (diversity + Pre-CoSSM MoE + Post-CoSSM MoE)."""
        diversity_loss = self.get_diversity_loss()
        total_moe_loss = self.get_total_moe_loss()
        
        # Ensure both are tensors on the same device
        if isinstance(diversity_loss, torch.Tensor) and isinstance(total_moe_loss, torch.Tensor):
            if diversity_loss.device != total_moe_loss.device:
                total_moe_loss = total_moe_loss.to(diversity_loss.device)
        
        return diversity_loss + total_moe_loss
    
    def get_ensemble_weights(self) -> Optional[torch.Tensor]:
        """Get ensemble weights."""
        return self.ensemble_head.get_ensemble_weights()
    
    def get_ensemble_info(self) -> Optional[Dict]:
        """Get ensemble information from last forward pass."""
        return self.last_ensemble_info
    
    def get_moe_info(self) -> Optional[List]:
        """Get Post-CoSSM MoE routing information from last forward pass."""
        return self.last_moe_info
    
    def get_pre_moe_info(self) -> Optional[List]:
        """Get Pre-CoSSM FiLM MoE routing information from last forward pass."""
        return self.last_pre_moe_info
    
    def get_individual_predictions(self, x, padding_mask=None) -> List[torch.Tensor]:
        """Get individual predictions from each encoder.
        
        Useful for analysis of encoder contributions.
        
        For Late Ensemble: Uses the individual classifiers
        For Early Ensemble: Uses a temporary shared classifier
        """
        encoder_features = self.feature_extractor(x, padding_mask)
        
        if self.ensemble_stage == 'late':
            # Late Ensemble: use per-encoder classifiers
            return self.ensemble_head.get_individual_predictions(encoder_features)
        else:
            # Early Ensemble: use a temporary shared classifier
            predictions = []
            temp_classifier = nn.Linear(
                encoder_features[0].shape[-1], 
                self.num_classes
            ).to(x.device)
            
            for feat in encoder_features:
                pred = temp_classifier(feat)
                predictions.append(pred)
            
            return predictions
    
    def get_individual_metrics(self, x, y, padding_mask=None) -> Dict:
        """Get individual encoder metrics for analysis.
        
        Args:
            x: Input tensor
            y: Labels
            padding_mask: Padding mask
            
        Returns:
            Dictionary with individual encoder accuracies and predictions
        """
        predictions = self.get_individual_predictions(x, padding_mask)
        
        metrics = {}
        for i, pred in enumerate(predictions):
            pred_binary = (pred > 0.).int()
            correct = (pred_binary == y.unsqueeze(1)).float().mean()
            metrics[f'encoder_{i}_acc'] = correct.item()
            metrics[f'encoder_{i}_pred'] = pred.detach()
        
        return metrics


if __name__ == '__main__':
    batch_size = 4
    seq_len = 100
    
    visual_dim = 136
    audio_dim = 25
    text_dim = 768
    total_dim = visual_dim + audio_dim + text_dim
    
    x = torch.randn(batch_size, seq_len, total_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, 2, (batch_size,)).float().unsqueeze(1)
    
    mamba_config = {
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'bidirectional': True
    }
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    # ======================================================
    # Test 1: Full model with Pre-CoSSM FiLM MoE + modality dropout
    # ======================================================
    print("="*80)
    print("Test 1: Full model (Post-MoE + Pre-CoSSM FiLM MoE + modality dropout)")
    print("="*80)
    
    model = DepMambaMultiEncoderEnsemble(
        num_encoders=2,
        audio_input_size=audio_dim,
        video_input_size=visual_dim,
        text_input_size=text_dim,
        mm_input_size=256,
        mm_output_sizes=[256, 64],
        d_ffn=1024,
        num_layers=2,
        mamba_config=mamba_config,
        ensemble_stage='late',
        fusion_type='weighted',
        num_classes=1,
        diversity_loss_weight=0.01,
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        moe_loss_weight=0.01,
        use_pre_cossm_moe=True,
        pre_cossm_num_experts=3,
        pre_cossm_top_k=2,
        pre_cossm_moe_loss_weight=0.01,
        modality_dropout_rate=0.15,
    )
    
    # Train mode (triggers modality dropout)
    model.train()
    output = model(x, mask, return_diversity_loss=True, return_moe_loss=True)
    div_loss = model.get_diversity_loss()
    moe_loss = model.get_moe_loss()
    pre_moe_loss = model.get_pre_moe_loss()
    total_aux = model.get_total_aux_loss()
    
    print(f"Output shape: {output.shape}")
    print(f"Diversity loss: {div_loss.item():.4f}")
    print(f"Post-CoSSM MoE loss: {moe_loss.item():.4f}")
    print(f"Pre-CoSSM FiLM MoE loss: {pre_moe_loss.item():.4f}")
    print(f"Total aux loss: {total_aux.item():.4f}")
    
    cls_loss = loss_fn(output, labels)
    total_loss = cls_loss + total_aux
    total_loss.backward()
    print(f"Backward pass OK, total loss: {total_loss.item():.4f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # ======================================================
    # Test 2: Missing modality (text = zero-vectors)
    # ======================================================
    print("\n" + "="*80)
    print("Test 2: Missing text modality (zero-vectors)")
    print("="*80)
    
    x_missing_text = x.clone()
    x_missing_text[:, :, visual_dim + audio_dim:] = 0.0
    
    model.eval()
    with torch.no_grad():
        output_missing = model(x_missing_text, mask)
    print(f"Output shape (missing text): {output_missing.shape}")
    print(f"Output mean: {output_missing.mean().item():.4f}")
    
    pre_moe_info = model.get_pre_moe_info()
    if pre_moe_info and len(pre_moe_info) > 0:
        enc0_info = pre_moe_info[0]
        if isinstance(enc0_info, dict):
            for mod, info in enc0_info.items():
                print(f"  {mod}: is_missing={info.get('is_missing', 'N/A')}, "
                      f"top_k_indices={info.get('top_k_indices', 'N/A')}")
    
    # ======================================================
    # Test 3: Eval mode (no modality dropout)
    # ======================================================
    print("\n" + "="*80)
    print("Test 3: Eval mode (no dropout)")
    print("="*80)
    
    model.eval()
    with torch.no_grad():
        output_eval = model(x, mask)
    print(f"Output shape (eval): {output_eval.shape}")
    print(f"Output mean: {output_eval.mean().item():.4f}")
    
    # ======================================================
    # Test 4: No Pre-CoSSM MoE
    # ======================================================
    print("\n" + "="*80)
    print("Test 4: Post-CoSSM MoE only (no Pre-CoSSM)")
    print("="*80)
    
    model_no_pre = DepMambaMultiEncoderEnsemble(
        num_encoders=2,
        audio_input_size=audio_dim,
        video_input_size=visual_dim,
        text_input_size=text_dim,
        mm_input_size=256,
        mm_output_sizes=[256, 64],
        d_ffn=1024,
        num_layers=2,
        mamba_config=mamba_config,
        ensemble_stage='late',
        fusion_type='weighted',
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        use_pre_cossm_moe=False,
        modality_dropout_rate=0.15,
    )
    
    model_no_pre.train()
    output_no_pre = model_no_pre(x, mask)
    print(f"Output shape: {output_no_pre.shape}")
    print(f"Pre-CoSSM MoE loss: {model_no_pre.get_pre_moe_loss().item():.4f} (should be 0)")
    
    total_params = sum(p.numel() for p in model_no_pre.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
