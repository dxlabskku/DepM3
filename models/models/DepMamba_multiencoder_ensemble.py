"""DepMamba Multi-Encoder Ensemble Model with MoE for Multimodal Depression Detection.
Multiple multimodal encoders with MoE trained end-to-end with ensemble fusion.

This model contains multiple TriModalMoE encoders that process the same
multimodal input (Audio + Visual + Text) and ensembles their outputs.

Each encoder includes:
- TriModalCoSSM: Collaborative SSM for 3 modalities
- EnSSM_MoE: Enhanced SSM with Mixture of Experts

Authors
-------
* Based on DepMamba by Jiaxin Ye 2024
* Multi-Encoder Ensemble with MoE by AI Assistant 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from speechbrain.nnet.activations import Swish
from speechbrain.nnet.normalization import LayerNorm

from mamba_ssm import Mamba
from .mamba.bimamba import Mamba as BiMamba
from .mamba.trimodal_mamba import TriModalMamba
from .moe import MultimodalMoEBlock
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
        # MoE parameters
        use_moe=True,
        num_experts=6,
        top_k_experts=2,
        moe_loss_weight=0.01,
    ):
        super().__init__()
        
        self.audio_input_size = audio_input_size
        self.video_input_size = video_input_size
        self.text_input_size = text_input_size
        self.use_moe = use_moe
        
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
        
        # Store MoE loss
        self.last_moe_loss = None
        self.last_moe_info = None
        
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
        
        # Project to common dimension
        xa = self.conv_audio(xa.permute(0, 2, 1)).permute(0, 2, 1)
        xv = self.conv_video(xv.permute(0, 2, 1)).permute(0, 2, 1)
        xt = self.conv_text(xt.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Collaborative SSM processing
        xa, xv, xt = self.cossm_encoder(xa, xv, xt)
        
        # Concatenate all 3 modalities
        x = torch.cat([xa, xv, xt], dim=-1)
        
        # Enhanced SSM processing with MoE
        x, moe_loss, moe_info = self.enssm_encoder(
            x,
            modality_features=modality_features,
            return_moe_loss=return_moe_loss and self.use_moe
        )
        
        # Store MoE loss and info
        if return_moe_loss and self.use_moe:
            self.last_moe_loss = moe_loss
            self.last_moe_info = moe_info
        
        return x
    
    def get_moe_loss(self):
        """Get MoE loss from last forward pass."""
        return self.last_moe_loss if self.last_moe_loss is not None else torch.tensor(0.0)
    
    def get_moe_info(self):
        """Get MoE routing info from last forward pass."""
        return self.last_moe_info


class EnsembleHead(nn.Module):
    """Ensemble fusion head that combines multiple encoder outputs (Early Fusion).
    
    Early Fusion: Feature 레벨에서 fusion 후 단일 classifier 사용
    
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
    """Late Ensemble Head - Classification 직전에 각 encoder별 개별 classifier 후 ensemble.
    
    Late Fusion: 각 encoder가 개별 classifier를 가지고, logits/predictions를 ensemble
    
    이 방식의 장점:
    - 각 encoder가 독립적으로 학습하여 다양성 확보
    - 전통적인 ensemble 방식으로 해석이 용이
    - 개별 encoder 성능 분석 가능
    
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
        
        # 각 encoder별 개별 classifier 생성
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
            # Meta-learner: 모든 encoder의 logits를 입력으로 받아 최종 예측
            self.meta_learner = nn.Sequential(
                nn.Linear(num_encoders * num_classes, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, num_classes)
            )
            
        elif ensemble_type == 'gating':
            # Gating network: feature 기반으로 각 classifier의 가중치 결정
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
        
        # 각 encoder별 개별 classification
        individual_logits = []
        for i, (feat, classifier) in enumerate(zip(encoder_features, self.classifiers)):
            logit = classifier(feat)  # (B, num_classes)
            individual_logits.append(logit)
        
        # Stack logits: (num_encoders, B, num_classes)
        stacked_logits = torch.stack(individual_logits, dim=0)
        
        # 개별 predictions 저장 (분석용)
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
    - Early Ensemble: Feature fusion 후 단일 classifier (기존 방식)
    - Late Ensemble: 개별 classifier 후 logits ensemble (새로운 방식)
    
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
        # MoE parameters
        use_moe: bool = True,
        num_experts: int = 6,
        top_k_experts: int = 2,
        moe_loss_weight: float = 0.01,
    ):
        super().__init__()
        
        self.num_encoders = num_encoders
        self.ensemble_stage = ensemble_stage
        self.fusion_type = fusion_type
        self.diversity_loss_weight = diversity_loss_weight
        self.num_classes = num_classes
        self.use_moe = use_moe
        self.moe_loss_weight = moe_loss_weight
        
        # Create multiple multimodal encoders with MoE
        self.encoders = nn.ModuleList()
        for i in range(num_encoders):
            if use_moe:
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
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k_experts=top_k_experts,
                    moe_loss_weight=moe_loss_weight,
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
            # Late Ensemble: 각 encoder별 개별 classifier 후 logits ensemble
            self.ensemble_head = LateEnsembleHead(
                num_encoders=num_encoders,
                feature_dim=feature_dim,
                ensemble_type=fusion_type,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            # Early Ensemble: Feature fusion 후 단일 classifier (기존 방식)
            self.ensemble_head = EnsembleHead(
                num_encoders=num_encoders,
                feature_dim=feature_dim,
                fusion_type=fusion_type,
                num_classes=num_classes,
            )
        
        # Store losses
        self.last_diversity_loss = None
        self.last_moe_loss = None
        self.last_ensemble_info = None
        self.last_moe_info = None
        
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
        moe_info_list = []
        
        for encoder in self.encoders:
            # Get encoder output
            if self.use_moe:
                enc_out = encoder(x, padding_mask, return_moe_loss=return_moe_loss)
                if return_moe_loss:
                    moe_loss = encoder.get_moe_loss()
                    if moe_loss is not None and isinstance(moe_loss, torch.Tensor):
                        total_moe_loss = total_moe_loss + moe_loss
                    moe_info_list.append(encoder.get_moe_info())
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
        if return_moe_loss and self.use_moe:
            self.last_moe_loss = total_moe_loss
            self.last_moe_info = moe_info_list
        
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
        """Get MoE load balancing loss from last forward pass."""
        if self.last_moe_loss is not None:
            if isinstance(self.last_moe_loss, torch.Tensor):
                return self.last_moe_loss
            return torch.tensor(self.last_moe_loss)
        return torch.tensor(0.0)
    
    def get_total_aux_loss(self) -> torch.Tensor:
        """Get total auxiliary loss (diversity + MoE)."""
        diversity_loss = self.get_diversity_loss()
        moe_loss = self.get_moe_loss()
        
        # Ensure both are tensors on the same device
        if isinstance(diversity_loss, torch.Tensor) and isinstance(moe_loss, torch.Tensor):
            if diversity_loss.device != moe_loss.device:
                moe_loss = moe_loss.to(diversity_loss.device)
        
        return diversity_loss + moe_loss
    
    def get_ensemble_weights(self) -> Optional[torch.Tensor]:
        """Get ensemble weights."""
        return self.ensemble_head.get_ensemble_weights()
    
    def get_ensemble_info(self) -> Optional[Dict]:
        """Get ensemble information from last forward pass."""
        return self.last_ensemble_info
    
    def get_moe_info(self) -> Optional[List]:
        """Get MoE routing information from last forward pass."""
        return self.last_moe_info
    
    def get_individual_predictions(self, x, padding_mask=None) -> List[torch.Tensor]:
        """Get individual predictions from each encoder.
        
        Useful for analysis of encoder contributions.
        
        For Late Ensemble: Uses the individual classifiers
        For Early Ensemble: Uses a temporary shared classifier
        """
        encoder_features = self.feature_extractor(x, padding_mask)
        
        if self.ensemble_stage == 'late':
            # Late Ensemble: 개별 classifier 사용
            return self.ensemble_head.get_individual_predictions(encoder_features)
        else:
            # Early Ensemble: 임시 공유 classifier 사용
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
    # Test the model
    batch_size = 8
    seq_len = 100
    
    # Input dimensions for D-Vlog
    visual_dim = 136
    audio_dim = 25
    text_dim = 768
    total_dim = visual_dim + audio_dim + text_dim
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, total_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, 2, (batch_size,)).float().unsqueeze(1)
    
    # Mamba config
    mamba_config = {
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'bidirectional': True
    }
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    print("="*80)
    print("Testing MoE + ENSEMBLE Model")
    print("="*80)
    
    # Test with MoE + Late Ensemble
    print("\n" + "="*80)
    print("Testing MoE + LATE ENSEMBLE")
    print("="*80)
    
    late_fusion_types = ['weighted', 'attention', 'gating']
    for fusion_type in late_fusion_types:
        print(f"\n{'-'*60}")
        print(f"MoE + Late Ensemble - {fusion_type.upper()}")
        print(f"{'-'*60}")
        
        model = DepMambaMultiEncoderEnsemble(
            num_encoders=3,
            audio_input_size=audio_dim,
            video_input_size=visual_dim,
            text_input_size=text_dim,
            mm_input_size=256,
            mm_output_sizes=[256, 64],
            d_ffn=1024,
            num_layers=2,
            mamba_config=mamba_config,
            ensemble_stage='late',
            fusion_type=fusion_type,
            num_classes=1,
            diversity_loss_weight=0.01,
            # MoE parameters
            use_moe=True,
            num_experts=6,
            top_k_experts=2,
            moe_loss_weight=0.01,
        )
        
        # Forward pass
        output = model(x, mask)
        diversity_loss = model.get_diversity_loss()
        moe_loss = model.get_moe_loss()
        total_aux_loss = model.get_total_aux_loss()
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"MoE enabled: {model.use_moe}")
        print(f"Diversity loss: {diversity_loss.item():.4f}")
        print(f"MoE loss: {moe_loss.item():.4f}")
        print(f"Total aux loss: {total_aux_loss.item():.4f}")
        
        # Get ensemble info
        info = model.get_ensemble_info()
        if info:
            if 'individual_logits' in info:
                print(f"Individual logits count: {len(info['individual_logits'])}")
            if 'ensemble_weights' in info:
                print(f"Ensemble weights: {info['ensemble_weights'].numpy()}")
        
        # Get MoE info
        moe_info = model.get_moe_info()
        if moe_info:
            print(f"MoE info count (per encoder): {len(moe_info)}")
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Test training
        classification_loss = loss_fn(output, labels)
        total_loss = classification_loss + total_aux_loss
        print(f"Classification loss: {classification_loss.item():.4f}")
        print(f"Total loss: {total_loss.item():.4f}")
    
    # Test with MoE + Early Ensemble
    print("\n" + "="*80)
    print("Testing MoE + EARLY ENSEMBLE")
    print("="*80)
    
    early_fusion_types = ['weighted', 'attention', 'concat']
    for fusion_type in early_fusion_types:
        print(f"\n{'-'*60}")
        print(f"MoE + Early Ensemble - {fusion_type.upper()}")
        print(f"{'-'*60}")
        
        model = DepMambaMultiEncoderEnsemble(
            num_encoders=3,
            audio_input_size=audio_dim,
            video_input_size=visual_dim,
            text_input_size=text_dim,
            mm_input_size=256,
            mm_output_sizes=[256, 64],
            d_ffn=1024,
            num_layers=2,
            mamba_config=mamba_config,
            ensemble_stage='early',
            fusion_type=fusion_type,
            num_classes=1,
            diversity_loss_weight=0.01,
            # MoE parameters
            use_moe=True,
            num_experts=6,
            top_k_experts=2,
            moe_loss_weight=0.01,
        )
        
        # Forward pass
        output = model(x, mask)
        diversity_loss = model.get_diversity_loss()
        moe_loss = model.get_moe_loss()
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"MoE enabled: {model.use_moe}")
        print(f"Diversity loss: {diversity_loss.item():.4f}")
        print(f"MoE loss: {moe_loss.item():.4f}")
        
        # Get ensemble weights if available
        weights = model.get_ensemble_weights()
        if weights is not None:
            print(f"Ensemble weights: {weights.detach().numpy()}")
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Test training
        classification_loss = loss_fn(output, labels)
        total_loss = classification_loss + diversity_loss + moe_loss
        print(f"Classification loss: {classification_loss.item():.4f}")
        print(f"Total loss: {total_loss.item():.4f}")
    
    # Test without MoE (ensemble only)
    print("\n" + "="*80)
    print("Testing ENSEMBLE ONLY (no MoE)")
    print("="*80)
    
    model = DepMambaMultiEncoderEnsemble(
        num_encoders=3,
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
        use_moe=False,  # No MoE
    )
    
    output = model(x, mask)
    diversity_loss = model.get_diversity_loss()
    moe_loss = model.get_moe_loss()
    
    print(f"MoE enabled: {model.use_moe}")
    print(f"Output shape: {output.shape}")
    print(f"Diversity loss: {diversity_loss.item():.4f}")
    print(f"MoE loss: {moe_loss.item():.4f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
