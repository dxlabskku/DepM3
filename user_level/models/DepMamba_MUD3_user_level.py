from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.DepMamba_multiencoder_ensemble import DepMambaMultiEncoderEnsemble

try:
    from models.mamba.mamba_blocks import MambaBlocksSequential
except ImportError:
    MambaBlocksSequential = None


class DepMambaMUD3UserLevel(nn.Module):
    """User-level depression detection for MUD3 dataset.

    Uses the existing DepMambaMultiEncoderEnsemble as a video-level feature
    extractor, then aggregates video representations per user with biGRU or BiMamba.
    """

    def __init__(
        self,
        # Video encoder params (passed to DepMambaMultiEncoderEnsemble)
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
        mamba_config: dict = None,
        ensemble_stage: str = 'late',
        fusion_type: str = 'weighted',
        diversity_loss_weight: float = 0.01,
        use_moe: bool = True,
        num_experts: int = 6,
        top_k_experts: int = 2,
        moe_loss_weight: float = 0.01,
        use_pre_cossm_moe: bool = True,
        pre_cossm_num_experts: int = 5,
        pre_cossm_top_k: int = 2,
        pre_cossm_moe_loss_weight: float = 0.01,
        modality_dropout_rate: float = 0.0,
        # User aggregator params
        user_aggregator: str = 'mamba',
        agg_hidden_dim: int = 128,
        agg_num_layers: int = 2,
        agg_dropout: float = 0.1,
        # Memory
        chunk_size: int = 64,
    ):
        super().__init__()

        self.video_input_size = video_input_size
        self.audio_input_size = audio_input_size
        self.text_input_size = text_input_size
        self.user_aggregator_type = user_aggregator
        self.chunk_size = chunk_size

        self.video_encoder = DepMambaMultiEncoderEnsemble(
            num_encoders=num_encoders,
            audio_input_size=audio_input_size,
            video_input_size=video_input_size,
            text_input_size=text_input_size,
            mm_input_size=mm_input_size,
            mm_output_sizes=mm_output_sizes,
            d_ffn=d_ffn,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            mamba_config=mamba_config,
            ensemble_stage=ensemble_stage,
            fusion_type=fusion_type,
            num_classes=1,
            diversity_loss_weight=diversity_loss_weight,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            moe_loss_weight=moe_loss_weight,
            use_pre_cossm_moe=use_pre_cossm_moe,
            pre_cossm_num_experts=pre_cossm_num_experts,
            pre_cossm_top_k=pre_cossm_top_k,
            pre_cossm_moe_loss_weight=pre_cossm_moe_loss_weight,
            modality_dropout_rate=modality_dropout_rate,
        )

        # feat_dim from video encoder: mm_output_sizes[-1] * 3 (trimodal concat)
        video_feat_dim = mm_output_sizes[-1] * 3  # e.g. 64 * 3 = 192

        self.projection = nn.Sequential(
            nn.Linear(video_feat_dim, agg_hidden_dim),
            nn.LayerNorm(agg_hidden_dim),
            nn.GELU(),
        )

        if user_aggregator == 'gru':
            self.user_agg = nn.GRU(
                input_size=agg_hidden_dim,
                hidden_size=agg_hidden_dim // 2,
                num_layers=agg_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=agg_dropout if agg_num_layers > 1 else 0.0,
            )
        elif user_aggregator == 'mamba':
            if MambaBlocksSequential is None:
                raise ImportError("MambaBlocksSequential not available")
            ssm_cfg = mamba_config or {}
            self.user_agg = MambaBlocksSequential(
                n_mamba=agg_num_layers,
                bidirectional=True,
                d_model=agg_hidden_dim,
                d_state=ssm_cfg.get('d_state', 16),
                expand=ssm_cfg.get('expand', 2),
                d_conv=ssm_cfg.get('d_conv', 4),
                fused_add_norm=False,
            )
        elif user_aggregator == 'mean':
            self.user_agg = None
        elif user_aggregator == 'attention':
            self.user_agg = nn.MultiheadAttention(
                embed_dim=agg_hidden_dim,
                num_heads=4,
                dropout=agg_dropout,
                batch_first=True,
            )
            self.attn_query = nn.Parameter(torch.randn(1, 1, agg_hidden_dim) * 0.02)
        elif user_aggregator == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=agg_hidden_dim,
                nhead=4,
                dim_feedforward=agg_hidden_dim * 4,
                dropout=agg_dropout,
                activation='gelu',
                batch_first=True,
            )
            self.user_agg = nn.TransformerEncoder(encoder_layer, num_layers=agg_num_layers)
        else:
            raise ValueError(f"Unknown user_aggregator: {user_aggregator}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(agg_hidden_dim),
            nn.Dropout(agg_dropout),
            nn.Linear(agg_hidden_dim, 1),
        )

    def _prepare_input(self, visual, acoustic):
        """Concat visual+acoustic and zero-pad text dimension.

        Args:
            visual:   (B, N, T, 136)
            acoustic: (B, N, T, 25)
        Returns:
            x: (B, N, T, 929)  where 929 = 136 + 25 + 768(zeros)
        """
        B, N, T, _ = visual.shape
        va = torch.cat([visual, acoustic], dim=-1)
        text_pad = torch.zeros(
            B, N, T, self.text_input_size,
            device=visual.device, dtype=visual.dtype
        )
        return torch.cat([va, text_pad], dim=-1)

    def _extract_video_features(self, x, return_moe_loss=True):
        """Run video encoder on all videos with chunking for memory efficiency.

        Args:
            x: (B*N, T, F) — all videos flattened into batch dim
        Returns:
            video_reprs: (B*N, feat_dim) — pooled per-video representations
        """
        total = x.shape[0]
        T = x.shape[1]

        frame_mask = (x.abs().sum(dim=-1) != 0).long()

        if total <= self.chunk_size:
            encoder_features = self.video_encoder.feature_extractor(
                x, frame_mask, return_moe_loss=return_moe_loss
            )
            video_reprs = torch.stack(encoder_features, dim=0).mean(dim=0)
            return video_reprs

        all_reprs = []
        for start in range(0, total, self.chunk_size):
            end = min(start + self.chunk_size, total)
            chunk_x = x[start:end]
            chunk_mask = frame_mask[start:end]
            enc_feats = self.video_encoder.feature_extractor(
                chunk_x, chunk_mask, return_moe_loss=(return_moe_loss and start == 0)
            )
            chunk_repr = torch.stack(enc_feats, dim=0).mean(dim=0)
            all_reprs.append(chunk_repr)

        return torch.cat(all_reprs, dim=0)

    def forward(self, visual, acoustic, mask,
                return_diversity_loss=True, return_moe_loss=True):
        """
        Args:
            visual:   (B, N, 44, 136)
            acoustic: (B, N, 44, 25)
            mask:     (B, N) — 1 for real videos, 0 for padding
            return_diversity_loss: compute diversity loss
            return_moe_loss: compute MoE load balancing loss
        Returns:
            logits: (B, 1)
        """
        B, N, T, _ = visual.shape

        x = self._prepare_input(visual, acoustic)
        x = x.view(B * N, T, -1)

        real_mask = mask.view(B * N).bool()
        real_x = x[real_mask]

        if real_x.shape[0] > 0:
            real_reprs = self._extract_video_features(real_x, return_moe_loss=return_moe_loss)
            feat_dim = real_reprs.shape[-1]
            video_reprs = torch.zeros(B * N, feat_dim, device=x.device, dtype=x.dtype)
            video_reprs[real_mask] = real_reprs
        else:
            feat_dim = self.projection[0].normalized_shape[0] if hasattr(self.projection[0], 'normalized_shape') else self.projection[-1].in_features
            video_reprs = torch.zeros(B * N, feat_dim, device=x.device, dtype=x.dtype)

        video_reprs = video_reprs.view(B, N, -1)

        if return_diversity_loss:
            enc_feats_for_div = [video_reprs]
            self.video_encoder.last_diversity_loss = \
                self.video_encoder.compute_diversity_loss(enc_feats_for_div)

        video_reprs = self.projection(video_reprs)

        if self.user_aggregator_type == 'gru':
            lengths = mask.sum(dim=1).cpu().clamp(min=1)
            packed = pack_padded_sequence(
                video_reprs, lengths, batch_first=True, enforce_sorted=False
            )
            out, _ = self.user_agg(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)

            idx = (lengths - 1).long().to(visual.device)
            idx = idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(-1))
            user_repr = out.gather(1, idx).squeeze(1)

        elif self.user_aggregator_type == 'mamba':
            out = self.user_agg(video_reprs)
            mask_expanded = mask.unsqueeze(-1).float()
            out = out * mask_expanded
            user_repr = out.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)

        elif self.user_aggregator_type == 'mean':
            mask_expanded = mask.unsqueeze(-1).float()
            user_repr = (video_reprs * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)

        elif self.user_aggregator_type == 'attention':
            B = video_reprs.shape[0]
            query = self.attn_query.expand(B, -1, -1)
            key_padding_mask = ~mask.bool()
            attn_out, _ = self.user_agg(query, video_reprs, video_reprs,
                                         key_padding_mask=key_padding_mask)
            user_repr = attn_out.squeeze(1)

        elif self.user_aggregator_type == 'transformer':
            src_key_padding_mask = ~mask.bool()
            out = self.user_agg(video_reprs, src_key_padding_mask=src_key_padding_mask)
            mask_expanded = mask.unsqueeze(-1).float()
            user_repr = (out * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)

        logits = self.classifier(user_repr)
        return logits

    # --- Delegate auxiliary loss methods to video_encoder ---
    def get_diversity_loss(self):
        return self.video_encoder.get_diversity_loss()

    def get_moe_loss(self):
        return self.video_encoder.get_moe_loss()

    def get_pre_moe_loss(self):
        return self.video_encoder.get_pre_moe_loss()

    def get_total_moe_loss(self):
        return self.video_encoder.get_total_moe_loss()

    def get_total_aux_loss(self):
        return self.video_encoder.get_total_aux_loss()

    def get_ensemble_weights(self):
        return self.video_encoder.get_ensemble_weights()

    def get_ensemble_info(self):
        return self.video_encoder.get_ensemble_info()

    def get_moe_info(self):
        return self.video_encoder.get_moe_info()

    def get_pre_moe_info(self):
        return self.video_encoder.get_pre_moe_info()
