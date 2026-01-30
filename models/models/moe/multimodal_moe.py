"""Multimodal Mixture of Experts (MoE) implementation.
Designed for zero-shot modality adaptation with Audio, Visual, and Text inputs.

Authors
-------
* AI Assistant 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ModalityExpert(nn.Module):
    """Expert network specialized for a specific modality."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, seq_len, output_dim)
        """
        return self.network(x)


class ModalityAwareGating(nn.Module):
    """Gating network that detects which modalities are present and routes accordingly.
    
    This gating mechanism can handle zero-shot scenarios where some modalities might be missing.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_experts: int,
        modality_dims: Dict[str, int],
        temperature: float = 1.0,
        top_k: int = 2
    ):
        """
        Args:
            input_dim: Common dimension after projection
            num_experts: Number of expert networks
            modality_dims: Dictionary mapping modality names to their feature dimensions
            temperature: Temperature for softmax (lower = more selective)
            top_k: Number of top experts to select (sparse MoE)
        """
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.top_k = top_k
        self.modality_dims = modality_dims
        
        # Modality detection networks
        self.modality_detectors = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            self.modality_detectors[modality] = nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        # Gating network that considers all modalities
        total_modality_features = len(modality_dims) + input_dim
        self.gate_network = nn.Sequential(
            nn.Linear(total_modality_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_experts)
        )
        
    def detect_modalities(
        self, 
        modality_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Detect which modalities are present and their confidence scores.
        
        Args:
            modality_features: Dict of raw modality features before projection
            
        Returns:
            modality_vector: (batch, num_modalities) binary presence indicators
            modality_scores: Dict of confidence scores per modality
        """
        batch_size = next(iter(modality_features.values())).shape[0]
        modality_scores = {}
        modality_presence = []
        
        for modality_name in self.modality_dims.keys():
            if modality_name in modality_features:
                feat = modality_features[modality_name]
                # Average pool over sequence dimension
                feat_pooled = feat.mean(dim=1)  # (batch, dim)
                score = self.modality_detectors[modality_name](feat_pooled)  # (batch, 1)
                modality_scores[modality_name] = score
                modality_presence.append(score)
            else:
                # Modality not present (zero-shot scenario)
                modality_scores[modality_name] = torch.zeros(batch_size, 1, device=feat.device)
                modality_presence.append(torch.zeros(batch_size, 1, device=feat.device))
        
        modality_vector = torch.cat(modality_presence, dim=1)  # (batch, num_modalities)
        return modality_vector, modality_scores
    
    def forward(
        self, 
        x: torch.Tensor,
        modality_features: Dict[str, torch.Tensor],
        use_top_k: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute gating weights for experts based on input and modality presence.
        
        Args:
            x: Fused features (batch, seq_len, input_dim)
            modality_features: Dict of raw modality features
            use_top_k: Whether to use top-k sparse gating
            
        Returns:
            gate_weights: (batch, num_experts) - weights for each expert
            gate_logits: (batch, num_experts) - raw logits before softmax
            modality_scores: Dict of modality presence scores
        """
        batch_size = x.shape[0]
        
        # Detect modalities
        modality_vector, modality_scores = self.detect_modalities(modality_features)
        
        # Pool input features
        x_pooled = x.mean(dim=1)  # (batch, input_dim)
        
        # Concatenate pooled features with modality presence vector
        gate_input = torch.cat([x_pooled, modality_vector], dim=1)
        
        # Compute gate logits
        gate_logits = self.gate_network(gate_input)  # (batch, num_experts)
        
        if use_top_k and self.top_k < self.num_experts:
            # Top-k sparse gating
            top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)
            gate_weights = torch.zeros_like(gate_logits)
            gate_weights.scatter_(1, top_k_indices, F.softmax(top_k_logits / self.temperature, dim=1))
        else:
            # Dense gating
            gate_weights = F.softmax(gate_logits / self.temperature, dim=1)
        
        return gate_weights, gate_logits, modality_scores


class MultimodalMoE(nn.Module):
    """Multimodal Mixture of Experts layer with zero-shot capability.
    
    This MoE can handle scenarios where some modalities are missing during inference.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 6,
        modality_dims: Dict[str, int] = None,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balance_loss_weight: float = 0.1,
        load_balance_loss_type: str = 'combined'
    ):
        """
        Args:
            input_dim: Input feature dimension (common dimension after projection)
            hidden_dim: Hidden dimension for expert networks
            output_dim: Output dimension
            num_experts: Number of expert networks
            modality_dims: Dict mapping modality names to their feature dimensions
            top_k: Number of experts to activate per input
            dropout: Dropout rate
            load_balance_loss_weight: Weight for load balancing auxiliary loss
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_loss_weight = load_balance_loss_weight
        self.load_balance_loss_type = load_balance_loss_type
        
        if modality_dims is None:
            modality_dims = {
                'audio': 161,
                'visual': 136,
                'text': 768
            }
        
        # Create expert networks
        self.experts = nn.ModuleList([
            ModalityExpert(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = ModalityAwareGating(
            input_dim=input_dim,
            num_experts=num_experts,
            modality_dims=modality_dims,
            top_k=top_k
        )
        
        # Initialize gating network to encourage uniform distribution
        self._initialize_gating_network()
        
        # Optional: Expert specialization indicators (for interpretability)
        self.expert_specialization = nn.Parameter(
            torch.randn(num_experts, len(modality_dims))
        )
    
    def _initialize_gating_network(self):
        """Initialize gating network to encourage uniform expert usage."""
        # Initialize the final layer with small random values to avoid initial bias
        if hasattr(self.gate, 'gate_network'):
            # Find the last linear layer (output layer)
            for i in range(len(self.gate.gate_network) - 1, -1, -1):
                layer = self.gate.gate_network[i]
                if isinstance(layer, nn.Linear) and layer.out_features == self.num_experts:
                    # Xavier uniform initialization with small scale
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    # Initialize bias to encourage uniform distribution
                    nn.init.constant_(layer.bias, 0.0)
                    break
    
    def compute_load_balance_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage uniform expert usage.
        
        Supports multiple loss types:
        - 'kl': KL divergence (original)
        - 'variance': Variance-based loss (penalizes imbalance)
        - 'combined': KL + Variance (recommended)
        - 'entropy': Entropy maximization
        
        Args:
            gate_logits: (batch, num_experts)
            
        Returns:
            loss: Scalar load balance loss
        """
        # Compute the fraction of inputs routed to each expert
        expert_usage = F.softmax(gate_logits, dim=1).mean(dim=0)  # (num_experts,)
        uniform_rate = 1.0 / self.num_experts
        
        if self.load_balance_loss_type == 'kl':
            # Original KL divergence
            uniform_dist = torch.ones_like(expert_usage) / self.num_experts
            loss = F.kl_div(
                expert_usage.log(),
                uniform_dist,
                reduction='batchmean'
            )
        
        elif self.load_balance_loss_type == 'variance':
            # Variance-based loss: penalize deviation from uniform
            loss = torch.var(expert_usage) / (uniform_rate ** 2)
        
        elif self.load_balance_loss_type == 'combined':
            # Combined: KL divergence + variance penalty
            uniform_dist = torch.ones_like(expert_usage) / self.num_experts
            kl_loss = F.kl_div(
                expert_usage.log(),
                uniform_dist,
                reduction='batchmean'
            )
            variance_loss = torch.var(expert_usage) / (uniform_rate ** 2)
            loss = kl_loss + 0.5 * variance_loss
        
        elif self.load_balance_loss_type == 'entropy':
            # Entropy maximization (negative entropy)
            entropy = -torch.sum(expert_usage * torch.log(expert_usage + 1e-10))
            max_entropy = torch.log(torch.tensor(self.num_experts, dtype=torch.float32, device=expert_usage.device))
            loss = 1.0 - (entropy / max_entropy)
        
        else:
            # Default to KL divergence
            uniform_dist = torch.ones_like(expert_usage) / self.num_experts
            loss = F.kl_div(
                expert_usage.log(),
                uniform_dist,
                reduction='batchmean'
            )
        
        return loss * self.load_balance_loss_weight
    
    def forward(
        self, 
        x: torch.Tensor,
        modality_features: Dict[str, torch.Tensor],
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input features (batch, seq_len, input_dim)
            modality_features: Dict of raw modality features for gating
            return_aux_loss: Whether to return auxiliary losses
            
        Returns:
            output: (batch, seq_len, output_dim)
            aux_loss: Auxiliary loss (load balancing) if return_aux_loss=True
            info: Dictionary with routing information
        """
        batch_size, seq_len, _ = x.shape
        
        # Get gating weights
        gate_weights, gate_logits, modality_scores = self.gate(x, modality_features)
        # gate_weights: (batch, num_experts)
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # (batch, seq_len, output_dim)
            expert_outputs.append(expert_out)
        
        # Stack expert outputs: (num_experts, batch, seq_len, output_dim)
        expert_outputs = torch.stack(expert_outputs, dim=0)
        
        # Weighted combination of expert outputs
        # gate_weights: (batch, num_experts) -> (batch, num_experts, 1, 1)
        gate_weights_expanded = gate_weights.unsqueeze(-1).unsqueeze(-1)
        
        # expert_outputs: (num_experts, batch, seq_len, output_dim)
        # Permute to: (batch, num_experts, seq_len, output_dim)
        expert_outputs = expert_outputs.permute(1, 0, 2, 3)
        
        # Weighted sum: (batch, seq_len, output_dim)
        output = (expert_outputs * gate_weights_expanded).sum(dim=1)
        
        # Compute auxiliary loss
        aux_loss = None
        if return_aux_loss:
            aux_loss = self.compute_load_balance_loss(gate_logits)
        
        # Collect routing information
        info = {
            'gate_weights': gate_weights,
            'gate_logits': gate_logits,
            'modality_scores': modality_scores,
            'expert_specialization': self.expert_specialization
        }
        
        return output, aux_loss, info


class MultimodalMoEBlock(nn.Module):
    """Complete MoE block with residual connection and layer norm."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 6,
        modality_dims: Dict[str, int] = None,
        top_k: int = 2,
        dropout: float = 0.1,
        load_balance_loss_weight: float = 0.1,
        load_balance_loss_type: str = 'combined'
    ):
        super().__init__()
        self.moe = MultimodalMoE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            modality_dims=modality_dims,
            top_k=top_k,
            dropout=dropout,
            load_balance_loss_weight=load_balance_loss_weight,
            load_balance_loss_type=load_balance_loss_type
        )
        self.norm = nn.LayerNorm(output_dim)
        
        # Projection if dimensions don't match
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = None
    
    def forward(
        self, 
        x: torch.Tensor,
        modality_features: Dict[str, torch.Tensor],
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Args:
            x: (batch, seq_len, input_dim)
            modality_features: Dict of raw modality features
            return_aux_loss: Whether to return auxiliary loss
            
        Returns:
            output: (batch, seq_len, output_dim)
            aux_loss: Auxiliary loss if return_aux_loss=True
            info: Routing information
        """
        # MoE forward
        moe_out, aux_loss, info = self.moe(x, modality_features, return_aux_loss)
        
        # Residual connection
        if self.projection is not None:
            x = self.projection(x)
        
        output = self.norm(x + moe_out)
        
        return output, aux_loss, info

