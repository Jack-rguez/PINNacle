"""
HPIT (Hybrid Physics-Informed Transformer) model implementation.

This module implements the core HPIT architecture that combines transformer
attention mechanisms with physics-informed constraints for snow water equivalent
forecasting.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import math

from .base import BaseTransformer, ModelConfig, ModelOutput

logger = logging.getLogger(__name__)

# Try to import flash attention for A100 optimization
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    logger.warning("Flash Attention not available. Install with: pip install flash-attn")

# Try to import fused operations for A100
try:
    from apex.normalization import FusedLayerNorm
    FUSED_OPS_AVAILABLE = True
except ImportError:
    FUSED_OPS_AVAILABLE = False
    logger.warning("Apex fused operations not available. Install with: pip install apex")


class SinActivation(nn.Module):
    """Custom sine activation function for seasonal patterns."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class SwishActivation(nn.Module):
    """Swish activation function (x * sigmoid(x)) - superior for many tasks."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class FlashMultiHeadAttention(nn.Module):
    """Flash Attention implementation for A100 GPU optimization."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.use_flash = FLASH_ATTENTION_AVAILABLE and torch.cuda.is_available()
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        
        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        if self.use_flash and self.training:
            # Use Flash Attention for training on A100
            try:
                attn_output = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0)
                attn_weights = None  # Flash attention doesn't return weights
            except Exception as e:
                logger.warning(f"Flash attention failed, falling back to standard attention: {e}")
                attn_output, attn_weights = self._standard_attention(q, k, v, attn_mask)
        else:
            # Standard attention for inference or when flash attention unavailable
            attn_output, attn_weights = self._standard_attention(q, k, v, attn_mask)
        
        # Reshape and project output
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output, attn_weights
    
    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard scaled dot-product attention."""
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back
        attn_output = attn_output.transpose(1, 2)  # (batch, seq_len, heads, head_dim)
        
        return attn_output, attn_weights.mean(dim=1)  # Average over heads for compatibility


class GPUOptimizedLayerNorm(nn.Module):
    """GPU-optimized LayerNorm using fused operations when available."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        if FUSED_OPS_AVAILABLE:
            self.norm = FusedLayerNorm(normalized_shape, eps=eps)
        else:
            self.norm = nn.LayerNorm(normalized_shape, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class AdaptiveFeatureSelection(nn.Module):
    """Adaptive feature selection with learned importance weights - GPU optimized."""

    def __init__(self, input_dim: int, selection_ratio: float = 0.8):
        super().__init__()
        self.input_dim = input_dim
        self.selection_ratio = selection_ratio
        
        # Feature importance network with GPU-optimized components
        self.importance_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),  # In-place for memory efficiency
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate feature importance
        importance = self.importance_net(x)
        
        # Apply gating
        gated_features = self.gate(x)
        
        # Combine with importance weighting (fused operation)
        return torch.addcmul(x * importance, gated_features, (1 - importance))


@dataclass
class HPITConfig(ModelConfig):
    """Configuration for HPIT model optimized for A100 GPU."""

    embedding_dim: int = 1024  # Larger for A100 capacity
    num_heads: int = 16  # More heads for A100
    num_layers: int = 12  # Deeper for A100
    hidden_dim: int = 4096  # Much larger for A100 memory
    num_attention_scales: int = 6  # More scales for A100
    physics_weight: float = 0.25  # Balanced physics influence
    temperature_scale: float = 1.0
    use_spatial_attention: bool = True
    use_temporal_attention: bool = True
    use_physics_layers: bool = True
    # Enhanced performance parameters
    use_residual_physics: bool = True
    use_feature_selection: bool = True
    use_batch_norm: bool = True
    use_layer_norm: bool = True
    activation: str = "swish"  # Superior activation for this domain
    use_gradient_checkpointing: bool = True  # Enable for A100 memory efficiency
    # Advanced performance improvements
    use_advanced_physics: bool = True
    use_ensemble_physics: bool = True
    physics_regularization: float = 0.003  # Reduced for larger model
    # Advanced features
    use_adaptive_attention: bool = True
    use_feature_interaction: bool = True
    use_domain_adaptation: bool = True
    use_multi_task_learning: bool = True
    feature_dropout: float = 0.06
    attention_dropout: float = 0.02
    # A100-specific optimizations
    use_flash_attention: bool = True
    use_fused_ops: bool = True
    compile_model: bool = True
    use_amp: bool = True  # Automatic Mixed Precision
    pde_name: str = ""  # selects PDE-specific physics sub-networks in PhysicsInformedLayer


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for capturing different temporal patterns."""

    def __init__(self, embedding_dim: int, num_scales: int = 3, num_heads: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_scales = num_scales
        self.num_heads = num_heads

        # Different attention scales (daily, weekly, seasonal)
        self.scale_factors = [1, 7, 30]  # days

        # Multi-head attention for each scale - use Flash Attention if available
        self.attention_layers = nn.ModuleList(
            [
                FlashMultiHeadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                ) if FLASH_ATTENTION_AVAILABLE else nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(num_scales)
            ]
        )

        # Scale-specific projections
        self.scale_projections = nn.ModuleList(
            [nn.Linear(embedding_dim, embedding_dim) for _ in range(num_scales)]
        )

        # Fusion layer
        self.fusion = nn.Linear(embedding_dim * num_scales, embedding_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-scale attention.

        Args:
            x: Input tensor (batch_size, seq_len, embedding_dim)
            mask: Attention mask

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        scale_outputs = []
        all_attention_weights = []

        for i, (attention, projection) in enumerate(
            zip(self.attention_layers, self.scale_projections)
        ):
            # Apply attention at this scale
            if isinstance(attention, FlashMultiHeadAttention):
                attn_output, attn_weights = attention(x, x, x, attn_mask=mask)
            else:
                attn_output, attn_weights = attention(x, x, x, attn_mask=mask)

            # Project and store
            scale_output = projection(attn_output)
            scale_outputs.append(scale_output)
            if attn_weights is not None:
                all_attention_weights.append(attn_weights)

        # Concatenate and fuse
        concatenated = torch.cat(scale_outputs, dim=-1)
        fused_output = self.fusion(concatenated)

        # Average attention weights across scales (if available)
        if all_attention_weights:
            avg_attention = torch.stack(all_attention_weights, dim=0).mean(dim=0)
        else:
            avg_attention = None

        return fused_output, avg_attention


class PhysicsInformedLayer(nn.Module):
    """Physics-informed layer with PDE-specific learned feature transformations.

    Each PDE family gets its own sub-networks that encode the PDE's structural
    physics (e.g. advection-diffusion balance for Burgers, Laplacian smoothing
    for Heat) as feature transformations on the embedded representation.
    These are learned approximators, not explicit PDE solvers.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, pde_name: str = ""):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pde_name = pde_name

        def _net():
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                GPUOptimizedLayerNorm(hidden_dim // 2),
                SwishActivation(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                GPUOptimizedLayerNorm(hidden_dim // 4),
                SwishActivation(),
                nn.Linear(hidden_dim // 4, 1),
            )

        def _fusion(n_components: int):
            return nn.Sequential(
                nn.Linear(n_components, hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                SwishActivation(),
                nn.Linear(hidden_dim // 4, 1),
            )

        if pde_name == "Burgers1D":
            # u_t + u*u_x - nu*u_xx = 0: nonlinear advection + diffusion
            self.advection_net = _net()
            self.diffusion_net = _net()
            self.physics_fusion = _fusion(2)
        elif pde_name == "Burgers2D":
            # vector: u1_t + u1*u1_x + u2*u1_y - nu*lap(u1) = 0, same for u2
            self.u_advection_net = _net()
            self.v_advection_net = _net()
            self.diffusion_net = _net()
            self.physics_fusion = _fusion(3)
        elif pde_name == "HeatComplexGeometry":
            # u_t - u_xx - u_yy = 0: isotropic Laplacian diffusion
            self.diffusion_x_net = _net()
            self.diffusion_y_net = _net()
            self.physics_fusion = _fusion(2)
        elif pde_name == "KuramotoSivashinsky":
            # u_t + alpha*u*u_x + beta*u_xx + gamma*u_xxxx = 0
            # advection (nonlinear), linear instability (u_xx), hyperdiffusion (u_xxxx)
            self.advection_net = _net()
            self.instability_net = _net()
            self.hyperdiffusion_net = _net()
            self.physics_fusion = _fusion(3)
        elif pde_name == "NavierStokes2D":
            # steady: momentum (velocity-pressure coupling), continuity (div-free), pressure
            self.momentum_net = _net()
            self.continuity_net = _net()
            self.pressure_net = _net()
            self.physics_fusion = _fusion(3)
        else:
            # Generic fallback: nonlinear feature + smoothing feature
            self.feature_net = _net()
            self.smoothing_net = _net()
            self.physics_fusion = _fusion(2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply PDE-specific physics feature transformations."""
        pde = self.pde_name

        if pde == "Burgers1D":
            adv = self.advection_net(x)
            dif = self.diffusion_net(x)
            return {"advection": adv, "diffusion": dif,
                    "physics_integrated": self.physics_fusion(torch.cat([adv, dif], dim=-1))}

        elif pde == "Burgers2D":
            u_adv = self.u_advection_net(x)
            v_adv = self.v_advection_net(x)
            dif   = self.diffusion_net(x)
            return {"u_advection": u_adv, "v_advection": v_adv, "diffusion": dif,
                    "physics_integrated": self.physics_fusion(torch.cat([u_adv, v_adv, dif], dim=-1))}

        elif pde == "HeatComplexGeometry":
            dx = self.diffusion_x_net(x)
            dy = self.diffusion_y_net(x)
            return {"diffusion_x": dx, "diffusion_y": dy,
                    "physics_integrated": self.physics_fusion(torch.cat([dx, dy], dim=-1))}

        elif pde == "KuramotoSivashinsky":
            adv  = self.advection_net(x)
            inst = self.instability_net(x)
            hyp  = self.hyperdiffusion_net(x)
            return {"advection": adv, "instability": inst, "hyperdiffusion": hyp,
                    "physics_integrated": self.physics_fusion(torch.cat([adv, inst, hyp], dim=-1))}

        elif pde == "NavierStokes2D":
            mom  = self.momentum_net(x)
            cont = self.continuity_net(x)
            pres = self.pressure_net(x)
            return {"momentum": mom, "continuity": cont, "pressure": pres,
                    "physics_integrated": self.physics_fusion(torch.cat([mom, cont, pres], dim=-1))}

        else:
            feat = self.feature_net(x)
            smth = self.smoothing_net(x)
            return {"feature": feat, "smoothing": smth,
                    "physics_integrated": self.physics_fusion(torch.cat([feat, smth], dim=-1))}


class HPITModel(BaseTransformer):
    """Hybrid Physics-Informed Transformer for SWE forecasting."""

    def __init__(self, config: HPITConfig):
        super().__init__(config)
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim

        # Build the model after setting all attributes
        self._build_model()

    def _build_model(self) -> None:
        """Build state-of-the-art HPIT architecture optimized for SWE prediction."""
        # Advanced feature selection with adaptive importance
        if self.config.use_feature_selection:
            self.adaptive_feature_selector = AdaptiveFeatureSelection(
                self.config.input_dim, selection_ratio=0.85
            )

        # Multi-layer input normalization with GPU optimization
        self.input_norm = GPUOptimizedLayerNorm(self.config.input_dim)
        self.feature_dropout = nn.Dropout(getattr(self.config, 'feature_dropout', 0.1))
            
        # Advanced input embedding with multiple pathways
        activation_fn = SwishActivation() if self.config.activation == "swish" else nn.GELU()
        
        self.input_embedding = nn.Sequential(
            nn.Linear(self.config.input_dim, self.hidden_dim),
            GPUOptimizedLayerNorm(self.hidden_dim),
            activation_fn,
            nn.Dropout(self.config.dropout * 0.5),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            GPUOptimizedLayerNorm(self.hidden_dim),
            activation_fn,
            nn.Dropout(self.config.dropout * 0.5),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )
        
        # Multiple residual projections for different feature types
        self.input_projection = nn.Linear(self.config.input_dim, self.embedding_dim)
        self.feature_projection = nn.Linear(self.config.input_dim, self.embedding_dim // 2)

        # Multi-scale attention mechanism
        if getattr(self.config, 'use_adaptive_attention', True):
            self.multi_scale_attention = MultiScaleAttention(
                self.embedding_dim, 
                num_scales=self.config.num_attention_scales,
                num_heads=self.num_heads
            )
        
        # GPU-optimized self-attention
        if getattr(self.config, 'use_flash_attention', True) and FLASH_ATTENTION_AVAILABLE:
            self.self_attention = FlashMultiHeadAttention(
                embed_dim=self.embedding_dim,
                num_heads=self.num_heads,
                dropout=getattr(self.config, 'attention_dropout', 0.05),
            )
        else:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=self.embedding_dim,
                num_heads=self.num_heads,
                dropout=getattr(self.config, 'attention_dropout', 0.05),
                batch_first=True,
            )

        # Advanced physics-informed processing
        if self.config.use_physics_layers:
            self.physics_layer = PhysicsInformedLayer(
                self.embedding_dim, hidden_dim=self.hidden_dim // 2,
                pde_name=getattr(self.config, 'pde_name', ''),
            )

        # Feature interaction networks
        if getattr(self.config, 'use_feature_interaction', True):
            self.feature_interaction = nn.Sequential(
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                activation_fn,
                nn.Linear(self.hidden_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
            )

        # Enhanced transformer-like layers with residual connections
        self.transformer_layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    embed_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    dropout=getattr(self.config, 'attention_dropout', 0.05),
                    batch_first=True,
                ),
                'norm1': nn.LayerNorm(self.embedding_dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.embedding_dim, self.hidden_dim),
                    activation_fn,
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    activation_fn,
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.hidden_dim, self.embedding_dim),
                ),
                'norm2': nn.LayerNorm(self.embedding_dim),
                'dropout': nn.Dropout(self.config.dropout),
            })
            self.transformer_layers.append(layer)

        # Advanced output processing with multiple heads
        self.output_norm = nn.LayerNorm(self.embedding_dim)
        
        # Main prediction head
        self.output_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            activation_fn,
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            activation_fn,
            nn.Dropout(self.config.dropout * 0.5),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            activation_fn,
            nn.Linear(self.hidden_dim // 4, self.config.output_dim),
        )
        
        # Uncertainty quantification head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim // 2),
            activation_fn,
            nn.Linear(self.hidden_dim // 2, self.config.output_dim),
            nn.Softplus(),  # Ensure positive uncertainty
        )
        
        # Auxiliary prediction heads for multi-task learning
        if getattr(self.config, 'use_multi_task_learning', True):
            self.aux_snow_depth_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.hidden_dim // 4),
                activation_fn,
                nn.Linear(self.hidden_dim // 4, 1),
            )
            
            self.aux_temperature_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.hidden_dim // 4),
                activation_fn,
                nn.Linear(self.hidden_dim // 4, 1),
            )

    def forward(
        self,
        x: torch.Tensor,
        spatial_features: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        **kwargs
    ) -> ModelOutput:
        """Forward pass through state-of-the-art HPIT model."""
        batch_size, seq_len, _ = x.shape
        
        # Reshape for batch processing if needed
        x_flat = x.view(-1, x.shape[-1])

        # Advanced feature selection and preprocessing
        if hasattr(self, 'adaptive_feature_selector'):
            x_flat = self.adaptive_feature_selector(x_flat)

        # Input normalization with dropout
        x_norm = self.input_norm(x_flat)
        x_norm = self.feature_dropout(x_norm)
        
        # Multi-pathway embedding
        x_embedded = self.input_embedding(x_norm)
        
        # Multiple residual connections
        x_residual = self.input_projection(x_flat)
        x_feature_residual = self.feature_projection(x_flat)
        
        # Combine embeddings
        x_embedded = x_embedded + x_residual
        if hasattr(self, 'feature_projection'):
            # Pad feature residual to match embedding dimension
            padding = self.embedding_dim - x_feature_residual.shape[-1]
            x_feature_residual = F.pad(x_feature_residual, (0, padding))
            x_embedded = x_embedded + 0.5 * x_feature_residual

        # Reshape back for attention
        x_embedded = x_embedded.view(batch_size, seq_len, -1)

        # Multi-scale attention (if enabled)
        attention_weights = None
        if hasattr(self, 'multi_scale_attention'):
            x_multi_scale, multi_attention = self.multi_scale_attention(x_embedded)
            x_embedded = x_embedded + 0.3 * x_multi_scale
            attention_weights = multi_attention

        # Self-attention for feature interactions
        x_attended, self_attention_weights = self.self_attention(
            x_embedded, x_embedded, x_embedded
        )

        # Combine attention weights
        if attention_weights is None:
            attention_weights = self_attention_weights

        # Residual connection
        x_attended = x_attended + x_embedded

        # Physics-informed processing
        physics_outputs = None
        if self.config.use_physics_layers and hasattr(self, 'physics_layer'):
            # Flatten for physics processing (reshape handles non-contiguous tensors)
            x_physics_flat = x_attended.reshape(-1, x_attended.shape[-1])
            physics_outputs = self.physics_layer(x_physics_flat)
            
            # Integrate physics
            physics_integrated = physics_outputs['physics_integrated']
            physics_integrated = physics_integrated.view(batch_size, seq_len, -1)
            
            # Add physics as residual
            x_attended = x_attended + self.config.physics_weight * physics_integrated

        # Feature interaction processing
        if hasattr(self, 'feature_interaction'):
            x_interaction = self.feature_interaction(x_attended)
            x_attended = x_attended + 0.2 * x_interaction

        # Advanced transformer layers
        for layer in self.transformer_layers:
            # Multi-head attention
            residual = x_attended
            x_attended = layer['norm1'](x_attended)
            attn_out, _ = layer['attention'](x_attended, x_attended, x_attended)
            x_attended = residual + layer['dropout'](attn_out)
            
            # Feed-forward network
            residual = x_attended
            x_attended = layer['norm2'](x_attended)
            ffn_out = layer['ffn'](x_attended)
            x_attended = residual + layer['dropout'](ffn_out)

        # Global representation (enhanced pooling)
        # Use both mean and max pooling for richer representation
        mean_pooled = x_attended.mean(dim=1)
        max_pooled, _ = x_attended.max(dim=1)
        final_hidden = mean_pooled + 0.3 * max_pooled
        
        final_hidden = self.output_norm(final_hidden)

        # Main predictions
        predictions = self.output_projection(final_hidden)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(final_hidden)

        # Auxiliary predictions for multi-task learning
        aux_predictions = {}
        if hasattr(self, 'aux_snow_depth_head'):
            aux_predictions['snow_depth'] = self.aux_snow_depth_head(final_hidden)
        if hasattr(self, 'aux_temperature_head'):
            aux_predictions['temperature'] = self.aux_temperature_head(final_hidden)

        # Prepare comprehensive metadata
        metadata = {
            "attention_weights": (
                attention_weights.detach().cpu().numpy()
                if attention_weights is not None
                else None
            ),
            "physics_outputs": physics_outputs,
            "aux_predictions": aux_predictions,
            "model_config": self.config.to_dict(),
            "final_hidden_stats": {
                "mean": final_hidden.mean().item(),
                "std": final_hidden.std().item(),
                "min": final_hidden.min().item(),
                "max": final_hidden.max().item(),
            }
        }

        return ModelOutput(
            predictions=predictions,
            uncertainty=uncertainty,
            attention_weights=attention_weights,
            hidden_states=final_hidden,
            metadata=metadata,
        )


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[: x.size(1), :].transpose(0, 1)