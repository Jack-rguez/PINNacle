"""
Base model classes and interfaces for SWE forecasting.

This module provides the foundational classes that all models inherit from,
ensuring consistent interfaces and common functionality across different
model architectures.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Base configuration class for all models."""
    input_dim: int
    output_dim: int = 1
    sequence_length: int = 365
    prediction_horizon: int = 1
    dropout: float = 0.1
    device: str = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "dropout": self.dropout,
            "device": self.device
        }


@dataclass
class ModelOutput:
    """Standard output format for all models."""
    predictions: torch.Tensor
    uncertainty: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert output to dictionary."""
        return {
            "predictions": self.predictions.detach().cpu().numpy(),
            "uncertainty": self.uncertainty.detach().cpu().numpy() if self.uncertainty is not None else None,
            "attention_weights": self.attention_weights.detach().cpu().numpy() if self.attention_weights is not None else None,
            "hidden_states": self.hidden_states.detach().cpu().numpy() if self.hidden_states is not None else None,
            "metadata": self.metadata or {}
        }


class BaseModel(nn.Module, ABC):
    """Abstract base class for all SWE forecasting models."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        self.to(self.device)
        
        # Initialize model components
        self._build_model()
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
    @abstractmethod
    def _build_model(self) -> None:
        """Build the model architecture. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            **kwargs: Additional model-specific arguments
            
        Returns:
            ModelOutput containing predictions and optional metadata
        """
        pass
    
    def predict(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        """
        Make predictions in evaluation mode.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            ModelOutput with predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, **kwargs)
    
    def get_model_size(self) -> int:
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, filepath: str) -> None:
        """Save model state dictionary."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model state dictionary."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")
    
    def freeze_parameters(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Model parameters frozen")
    
    def unfreeze_parameters(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Model parameters unfrozen")


class BaseTransformer(BaseModel):
    """Base class for transformer-based models."""
    
    def __init__(self, config: ModelConfig):
        # Don't call super().__init__ to avoid calling _build_model twice
        nn.Module.__init__(self)
        self.config = config
        self.device = torch.device(config.device)
        self.to(self.device)
        
        self.embedding_dim = getattr(config, 'embedding_dim', config.input_dim)
        self.num_heads = getattr(config, 'num_heads', 8)
        self.num_layers = getattr(config, 'num_layers', 6)
        self.hidden_dim = getattr(config, 'hidden_dim', 128)
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
    def _build_model(self) -> None:
        """Build transformer components."""
        # Input embedding
        self.input_embedding = nn.Linear(self.config.input_dim, self.embedding_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.embedding_dim, self.config.sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(self.embedding_dim, self.config.output_dim)
    
    def forward(self, x: torch.Tensor, **kwargs) -> ModelOutput:
        """Forward pass through transformer."""
        # Input embedding
        x_embedded = self.input_embedding(x)
        
        # Add positional encoding
        x_encoded = self.pos_encoding(x_embedded)
        
        # Transformer encoding
        transformer_output = self.transformer(x_encoded)
        
        # Get predictions (use last timestep for forecasting)
        predictions = self.output_projection(transformer_output[:, -1, :])
        
        return ModelOutput(predictions=predictions)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class ModelFactory:
    """Factory class for creating model instances."""
    
    _models = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register a model class."""
        cls._models[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def create_model(cls, name: str, config: ModelConfig) -> BaseModel:
        """Create a model instance by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available models: {list(cls._models.keys())}")
        
        model_class = cls._models[name]
        return model_class(config)
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered models."""
        return list(cls._models.keys())


# Register base models
ModelFactory.register_model("base", BaseModel)
ModelFactory.register_model("transformer", BaseTransformer)
