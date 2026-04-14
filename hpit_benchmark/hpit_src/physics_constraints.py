"""
Physics-informed constraints and loss functions for SWE forecasting.

This module implements physics-based constraints that ensure model predictions
follow fundamental snow hydrology principles including mass balance, energy
balance, and temperature gradients.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class PhysicsConstraints:
    """Main class for applying physics constraints to SWE predictions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.water_density = 1000.0  # kg/m³
        self.snow_density = 300.0    # kg/m³ (typical)
        self.latent_heat_fusion = 334000.0  # J/kg
        self.specific_heat_water = 4180.0   # J/(kg·K)
        self.specific_heat_ice = 2100.0     # J/(kg·K)
        self.stefan_boltzmann = 5.67e-8     # W/(m²·K⁴)
        
    def mass_balance_constraint(self, swe_pred: torch.Tensor, 
                               precipitation: torch.Tensor,
                               melt: torch.Tensor,
                               sublimation: torch.Tensor) -> torch.Tensor:
        """Mass balance constraint: dSWE/dt = P - M - S"""
        swe_change = torch.diff(swe_pred, dim=1)
        mass_balance = swe_change - (precipitation[:, 1:] - melt[:, 1:] - sublimation[:, 1:])
        return torch.mean(torch.abs(mass_balance))
    
    def energy_balance_constraint(self, temperature: torch.Tensor,
                                 radiation: torch.Tensor,
                                 wind_speed: torch.Tensor,
                                 humidity: torch.Tensor) -> torch.Tensor:
        """Energy balance constraint for snow surface."""
        sensible_heat = 1.2 * 1005 * wind_speed * (temperature - 273.15)
        latent_heat = 2.5e6 * wind_speed * (0.622 * humidity / 101325)
        net_energy = radiation - sensible_heat - latent_heat
        return torch.mean(torch.abs(net_energy))
    
    def temperature_gradient_constraint(self, temperature: torch.Tensor,
                                      elevation: torch.Tensor) -> torch.Tensor:
        """Temperature gradient constraint (lapse rate)."""
        expected_lapse_rate = -6.5e-3  # K/m
        
        if elevation.dim() > 1:
            elevation_diff = torch.diff(elevation, dim=1)
            temp_diff = torch.diff(temperature, dim=1)
        else:
            elevation_diff = elevation[1:] - elevation[:-1]
            temp_diff = temperature[1:] - temperature[:-1]
        
        elevation_diff = torch.where(elevation_diff == 0, 1e-6, elevation_diff)
        actual_lapse_rate = temp_diff / elevation_diff
        lapse_rate_error = torch.abs(actual_lapse_rate - expected_lapse_rate)
        
        return torch.mean(lapse_rate_error)


class MassBalanceConstraint(nn.Module):
    """Neural network implementation of mass balance constraint."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mass_balance_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mass_balance_net(x)


class EnergyBalanceConstraint(nn.Module):
    """Neural network implementation of energy balance constraint."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.energy_balance_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy_balance_net(x)


class SnowMeltConstraint(nn.Module):
    """Snow melt constraint based on temperature and energy."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.melt_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, temperature: torch.Tensor, 
                radiation: torch.Tensor) -> torch.Tensor:
        x = torch.cat([temperature, radiation], dim=-1)
        return self.melt_net(x)


class PhysicsLoss(nn.Module):
    """Combined physics loss function."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.weights = weights or {
            'mass_balance': 0.1,
            'energy_balance': 0.05,
            'temperature_gradient': 0.05,
            'snow_density': 0.1
        }
        
        self.physics_constraints = PhysicsConstraints()
        
    def forward(self, predictions: torch.Tensor, 
                inputs: torch.Tensor,
                metadata: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Calculate combined physics loss."""
        total_loss = 0.0
        
        if metadata is None:
            metadata = {}
        
        temperature = inputs[:, :, 0] if inputs.shape[-1] > 0 else None
        precipitation = inputs[:, :, 1] if inputs.shape[-1] > 1 else None
        elevation = metadata.get('elevation', None)
        
        # Mass balance constraint
        if precipitation is not None:
            melt = torch.sigmoid(temperature / 5.0) * predictions
            sublimation = torch.zeros_like(predictions)
            
            mass_balance_loss = self.physics_constraints.mass_balance_constraint(
                predictions, precipitation, melt, sublimation
            )
            total_loss += self.weights['mass_balance'] * mass_balance_loss
        
        # Energy balance constraint
        if 'radiation' in metadata:
            radiation = metadata['radiation']
            wind_speed = metadata.get('wind_speed', torch.ones_like(temperature))
            humidity = metadata.get('humidity', torch.ones_like(temperature))
            
            energy_balance_loss = self.physics_constraints.energy_balance_constraint(
                temperature, radiation, wind_speed, humidity
            )
            total_loss += self.weights['energy_balance'] * energy_balance_loss
        
        # Temperature gradient constraint
        if elevation is not None:
            temp_gradient_loss = self.physics_constraints.temperature_gradient_constraint(
                temperature, elevation
            )
            total_loss += self.weights['temperature_gradient'] * temp_gradient_loss
        
        return total_loss


class CombinedLoss(nn.Module):
    """Combined data loss and physics loss."""
    
    def __init__(self, data_loss: nn.Module, physics_loss: PhysicsLoss, 
                 physics_weight: float = 0.1):
        super().__init__()
        self.data_loss = data_loss
        self.physics_loss = physics_loss
        self.physics_weight = physics_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                inputs: torch.Tensor, metadata: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Calculate combined loss."""
        data_loss_value = self.data_loss(predictions, targets)
        physics_loss_value = self.physics_loss(predictions, inputs, metadata)
        total_loss = data_loss_value + self.physics_weight * physics_loss_value
        return total_loss
