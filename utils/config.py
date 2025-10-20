"""
Configuration management for NGIS.

Handles loading, validation, and access to configuration parameters
from YAML files and command line arguments.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from dataclasses import dataclass, field
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: str = "data/"
    batch_size: int = 32
    num_workers: int = 4
    segment_length: int = 1000
    overlap: float = 0.0
    sampling_rate: float = 1000.0
    channels: Optional[list] = None
    preprocess: bool = True
    augment: bool = True
    preload: bool = False  # Lazy loading by default to save memory


@dataclass
class ModelConfig:
    """Model configuration."""
    n_channels: int = 128
    n_neurons: int = 256
    n_layers: int = 3
    hidden_dim: int = 64
    graph_type: str = "functional"
    connectivity_threshold: float = 0.1
    dropout: float = 0.1
    lif_params: Dict[str, Any] = field(default_factory=dict)
    synapse_params: Dict[str, Any] = field(default_factory=dict)
    readout_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    output_dir: str = "outputs/"
    save_frequency: int = 10
    log_frequency: int = 100
    validation_frequency: int = 1
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0


@dataclass
class LossConfig:
    """Loss function configuration."""
    eeg_weight: float = 1.0
    spiking_weight: float = 0.1
    biological_weight: float = 0.01
    regularization_weight: float = 0.001


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "reduce_lr_on_plateau"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointingConfig:
    """Checkpointing configuration."""
    save_dir: str = "checkpoints/"
    save_frequency: int = 10
    max_checkpoints: int = 5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_frequency: int = 100
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "ngis"


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Set default scheduler parameters
        if not self.scheduler.params:
            if self.scheduler.type == "reduce_lr_on_plateau":
                self.scheduler.params = {
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 5,
                    "min_lr": 1e-6
                }
            elif self.scheduler.type == "cosine":
                self.scheduler.params = {
                    "T_max": self.training.epochs,
                    "eta_min": 1e-6
                }
        
        # Set default LIF parameters
        if not self.model.lif_params:
            self.model.lif_params = {
                "tau_m": 20.0,
                "v_rest": -65.0,
                "v_threshold": -55.0,
                "v_reset": -65.0,
                "refractory_period": 2.0
            }
        
        # Set default synapse parameters
        if not self.model.synapse_params:
            self.model.synapse_params = {
                "tau_s": 5.0,
                "weight_scale": 1.0,
                "plasticity": True
            }
        
        # Set default readout parameters
        if not self.model.readout_params:
            self.model.readout_params = {
                "readout_type": "linear",
                "activation": "tanh"
            }
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
            
        Returns:
            Config object.
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to OmegaConf for better handling
        conf = OmegaConf.create(config_dict)
        
        # Create config object
        config = cls()
        
        # Update with loaded values
        for key, value in conf.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def to_yaml(self, config_path: Union[str, Path]):
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save configuration file.
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid.
        """
        errors = []
        
        # Validate data configuration
        if self.data.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.data.segment_length <= 0:
            errors.append("segment_length must be positive")
        
        if not (0 <= self.data.overlap < 1):
            errors.append("overlap must be between 0 and 1")
        
        # Validate model configuration
        if self.model.n_channels <= 0:
            errors.append("n_channels must be positive")
        
        if self.model.n_neurons <= 0:
            errors.append("n_neurons must be positive")
        
        if self.model.n_layers <= 0:
            errors.append("n_layers must be positive")
        
        if self.model.hidden_dim <= 0:
            errors.append("hidden_dim must be positive")
        
        if not (0 <= self.model.dropout <= 1):
            errors.append("dropout must be between 0 and 1")
        
        # Validate training configuration
        if self.training.epochs <= 0:
            errors.append("epochs must be positive")
        
        if self.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if self.training.batch_size <= 0:
            errors.append("training batch_size must be positive")
        
        # Validate loss configuration
        if self.loss.eeg_weight < 0:
            errors.append("eeg_weight must be non-negative")
        
        if self.loss.spiking_weight < 0:
            errors.append("spiking_weight must be non-negative")
        
        if self.loss.biological_weight < 0:
            errors.append("biological_weight must be non-negative")
        
        if self.loss.regularization_weight < 0:
            errors.append("regularization_weight must be non-negative")
        
        # Validate optimizer configuration
        if self.optimizer.learning_rate <= 0:
            errors.append("optimizer learning_rate must be positive")
        
        if self.optimizer.weight_decay < 0:
            errors.append("weight_decay must be non-negative")
        
        # Report errors
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary.
        
        Returns:
            Dictionary with configuration summary.
        """
        return {
            'data': {
                'batch_size': self.data.batch_size,
                'segment_length': self.data.segment_length,
                'sampling_rate': self.data.sampling_rate,
                'n_channels': self.data.channels or self.model.n_channels
            },
            'model': {
                'n_neurons': self.model.n_neurons,
                'n_layers': self.model.n_layers,
                'hidden_dim': self.model.hidden_dim,
                'graph_type': self.model.graph_type,
                'dropout': self.model.dropout
            },
            'training': {
                'epochs': self.training.epochs,
                'learning_rate': self.training.learning_rate,
                'batch_size': self.training.batch_size
            },
            'loss': {
                'eeg_weight': self.loss.eeg_weight,
                'spiking_weight': self.loss.spiking_weight,
                'biological_weight': self.loss.biological_weight,
                'regularization_weight': self.loss.regularization_weight
            },
            'optimizer': {
                'type': self.optimizer.type,
                'learning_rate': self.optimizer.learning_rate,
                'weight_decay': self.optimizer.weight_decay
            },
            'scheduler': {
                'type': self.scheduler.type,
                'params': self.scheduler.params
            }
        } 