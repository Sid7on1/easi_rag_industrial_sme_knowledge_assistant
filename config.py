import os
import yaml
from pathlib import Path
from pydantic import BaseModel, BaseSettings, Field
from typing import Dict, Any
import logging
from logging.handlers import RotatingFileHandler
from logging.config import dictConfig

# Set up logging
dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'config.log',
            'maxBytes': 1000000,
            'backupCount': 1,
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi', 'file']
    }
})

logger = logging.getLogger(__name__)

class Config(BaseModel):
    """Base configuration model"""
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class Settings(BaseSettings):
    """Base settings model"""
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class ModelConfig(BaseModel):
    """Model configuration model"""
    model_name: str = Field(..., description="Name of the model")
    model_path: str = Field(..., description="Path to the model")
    batch_size: int = Field(32, description="Batch size for training")
    epochs: int = Field(10, description="Number of epochs for training")

class RetrievalConfig(BaseModel):
    """Retrieval configuration model"""
    retrieval_name: str = Field(..., description="Name of the retrieval model")
    retrieval_path: str = Field(..., description="Path to the retrieval model")
    top_k: int = Field(10, description="Top k results to return")
    similarity_threshold: float = Field(0.5, description="Similarity threshold for retrieval")

class ConfigManager:
    """Config manager class"""
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = None

    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                self.config = Config(**config)
                logger.info("Config loaded successfully")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    def validate_config(self):
        """Validate configuration"""
        if self.config is None:
            logger.error("Config not loaded")
            return False
        try:
            self.config.validate()
            logger.info("Config validated successfully")
            return True
        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return False

    def get_model_config(self):
        """Get model configuration"""
        if self.config is None:
            logger.error("Config not loaded")
            return None
        try:
            model_config = ModelConfig(**self.config.model_config)
            logger.info("Model config loaded successfully")
            return model_config
        except Exception as e:
            logger.error(f"Error loading model config: {e}")
            return None

    def get_retrieval_config(self):
        """Get retrieval configuration"""
        if self.config is None:
            logger.error("Config not loaded")
            return None
        try:
            retrieval_config = RetrievalConfig(**self.config.retrieval_config)
            logger.info("Retrieval config loaded successfully")
            return retrieval_config
        except Exception as e:
            logger.error(f"Error loading retrieval config: {e}")
            return None

def load_config(config_file: str):
    """Load configuration from file"""
    config_manager = ConfigManager(config_file)
    config_manager.load_config()
    return config_manager

def validate_config(config_manager: ConfigManager):
    """Validate configuration"""
    return config_manager.validate_config()

def get_model_config(config_manager: ConfigManager):
    """Get model configuration"""
    return config_manager.get_model_config()

def get_retrieval_config(config_manager: ConfigManager):
    """Get retrieval configuration"""
    return config_manager.get_retrieval_config()

if __name__ == "__main__":
    config_file = "config.yaml"
    config_manager = load_config(config_file)
    if validate_config(config_manager):
        model_config = get_model_config(config_manager)
        retrieval_config = get_retrieval_config(config_manager)
        print(model_config)
        print(retrieval_config)
    else:
        logger.error("Config validation failed")