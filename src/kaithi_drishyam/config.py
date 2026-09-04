"""Configuration management for Kaithi-Drishyam."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="KAITHI_",
        case_sensitive=False,
    )

    # Paths
    base_dir: Path = Field(default=Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))

    # Bhashini API Configuration
    bhashini_api_key: Optional[str] = Field(default=None)
    bhashini_api_url: str = Field(default="https://api.bhashini.gov.in")
    bhashini_udyat_url: str = Field(default="https://udyat.bhashini.gov.in/api")
    bhashini_timeout: int = Field(default=30)

    # Model Configuration
    crnn_hidden_size: int = Field(default=256)
    crnn_num_layers: int = Field(default=2)
    crnn_dropout: float = Field(default=0.1)

    # Processing Configuration
    max_image_size_mb: int = Field(default=50)
    max_skew_angle: float = Field(default=15.0)
    confidence_threshold: float = Field(default=0.7)
    processing_timeout: int = Field(default=30)

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    max_concurrent_requests: int = Field(default=10)
    rate_limit_per_minute: int = Field(default=60)

    # Device Configuration
    use_gpu: bool = Field(default=True)
    gpu_device_id: int = Field(default=0)

    @property
    def device(self) -> str:
        """Get PyTorch device string."""
        import torch

        if self.use_gpu and torch.cuda.is_available():
            return f"cuda:{self.gpu_device_id}"
        return "cpu"

    def get_data_path(self, subdir: str = "") -> Path:
        """Get absolute path within data directory."""
        return self.base_dir / self.data_dir / subdir

    def get_model_path(self, subdir: str = "") -> Path:
        """Get absolute path within models directory."""
        return self.base_dir / self.models_dir / subdir


# Global settings instance
settings = Settings()
