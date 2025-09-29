"""
Configuration settings for Claims Health Analytics system.
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class AzureConfig:
    """Azure service configuration."""
    openai_endpoint: str
    openai_deployment_name: str
    openai_api_key: Optional[str] = None
    
    # Text Analytics for Health configuration
    text_analytics_endpoint: Optional[str] = None
    text_analytics_key: Optional[str] = None
    
    # Document Intelligence configuration
    document_intelligence_endpoint: Optional[str] = None
    document_intelligence_key: Optional[str] = None
    
    def __post_init__(self):
        """Validate required configuration."""
        if not self.openai_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required")
        if not self.openai_deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME is required")

@dataclass
class AgentConfig:
    """Agent-specific configuration."""
    document_extraction_model: str = "gpt-4.1-mini"
    health_analytics_model: str = "text-analytics-health"
    reasoning_model: str = "gpt-4.1"
    max_tokens: int = 2000
    temperature: float = 0.3
    confidence_threshold: float = 0.7
    openai_api_version: str = "2024-12-01-preview"  # Use compatible API version

def get_azure_config() -> AzureConfig:
    """Get Azure configuration from environment variables."""
    return AzureConfig(
        openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        openai_deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        text_analytics_endpoint=os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT"),
        text_analytics_key=os.getenv("AZURE_TEXT_ANALYTICS_KEY"),
        document_intelligence_endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
        document_intelligence_key=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
    )

def get_agent_config() -> AgentConfig:
    """Get agent configuration with optional overrides from environment."""
    return AgentConfig(
        document_extraction_model=os.getenv("DOCUMENT_EXTRACTION_MODEL", "gpt-4.1-mini"),
        health_analytics_model=os.getenv("HEALTH_ANALYTICS_MODEL", "text-analytics-health"),
        reasoning_model=os.getenv("REASONING_MODEL", "gpt-4.1"),
        max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
        temperature=float(os.getenv("TEMPERATURE", "0.3")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
    )