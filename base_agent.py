"""
Base agent class for Claims Health Analytics system.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from config import AzureConfig, AgentConfig
from utils import setup_logging, timing_decorator
from models import ClaimSubmission

class BaseAgent(ABC):
    """Base class for all agents in the claims processing system."""
    
    def __init__(self, azure_config: AzureConfig, agent_config: AgentConfig):
        """Initialize the base agent."""
        self.azure_config = azure_config
        self.agent_config = agent_config
        self.logger = setup_logging()
        self.credential = self._setup_credentials()
    
    def _setup_credentials(self):
        """Set up Azure credentials."""
        try:
            # Try DefaultAzureCredential first
            return DefaultAzureCredential()
        except Exception as e:
            self.logger.warning(f"DefaultAzureCredential failed: {e}")
            # Fall back to API key if available
            if self.azure_config.openai_api_key:
                return AzureKeyCredential(self.azure_config.openai_api_key)
            else:
                raise ValueError("No valid Azure credentials found")
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input data and return results."""
        pass
    
    @timing_decorator
    async def execute_with_retry(self, operation, max_retries: int = 3, delay: float = 1.0):
        """Execute operation with retry logic."""
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Operation failed after {max_retries} attempts: {e}")
                    raise
                else:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
    
    def validate_input(self, input_data: Any, required_fields: List[str]) -> bool:
        """Validate input data has required fields."""
        if isinstance(input_data, dict):
            missing_fields = [field for field in required_fields if field not in input_data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            return True
        return False
    
    def log_processing_step(self, claim: ClaimSubmission, step_name: str, result: str, duration: float):
        """Log a processing step for the claim."""
        claim.add_processing_step(step_name, result, duration)
        self.logger.info(f"Claim {claim.claim_id}: {step_name} completed in {duration:.2f}s - {result}")

class AgentOrchestrator:
    """Orchestrates the execution of multiple agents in the claims processing pipeline."""
    
    def __init__(self, azure_config: AzureConfig, agent_config: AgentConfig):
        """Initialize the orchestrator."""
        self.azure_config = azure_config
        self.agent_config = agent_config
        self.logger = setup_logging()
        self.agents = {}
    
    def register_agent(self, name: str, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name}")
    
    async def process_claim(self, claim: ClaimSubmission) -> ClaimSubmission:
        """Process a claim through all registered agents."""
        self.logger.info(f"Starting claim processing for {claim.claim_id}")
        
        try:
            # Document Extraction Agent
            if "document_extraction" in self.agents and claim.documents:
                self.logger.info(f"Processing documents for claim {claim.claim_id}")
                claim.status = claim.status.DOCUMENT_EXTRACTION
                
                extraction_results = []
                for doc_metadata in claim.documents:
                    result = await self.agents["document_extraction"].process(doc_metadata)
                    extraction_results.append(result)
                
                claim.document_extraction_results = extraction_results
                claim.status = claim.status.HEALTH_ANALYSIS
            
            # Health Analytics Agent
            if "health_analytics" in self.agents and claim.document_extraction_results:
                self.logger.info(f"Performing health analysis for claim {claim.claim_id}")
                
                # Combine all extracted text for health analysis
                combined_text = " ".join([
                    result.extracted_text for result in claim.document_extraction_results
                ])
                
                health_analysis = await self.agents["health_analytics"].process(combined_text)
                claim.health_analysis_result = health_analysis
                claim.status = claim.status.UNDERWRITER_REVIEW
            
            # Underwriting Reasoning Agent
            if "underwriting" in self.agents and claim.health_analysis_result:
                self.logger.info(f"Generating underwriting recommendation for claim {claim.claim_id}")
                
                underwriting_input = {
                    "claim": claim,
                    "health_analysis": claim.health_analysis_result,
                    "documents": claim.document_extraction_results
                }
                
                recommendation = await self.agents["underwriting"].process(underwriting_input)
                claim.underwriter_recommendation = recommendation
                
                # Update final status based on recommendation
                if recommendation.recommendation.upper() == "APPROVE":
                    claim.status = claim.status.APPROVED
                elif recommendation.recommendation.upper() == "REJECT":
                    claim.status = claim.status.REJECTED
                else:
                    claim.status = claim.status.REQUIRES_ADDITIONAL_INFO
            
            self.logger.info(f"Claim processing completed for {claim.claim_id} with status: {claim.status.value}")
            return claim
            
        except Exception as e:
            self.logger.error(f"Error processing claim {claim.claim_id}: {e}")
            claim.add_processing_step("error", str(e), 0.0)
            raise
    
    async def process_multiple_claims(self, claims: List[ClaimSubmission]) -> List[ClaimSubmission]:
        """Process multiple claims concurrently."""
        self.logger.info(f"Processing {len(claims)} claims concurrently")
        
        tasks = [self.process_claim(claim) for claim in claims]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_claims = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to process claim {claims[i].claim_id}: {result}")
                claims[i].add_processing_step("error", str(result), 0.0)
                processed_claims.append(claims[i])
            else:
                processed_claims.append(result)
        
        return processed_claims
    
    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all registered agents."""
        return {name: "active" for name in self.agents.keys()}