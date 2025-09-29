"""
Main application for Claims Health Analytics system.
Orchestrates document extraction, health analytics, and underwriting agents.
"""
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import argparse

from config import get_azure_config, get_agent_config
from base_agent import AgentOrchestrator
from document_extraction_agent import DocumentExtractionAgent
from health_analytics_agent import HealthAnalyticsAgent
from underwriting_agent import UnderwritingReasoningAgent
from models import ClaimSubmission, DocumentMetadata, ClaimStatus
from utils import (
    setup_logging, generate_claim_id, generate_document_id,
    HealthAnalyticsException, DocumentProcessingException, UnderwritingException
)

class ClaimsHealthAnalyticsApp:
    """Main application for claims health analytics."""
    
    def __init__(self):
        """Initialize the application."""
        self.logger = setup_logging()
        self.azure_config = get_azure_config()
        self.agent_config = get_agent_config()
        self.orchestrator = AgentOrchestrator(self.azure_config, self.agent_config)
        self._setup_agents()
    
    def _setup_agents(self):
        """Initialize and register all agents."""
        try:
            # Initialize agents
            document_agent = DocumentExtractionAgent(self.azure_config, self.agent_config)
            health_agent = HealthAnalyticsAgent(self.azure_config, self.agent_config)
            underwriting_agent = UnderwritingReasoningAgent(self.azure_config, self.agent_config)
            
            # Register agents with orchestrator
            self.orchestrator.register_agent("document_extraction", document_agent)
            self.orchestrator.register_agent("health_analytics", health_agent)
            self.orchestrator.register_agent("underwriting", underwriting_agent)
            
            self.logger.info("All agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup agents: {e}")
            raise
    
    async def process_claim_from_files(
        self, 
        document_paths: List[str], 
        claimant_info: Dict[str, any],
        claim_id: Optional[str] = None
    ) -> ClaimSubmission:
        """Process a claim from document files."""
        
        # Generate claim ID if not provided
        if not claim_id:
            claim_id = generate_claim_id()
        
        self.logger.info(f"Processing claim {claim_id} with {len(document_paths)} documents")
        
        # Validate document files
        for doc_path in document_paths:
            if not os.path.exists(doc_path):
                raise FileNotFoundError(f"Document not found: {doc_path}")
        
        # Create document metadata
        documents = []
        for doc_path in document_paths:
            file_stats = os.stat(doc_path)
            metadata = DocumentMetadata(
                file_name=os.path.basename(doc_path),
                file_type=doc_path.split('.')[-1].lower(),
                file_size=file_stats.st_size,
                upload_time=datetime.fromtimestamp(file_stats.st_ctime),
                document_id=generate_document_id()
            )
            documents.append(metadata)
        
        # Create claim submission
        claim = ClaimSubmission(
            claim_id=claim_id,
            status=ClaimStatus.SUBMITTED,
            submitted_at=datetime.now(),
            claimant_info=claimant_info,
            documents=documents
        )
        
        # Process claim through orchestrator
        try:
            processed_claim = await self.orchestrator.process_claim(claim)
            self.logger.info(f"Claim {claim_id} processed successfully. Final status: {processed_claim.status.value}")
            return processed_claim
            
        except Exception as e:
            self.logger.error(f"Failed to process claim {claim_id}: {e}")
            claim.add_processing_step("error", str(e), 0.0)
            raise
    
    async def process_claim_from_text(
        self,
        text_content: str,
        claimant_info: Dict[str, any],
        claim_id: Optional[str] = None
    ) -> ClaimSubmission:
        """Process a claim from text content."""
        
        # Generate claim ID if not provided
        if not claim_id:
            claim_id = generate_claim_id()
        
        self.logger.info(f"Processing claim {claim_id} from text content")
        
        # Create document metadata for text
        metadata = DocumentMetadata(
            file_name="text_input.txt",
            file_type="txt",
            file_size=len(text_content.encode('utf-8')),
            upload_time=datetime.now(),
            document_id=generate_document_id()
        )
        
        # Create claim submission
        claim = ClaimSubmission(
            claim_id=claim_id,
            status=ClaimStatus.SUBMITTED,
            submitted_at=datetime.now(),
            claimant_info=claimant_info,
            documents=[metadata]
        )
        
        # Process text directly through health analytics
        try:
            # Skip document extraction and go directly to health analytics
            health_agent = self.orchestrator.agents["health_analytics"]
            health_analysis = await health_agent.process(text_content)
            claim.health_analysis_result = health_analysis
            claim.status = ClaimStatus.UNDERWRITER_REVIEW
            
            # Generate underwriting recommendation
            underwriting_agent = self.orchestrator.agents["underwriting"]
            underwriting_input = {
                "claim": claim,
                "health_analysis": health_analysis,
                "documents": []
            }
            recommendation = await underwriting_agent.process(underwriting_input)
            claim.underwriter_recommendation = recommendation
            
            # Update final status
            if recommendation.recommendation.upper() == "APPROVE":
                claim.status = ClaimStatus.APPROVED
            elif recommendation.recommendation.upper() == "REJECT":
                claim.status = ClaimStatus.REJECTED
            else:
                claim.status = ClaimStatus.REQUIRES_ADDITIONAL_INFO
            
            self.logger.info(f"Claim {claim_id} processed successfully. Final status: {claim.status.value}")
            return claim
            
        except Exception as e:
            self.logger.error(f"Failed to process claim {claim_id}: {e}")
            claim.add_processing_step("error", str(e), 0.0)
            raise
    
    async def batch_process_claims(self, claims_data: List[Dict[str, any]]) -> List[ClaimSubmission]:
        """Process multiple claims in batch."""
        self.logger.info(f"Processing {len(claims_data)} claims in batch")
        
        tasks = []
        for claim_data in claims_data:
            if "document_paths" in claim_data:
                task = self.process_claim_from_files(
                    claim_data["document_paths"],
                    claim_data["claimant_info"],
                    claim_data.get("claim_id")
                )
            elif "text_content" in claim_data:
                task = self.process_claim_from_text(
                    claim_data["text_content"],
                    claim_data["claimant_info"],
                    claim_data.get("claim_id")
                )
            else:
                self.logger.error(f"Invalid claim data: {claim_data}")
                continue
            
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_claims = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to process claim {i}: {result}")
                # Create error claim
                error_claim = ClaimSubmission(
                    claim_id=generate_claim_id(),
                    status=ClaimStatus.REJECTED,
                    submitted_at=datetime.now(),
                    claimant_info=claims_data[i].get("claimant_info", {}),
                    documents=[]
                )
                error_claim.add_processing_step("error", str(result), 0.0)
                processed_claims.append(error_claim)
            else:
                processed_claims.append(result)
        
        return processed_claims
    
    def save_claim_report(self, claim: ClaimSubmission, output_path: str):
        """Save claim processing report to file."""
        try:
            report = {
                "claim_summary": {
                    "claim_id": claim.claim_id,
                    "status": claim.status.value,
                    "submitted_at": claim.submitted_at.isoformat(),
                    "last_updated": claim.last_updated.isoformat(),
                    "processing_duration": sum(
                        step.get("duration", 0) for step in claim.processing_history
                    )
                },
                "claimant_info": claim.claimant_info,
                "documents": [doc.__dict__ for doc in claim.documents],
                "health_analysis": claim.health_analysis_result.__dict__ if claim.health_analysis_result else None,
                "underwriter_recommendation": claim.underwriter_recommendation.__dict__ if claim.underwriter_recommendation else None,
                "processing_history": claim.processing_history
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Claim report saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save claim report: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, any]:
        """Get system status and agent health."""
        return {
            "status": "operational",
            "agents": self.orchestrator.get_agent_status(),
            "configuration": {
                "document_extraction_model": self.agent_config.document_extraction_model,
                "health_analytics_model": self.agent_config.health_analytics_model,
                "reasoning_model": self.agent_config.reasoning_model,
                "confidence_threshold": self.agent_config.confidence_threshold
            },
            "timestamp": datetime.now().isoformat()
        }

async def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Claims Health Analytics System")
    parser.add_argument("--mode", choices=["files", "text", "demo"], default="demo",
                       help="Processing mode")
    parser.add_argument("--documents", nargs="+", help="Document file paths")
    parser.add_argument("--text", help="Text content to analyze")
    parser.add_argument("--claimant", help="Claimant info JSON file")
    parser.add_argument("--output", help="Output report path")
    parser.add_argument("--batch", help="Batch processing JSON file")
    
    args = parser.parse_args()
    
    # Initialize application
    app = ClaimsHealthAnalyticsApp()
    
    try:
        if args.mode == "demo":
            # Run demo with sample data
            await run_demo(app)
        
        elif args.mode == "files" and args.documents:
            # Process from files
            claimant_info = {}
            if args.claimant:
                with open(args.claimant, 'r') as f:
                    claimant_info = json.load(f)
            
            claim = await app.process_claim_from_files(args.documents, claimant_info)
            
            if args.output:
                app.save_claim_report(claim, args.output)
            else:
                print(json.dumps(claim.to_dict(), indent=2, default=str))
        
        elif args.mode == "text" and args.text:
            # Process from text
            claimant_info = {}
            if args.claimant:
                with open(args.claimant, 'r') as f:
                    claimant_info = json.load(f)
            
            claim = await app.process_claim_from_text(args.text, claimant_info)
            
            if args.output:
                app.save_claim_report(claim, args.output)
            else:
                print(json.dumps(claim.to_dict(), indent=2, default=str))
        
        elif args.batch:
            # Batch processing
            with open(args.batch, 'r') as f:
                batch_data = json.load(f)
            
            claims = await app.batch_process_claims(batch_data)
            
            for i, claim in enumerate(claims):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"output/claim_report_{timestamp}_{i+1}.json"
                app.save_claim_report(claim, output_path)
        
        else:
            print("Invalid arguments. Use --help for usage information.")
            return 1
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

async def run_demo(app: ClaimsHealthAnalyticsApp):
    """Run a demonstration of the claims processing system."""
    print("=== Claims Health Analytics Demo ===")
    print()
    
    # Sample claimant information
    claimant_info = {
        "name": "John Doe",
        "age": 45,
        "policy_number": "POL-2024-001",
        "claim_type": "medical_expenses",
        "claim_amount": 15000.00
    }
    
    # Sample medical text
    sample_text = """
    Patient: John Doe, 45 years old
    
    Medical History:
    - Diagnosed with Type 2 diabetes in 2020
    - Hypertension managed with Lisinopril 10mg daily
    - Recent chest pain episodes
    
    Current Symptoms:
    - Chest discomfort during physical activity
    - Occasional shortness of breath
    - Fatigue
    
    Recent Procedures:
    - Cardiac stress test performed on 01/15/2024
    - Blood glucose monitoring
    - Echocardiogram scheduled
    
    Medications:
    - Metformin 500mg twice daily
    - Lisinopril 10mg daily
    - Aspirin 81mg daily
    
    Doctor's Notes:
    Patient presents with cardiac symptoms. Recommend cardiology consultation.
    Continue current diabetes management. Monitor blood pressure.
    """
    
    print("Processing sample claim...")
    print(f"Claimant: {claimant_info['name']}")
    print(f"Policy: {claimant_info['policy_number']}")
    print(f"Claim Amount: ${claimant_info['claim_amount']:,.2f}")
    print()
    
    try:
        # Process the sample claim
        claim = await app.process_claim_from_text(sample_text, claimant_info)
        
        # Display results
        print("=== PROCESSING RESULTS ===")
        print()
        print(f"Claim ID: {claim.claim_id}")
        print(f"Status: {claim.status.value}")
        print()
        
        if claim.health_analysis_result:
            health = claim.health_analysis_result
            print("Health Analysis:")
            print(f"  Conditions: {', '.join(health.medical_conditions) if health.medical_conditions else 'None'}")
            print(f"  Medications: {', '.join(health.medications) if health.medications else 'None'}")
            print(f"  Symptoms: {', '.join(health.symptoms) if health.symptoms else 'None'}")
            print(f"  Risk Factors: {', '.join(health.risk_factors) if health.risk_factors else 'None'}")
            print(f"  Assessment: {health.overall_health_assessment}")
            print(f"  Confidence: {health.confidence_level.value}")
            print()
        
        if claim.underwriter_recommendation:
            rec = claim.underwriter_recommendation
            print("Underwriting Recommendation:")
            print(f"  Decision: {rec.recommendation}")
            print(f"  Confidence: {rec.confidence_score:.2f}")
            print(f"  Estimated Amount: ${rec.estimated_claim_amount:,.2f}" if rec.estimated_claim_amount else "  Estimated Amount: N/A")
            print(f"  Risk Assessment: {rec.risk_assessment}")
            print()
            
            if rec.red_flags:
                print("  Red Flags:")
                for flag in rec.red_flags:
                    print(f"    - {flag}")
                print()
            
            if rec.required_actions:
                print("  Required Actions:")
                for action in rec.required_actions:
                    print(f"    - {action}")
                print()
        
        # Show processing history
        print("Processing History:")
        for step in claim.processing_history:
            print(f"  {step['step']}: {step['result']} ({step['duration']:.2f}s)")
        
        # Save demo report to output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/demo_claim_report_{timestamp}.json"
        app.save_claim_report(claim, output_path)
        print(f"\nDetailed report saved to: {output_path}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())