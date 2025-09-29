"""
Claims Health Analytics with OpenTelemetry tracing support.
"""
import os
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

# Set up OpenTelemetry tracing for monitoring and observability
os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

from opentelemetry import trace, _events, _logs
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

from main import ClaimsHealthAnalyticsApp
from models import ClaimSubmission, DocumentMetadata
from utils import setup_logging, generate_claim_id
from config import get_azure_config, get_agent_config

class TracedClaimsAnalyticsApp(ClaimsHealthAnalyticsApp):
    """Claims Health Analytics with OpenTelemetry tracing."""
    
    def __init__(self):
        """Initialize with tracing support."""
        self._setup_tracing()
        super().__init__()
        self.tracer = trace.get_tracer(__name__)
    
    def _setup_tracing(self):
        """Set up OpenTelemetry tracing."""
        try:
            resource = Resource(attributes={
                "service.name": "claims-health-analytics",
                "service.version": "1.0.0",
                "service.instance.id": os.getenv("HOSTNAME", "local-instance")
            })
            
            provider = TracerProvider(resource=resource)
            
            # Configure OTLP exporter (defaulting to localhost)
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
            
            otlp_exporter = OTLPSpanExporter(
                endpoint=f"{otlp_endpoint}/v1/traces",
            )
            processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            
            # Configure logging and events
            _logs.set_logger_provider(LoggerProvider())
            _logs.get_logger_provider().add_log_record_processor(
                BatchLogRecordProcessor(OTLPLogExporter(endpoint=f"{otlp_endpoint}/v1/logs"))
            )
            _events.set_event_logger_provider(EventLoggerProvider())
            
            # Instrument OpenAI SDK
            OpenAIInstrumentor().instrument()
            
            self.logger.info("OpenTelemetry tracing initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize tracing: {e}")
            # Continue without tracing if setup fails
    
    async def process_claim_from_files_traced(
        self, 
        document_paths: List[str], 
        claimant_info: Dict[str, any],
        claim_id: Optional[str] = None
    ) -> ClaimSubmission:
        """Process claim with tracing."""
        with self.tracer.start_as_current_span("process_claim_from_files") as span:
            # Add span attributes
            span.set_attribute("claim.document_count", len(document_paths))
            span.set_attribute("claim.claimant_name", claimant_info.get("name", "unknown"))
            span.set_attribute("claim.claim_type", claimant_info.get("claim_type", "unknown"))
            
            if claim_id:
                span.set_attribute("claim.id", claim_id)
            
            try:
                # Process claim using parent method
                claim = await super().process_claim_from_files(document_paths, claimant_info, claim_id)
                
                # Add result attributes
                span.set_attribute("claim.final_status", claim.status.value)
                span.set_attribute("claim.processing_success", True)
                
                if claim.underwriter_recommendation:
                    span.set_attribute("claim.recommendation", claim.underwriter_recommendation.recommendation)
                    span.set_attribute("claim.confidence_score", claim.underwriter_recommendation.confidence_score)
                
                return claim
                
            except Exception as e:
                span.set_attribute("claim.processing_success", False)
                span.set_attribute("claim.error", str(e))
                span.record_exception(e)
                raise
    
    async def process_claim_from_text_traced(
        self,
        text_content: str,
        claimant_info: Dict[str, any],
        claim_id: Optional[str] = None
    ) -> ClaimSubmission:
        """Process claim from text with tracing."""
        with self.tracer.start_as_current_span("process_claim_from_text") as span:
            # Add span attributes
            span.set_attribute("claim.text_length", len(text_content))
            span.set_attribute("claim.claimant_name", claimant_info.get("name", "unknown"))
            span.set_attribute("claim.claim_type", claimant_info.get("claim_type", "unknown"))
            
            if claim_id:
                span.set_attribute("claim.id", claim_id)
            
            try:
                # Process claim using parent method
                claim = await super().process_claim_from_text(text_content, claimant_info, claim_id)
                
                # Add result attributes
                span.set_attribute("claim.final_status", claim.status.value)
                span.set_attribute("claim.processing_success", True)
                
                if claim.health_analysis_result:
                    span.set_attribute("health.conditions_count", len(claim.health_analysis_result.medical_conditions))
                    span.set_attribute("health.medications_count", len(claim.health_analysis_result.medications))
                    span.set_attribute("health.confidence_level", claim.health_analysis_result.confidence_level.value)
                
                if claim.underwriter_recommendation:
                    span.set_attribute("underwriting.recommendation", claim.underwriter_recommendation.recommendation)
                    span.set_attribute("underwriting.confidence_score", claim.underwriter_recommendation.confidence_score)
                    span.set_attribute("underwriting.estimated_amount", claim.underwriter_recommendation.estimated_claim_amount or 0)
                
                return claim
                
            except Exception as e:
                span.set_attribute("claim.processing_success", False)
                span.set_attribute("claim.error", str(e))
                span.record_exception(e)
                raise
    
    async def batch_process_claims_traced(self, claims_data: List[Dict[str, any]]) -> List[ClaimSubmission]:
        """Batch process claims with tracing."""
        with self.tracer.start_as_current_span("batch_process_claims") as span:
            span.set_attribute("batch.size", len(claims_data))
            
            try:
                # Process claims using parent method
                claims = await super().batch_process_claims(claims_data)
                
                # Add batch result metrics
                successful_claims = sum(1 for claim in claims if claim.status.value != "rejected")
                span.set_attribute("batch.successful_count", successful_claims)
                span.set_attribute("batch.failed_count", len(claims) - successful_claims)
                span.set_attribute("batch.processing_success", True)
                
                return claims
                
            except Exception as e:
                span.set_attribute("batch.processing_success", False)
                span.set_attribute("batch.error", str(e))
                span.record_exception(e)
                raise

async def demo_with_tracing():
    """Demo with tracing enabled."""
    print("Claims Health Analytics - Tracing Demo")
    print("=" * 50)
    
    # Initialize traced application
    app = TracedClaimsAnalyticsApp()
    
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
    
    print("Processing sample claim with tracing...")
    print(f"Claimant: {claimant_info['name']}")
    print(f"Policy: {claimant_info['policy_number']}")
    print(f"Claim Amount: ${claimant_info['claim_amount']:,.2f}")
    print()
    
    try:
        # Process the sample claim with tracing
        claim = await app.process_claim_from_text_traced(sample_text, claimant_info)
        
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
        
        # Show processing history
        print("Processing History:")
        for step in claim.processing_history:
            print(f"  {step['step']}: {step['result']} ({step.get('duration', 0):.2f}s)")
        
        print(f"\nClaim processing completed successfully!")
        print("Tracing data has been sent to OpenTelemetry collector.")
        print("If you have a tracing backend (e.g., Jaeger) running,")
        print("you can view the detailed trace information there.")
        
        return claim
        
    except Exception as e:
        print(f"Demo failed: {e}")
        raise

async def main():
    """Main entry point for traced demo."""
    try:
        await demo_with_tracing()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    asyncio.run(main())