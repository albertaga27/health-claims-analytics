"""
Usage Examples for Claims Health Analytics System
"""
import asyncio
import json
from datetime import datetime

# Example 1: Basic Usage
async def example_basic_usage():
    """Basic usage example."""
    from main import ClaimsHealthAnalyticsApp
    
    # Initialize the application
    app = ClaimsHealthAnalyticsApp()
    
    # Define claimant information
    claimant_info = {
        "name": "Jane Smith",
        "age": 38,
        "policy_number": "POL-2024-100",
        "claim_type": "medical_consultation",
        "claim_amount": 2500.00
    }
    
    # Medical text content
    medical_text = """
    Patient Jane Smith, 38 years old, presents for annual physical examination.
    
    Medical History:
    - No significant past medical history
    - Allergic to shellfish
    
    Current Health:
    - Generally healthy
    - Reports occasional headaches
    - No current medications
    
    Physical Examination:
    - Vital signs normal
    - No acute findings
    
    Assessment:
    - Healthy adult female
    - Recommend routine screening labs
    """
    
    # Process the claim
    claim = await app.process_claim_from_text(medical_text, claimant_info)
    
    print(f"Claim processed: {claim.claim_id}")
    print(f"Final status: {claim.status.value}")
    if claim.underwriter_recommendation:
        print(f"Recommendation: {claim.underwriter_recommendation.recommendation}")

# Example 2: Batch Processing
async def example_batch_processing():
    """Batch processing example."""
    from main import ClaimsHealthAnalyticsApp
    
    app = ClaimsHealthAnalyticsApp()
    
    # Multiple claims data
    batch_claims = [
        {
            "text_content": "Patient with diabetes, routine follow-up visit.",
            "claimant_info": {"name": "Patient 1", "age": 55, "policy_number": "POL-001"}
        },
        {
            "text_content": "Emergency room visit for chest pain, discharged after tests.",
            "claimant_info": {"name": "Patient 2", "age": 42, "policy_number": "POL-002"}
        },
        {
            "text_content": "Routine vaccination, no complications.",
            "claimant_info": {"name": "Patient 3", "age": 30, "policy_number": "POL-003"}
        }
    ]
    
    # Process all claims
    processed_claims = await app.batch_process_claims(batch_claims)
    
    print(f"Processed {len(processed_claims)} claims:")
    for claim in processed_claims:
        print(f"  {claim.claim_id}: {claim.status.value}")

# Example 3: File Processing
async def example_file_processing():
    """File processing example."""
    from main import ClaimsHealthAnalyticsApp
    import tempfile
    import os
    
    app = ClaimsHealthAnalyticsApp()
    
    # Create a temporary medical document
    medical_content = """
    MEDICAL RECORD
    
    Patient: Bob Johnson
    Date: 2024-01-20
    
    Chief Complaint: Follow-up for hypertension
    
    History: 
    Patient is a 62-year-old male with essential hypertension.
    Currently taking Lisinopril 10mg daily.
    Blood pressure well controlled.
    
    Physical Exam:
    BP: 125/80 mmHg
    Otherwise normal examination
    
    Assessment: Hypertension, well controlled
    Plan: Continue current medication
    """
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(medical_content)
        temp_file = f.name
    
    try:
        claimant_info = {
            "name": "Bob Johnson",
            "age": 62,
            "policy_number": "POL-2024-200",
            "claim_type": "routine_visit"
        }
        
        # Process from file
        claim = await app.process_claim_from_files([temp_file], claimant_info)
        
        print(f"File processed: {claim.claim_id}")
        print(f"Status: {claim.status.value}")
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file)

# Example 4: Complex Medical Case
async def example_complex_case():
    """Complex medical case example."""
    from main import ClaimsHealthAnalyticsApp
    
    app = ClaimsHealthAnalyticsApp()
    
    complex_medical_text = """
    COMPREHENSIVE MEDICAL REPORT
    
    Patient: Maria Rodriguez, 58 years old
    
    PRIMARY DIAGNOSIS: Acute myocardial infarction (STEMI)
    
    MEDICAL HISTORY:
    - Type 2 diabetes mellitus (15 years)
    - Hypertension (10 years)
    - Hyperlipidemia
    - Family history of coronary artery disease
    - Former smoker (quit 2 years ago)
    
    PRESENTING SYMPTOMS:
    - Severe chest pain radiating to left arm
    - Shortness of breath
    - Nausea and diaphoresis
    - Onset 2 hours prior to arrival
    
    CURRENT MEDICATIONS:
    - Metformin 1000mg BID
    - Lisinopril 20mg daily
    - Atorvastatin 40mg daily
    - Aspirin 81mg daily
    
    PROCEDURES PERFORMED:
    - Emergency cardiac catheterization
    - Percutaneous coronary intervention (PCI)
    - Drug-eluting stent placement to LAD
    
    LABORATORY RESULTS:
    - Troponin I: 45.2 ng/mL (elevated)
    - CK-MB: 78 ng/mL (elevated)
    - Glucose: 245 mg/dL
    - HbA1c: 8.9%
    
    COMPLICATIONS:
    - Acute heart failure
    - Hyperglycemia
    
    TREATMENT PLAN:
    - Dual antiplatelet therapy
    - Beta-blocker therapy
    - ACE inhibitor optimization
    - Diabetes management intensification
    - Cardiac rehabilitation referral
    
    PROGNOSIS:
    - Guarded due to extent of disease
    - Requires intensive monitoring
    - Long-term cardiac follow-up essential
    
    ESTIMATED HOSPITAL STAY: 5-7 days
    ESTIMATED TOTAL COST: $75,000 - $100,000
    """
    
    claimant_info = {
        "name": "Maria Rodriguez",
        "age": 58,
        "policy_number": "POL-2024-URGENT-001",
        "claim_type": "emergency_cardiac_intervention",
        "claim_amount": 85000.00
    }
    
    # Process complex case
    claim = await app.process_claim_from_text(complex_medical_text, claimant_info)
    
    print("=== COMPLEX CASE ANALYSIS ===")
    print(f"Claim ID: {claim.claim_id}")
    print(f"Status: {claim.status.value}")
    
    if claim.health_analysis_result:
        health = claim.health_analysis_result
        print(f"\nHealth Analysis:")
        print(f"  Conditions: {health.medical_conditions}")
        print(f"  Medications: {health.medications}")
        print(f"  Risk Factors: {health.risk_factors}")
        print(f"  Confidence: {health.confidence_level.value}")
    
    if claim.underwriter_recommendation:
        rec = claim.underwriter_recommendation
        print(f"\nUnderwriting Recommendation:")
        print(f"  Decision: {rec.recommendation}")
        print(f"  Confidence: {rec.confidence_score:.2f}")
        print(f"  Risk Assessment: {rec.risk_assessment}")
        print(f"  Estimated Amount: ${rec.estimated_claim_amount:,.2f}" if rec.estimated_claim_amount else "N/A")
        
        if rec.red_flags:
            print(f"  Red Flags: {rec.red_flags}")
        if rec.required_actions:
            print(f"  Required Actions: {rec.required_actions}")

# Example 5: System Monitoring
async def example_system_monitoring():
    """System monitoring example."""
    from main import ClaimsHealthAnalyticsApp
    
    app = ClaimsHealthAnalyticsApp()
    
    # Get system status
    status = app.get_system_status()
    
    print("=== SYSTEM STATUS ===")
    print(json.dumps(status, indent=2))
    
    # Check agent availability
    agents = status.get('agents', {})
    print(f"\nActive Agents: {len(agents)}")
    for agent_name, agent_status in agents.items():
        print(f"  {agent_name}: {agent_status}")

# Example 6: Error Handling
async def example_error_handling():
    """Error handling example."""
    from main import ClaimsHealthAnalyticsApp
    from utils import HealthAnalyticsException
    
    app = ClaimsHealthAnalyticsApp()
    
    # Test with invalid input
    try:
        claimant_info = {
            "name": "Test Patient",
            "age": 25,
            "policy_number": "TEST-001"
        }
        
        # Empty text should trigger error handling
        claim = await app.process_claim_from_text("", claimant_info)
        
    except HealthAnalyticsException as e:
        print(f"Health analytics error handled: {e}")
    except Exception as e:
        print(f"General error handled: {e}")

# Example 7: Custom Configuration
async def example_custom_configuration():
    """Custom configuration example."""
    import os
    from main import ClaimsHealthAnalyticsApp
    
    # Override configuration
    os.environ["REASONING_MODEL"] = "gpt-4.1"
    os.environ["CONFIDENCE_THRESHOLD"] = "0.8"
    os.environ["TEMPERATURE"] = "0.2"
    
    app = ClaimsHealthAnalyticsApp()
    
    # Verify configuration
    config = app.agent_config
    print(f"Reasoning Model: {config.reasoning_model}")
    print(f"Confidence Threshold: {config.confidence_threshold}")
    print(f"Temperature: {config.temperature}")

async def run_all_examples():
    """Run all examples."""
    print("Claims Health Analytics - Usage Examples")
    print("=" * 50)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Batch Processing", example_batch_processing),
        ("File Processing", example_file_processing),
        ("Complex Medical Case", example_complex_case),
        ("System Monitoring", example_system_monitoring),
        ("Error Handling", example_error_handling),
        ("Custom Configuration", example_custom_configuration),
    ]
    
    for name, example_func in examples:
        print(f"\n--- {name} ---")
        try:
            await example_func()
            print("✅ Example completed successfully")
        except Exception as e:
            print(f"❌ Example failed: {e}")
        print()

if __name__ == "__main__":
    asyncio.run(run_all_examples())