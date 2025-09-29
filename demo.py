#!/usr/bin/env python3
"""
Simple demo script for Claims Health Analytics system.
This script demonstrates basic functionality without requiring full Azure setup.
"""
import asyncio
import json
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

async def demo_basic_functionality():
    """Demo basic functionality without full Azure setup."""
    print("Claims Health Analytics - Basic Demo")
    print("=" * 50)
    
    # Sample medical text
    medical_text = """
    Patient: John Doe, Age 45
    
    Medical History:
    - Type 2 diabetes diagnosed in 2020
    - Hypertension managed with Lisinopril 10mg daily
    - Recent chest pain episodes during exercise
    
    Current Symptoms:
    - Chest discomfort with exertion
    - Shortness of breath
    - Fatigue
    
    Current Medications:
    - Metformin 1000mg twice daily
    - Lisinopril 10mg daily
    - Aspirin 81mg daily
    - Atorvastatin 20mg daily
    
    Recent Tests:
    - Cardiac stress test scheduled
    - HbA1c: 7.8%
    - Blood pressure: 142/88 mmHg
    
    Assessment:
    Patient with diabetes and hypertension presenting with cardiac symptoms.
    Recommend cardiology consultation and diabetes management optimization.
    """
    
    claimant_info = {
        "name": "John Doe",
        "age": 45,
        "policy_number": "POL-2024-001",
        "claim_type": "medical_consultation",
        "claim_amount": 5000.00
    }
    
    print("Sample Medical Text:")
    print("-" * 30)
    print(medical_text[:300] + "...")
    print()
    
    print("Claimant Information:")
    print("-" * 30)
    print(json.dumps(claimant_info, indent=2))
    print()
    
    # Try to import and test the application
    try:
        from main import ClaimsHealthAnalyticsApp
        
        print("Initializing Claims Health Analytics system...")
        app = ClaimsHealthAnalyticsApp()
        
        print("Processing sample claim...")
        claim = await app.process_claim_from_text(medical_text, claimant_info)
        
        print("\n" + "=" * 50)
        print("PROCESSING RESULTS")
        print("=" * 50)
        
        print(f"Claim ID: {claim.claim_id}")
        print(f"Status: {claim.status.value}")
        print()
        
        if claim.health_analysis_result:
            health = claim.health_analysis_result
            print("Health Analysis Results:")
            print(f"  Medical Conditions: {', '.join(health.medical_conditions) if health.medical_conditions else 'None detected'}")
            print(f"  Medications: {', '.join(health.medications) if health.medications else 'None detected'}")
            print(f"  Symptoms: {', '.join(health.symptoms) if health.symptoms else 'None detected'}")
            print(f"  Risk Factors: {', '.join(health.risk_factors) if health.risk_factors else 'None identified'}")
            print(f"  Overall Assessment: {health.overall_health_assessment}")
            print(f"  Confidence Level: {health.confidence_level.value}")
            print()
        
        if claim.underwriter_recommendation:
            rec = claim.underwriter_recommendation
            print("Underwriting Recommendation:")
            print(f"  Decision: {rec.recommendation}")
            print(f"  Confidence Score: {rec.confidence_score:.2f}")
            if rec.estimated_claim_amount:
                print(f"  Estimated Claim Amount: ${rec.estimated_claim_amount:,.2f}")
            print(f"  Risk Assessment: {rec.risk_assessment}")
            print()
            
            if rec.red_flags:
                print("  Red Flags Identified:")
                for flag in rec.red_flags:
                    print(f"    • {flag}")
                print()
            
            if rec.required_actions:
                print("  Required Actions:")
                for action in rec.required_actions:
                    print(f"    • {action}")
                print()
            
            if rec.approval_conditions:
                print("  Approval Conditions:")
                for condition in rec.approval_conditions:
                    print(f"    • {condition}")
                print()
        
        print("Processing History:")
        total_time = 0
        for step in claim.processing_history:
            print(f"  {step['step']}: {step['result']} (Duration: {step.get('duration', 0):.2f}s)")
            total_time += step.get('duration', 0)
        print(f"  Total Processing Time: {total_time:.2f}s")
        print()
        
        # Save detailed report to output folder
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"output/demo_claim_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(claim.to_dict(), f, indent=2, default=str)
        
        print(f"Detailed report saved to: {report_file}")
        print()
        
        # Show system status
        status = app.get_system_status()
        print("System Status:")
        print(f"  Overall Status: {status['status']}")
        print(f"  Active Agents: {', '.join(status['agents'].keys())}")
        print(f"  Configuration: {status['configuration']['reasoning_model']} (reasoning)")
        print()
        
        print("Demo completed successfully! ✅")
        return True
        
    except ImportError as e:
        print(f"⚠️  Module import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print()
        print("This might be due to missing Azure configuration.")
        print("Please check your .env file and ensure Azure services are properly configured.")
        print()
        print("For testing without Azure services, you can:")
        print("1. Set up a minimal Azure OpenAI endpoint")
        print("2. Use the fallback processing modes")
        print("3. Review the sample outputs in the documentation")
        return False

def show_configuration_help():
    """Show configuration help."""
    print("\nConfiguration Help:")
    print("-" * 20)
    print("To run this demo with full functionality, you need:")
    print()
    print("1. Azure OpenAI Service (Required):")
    print("   - Create an Azure OpenAI resource")
    print("   - Deploy a GPT-4 model (gpt-4.1 or gpt-4.1-mini recommended)")
    print("   - Get endpoint URL and API key")
    print()
    print("2. Azure Text Analytics for Health (Optional):")
    print("   - Create a Text Analytics resource")
    print("   - Enable Health feature")
    print("   - Get endpoint URL and API key")
    print()
    print("3. Azure Document Intelligence (Optional):")
    print("   - Create a Document Intelligence resource")
    print("   - Get endpoint URL and API key")
    print()
    print("4. Configure .env file:")
    print("   cp .env.sample .env")
    print("   # Edit .env with your Azure credentials")
    print()
    print("5. Run the demo:")
    print("   python demo.py")

async def main():
    """Main demo function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_configuration_help()
        return
    
    success = await demo_basic_functionality()
    
    if not success:
        print()
        show_configuration_help()
        return 1
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())