#!/usr/bin/env python3
"""
Claims Health Analytics - Command Line Interface
Easy-to-use interface for processing claims with the new folder structure.
"""
import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from main import ClaimsHealthAnalyticsApp

def ensure_directories():
    """Ensure input and output directories exist."""
    Path("input/sample_claims").mkdir(parents=True, exist_ok=True)
    Path("output").mkdir(parents=True, exist_ok=True)

async def process_single_claim_from_file(medical_file: str, claimant_file: str = None):
    """Process a single claim from document files (PDF, DOCX, images, etc.)."""
    app = ClaimsHealthAnalyticsApp()
    
    # Validate that the medical file exists
    if not os.path.exists(medical_file):
        raise FileNotFoundError(f"Medical document file not found: {medical_file}")
    
    # Read claimant info if provided
    claimant_info = {}
    if claimant_file and os.path.exists(claimant_file):
        with open(claimant_file, 'r') as f:
            claimant_info = json.load(f)
    else:
        # Default claimant info
        claimant_info = {
            "name": "Sample Patient",
            "age": 45,
            "policy_number": f"POL-{datetime.now().strftime('%Y%m%d')}-001",
            "claim_type": "medical_consultation",
            "claim_amount": 5000.0
        }
    
    # Process claim from files using the Document Intelligence capabilities
    claim = await app.process_claim_from_files([medical_file], claimant_info)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(medical_file).stem
    output_path = f"output/claim_report_{base_name}_{timestamp}.json"
    app.save_claim_report(claim, output_path)
    
    print(f"✅ Claim processed successfully from file: {medical_file}")
    print(f"📄 Report saved to: {output_path}")
    print(f"🏥 Decision: {claim.underwriter_recommendation.recommendation}")
    print(f"💰 Estimated Amount: ${claim.underwriter_recommendation.estimated_claim_amount:,.2f}")
    
    return claim

async def process_single_claim(medical_file: str, claimant_file: str = None):
    """Process a single claim from input files."""
    app = ClaimsHealthAnalyticsApp()
    
    # Read medical record
    with open(medical_file, 'r') as f:
        medical_text = f.read()
    
    # Read claimant info if provided
    claimant_info = {}
    if claimant_file and os.path.exists(claimant_file):
        with open(claimant_file, 'r') as f:
            claimant_info = json.load(f)
    else:
        # Default claimant info
        claimant_info = {
            "name": "Sample Patient",
            "age": 45,
            "policy_number": f"POL-{datetime.now().strftime('%Y%m%d')}-001",
            "claim_type": "medical_consultation",
            "claim_amount": 5000.0
        }
    
    # Process claim
    claim = await app.process_claim_from_text(medical_text, claimant_info)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(medical_file).stem
    output_path = f"output/claim_report_{base_name}_{timestamp}.json"
    app.save_claim_report(claim, output_path)
    
    print(f"✅ Claim processed successfully!")
    print(f"📄 Report saved to: {output_path}")
    print(f"🏥 Decision: {claim.underwriter_recommendation.recommendation}")
    print(f"💰 Estimated Amount: ${claim.underwriter_recommendation.estimated_claim_amount:,.2f}")
    
    return claim

async def process_all_samples(mode='text'):
    """Process all sample claims in the input folder."""
    input_dir = Path("input/sample_claims")
    
    if mode == 'files':
        # Find all document files (PDF, DOCX, images)
        medical_files = []
        for pattern in ['*.pdf', '*.docx', '*.doc', '*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
            medical_files.extend(list(input_dir.glob(pattern)))
    else:
        # Find all medical record text files
        medical_files = list(input_dir.glob("*.txt"))
    
    if not medical_files:
        file_types = "document files (.pdf, .docx, .png, .jpg, etc.)" if mode == 'files' else "medical record files (.txt)"
        print(f"❌ No {file_types} found in input/sample_claims/")
        return
    
    app = ClaimsHealthAnalyticsApp()
    processed_claims = []
    
    print(f"🏥 Processing {len(medical_files)} medical records using {mode} mode...")
    print()
    
    for medical_file in medical_files:
        print(f"Processing: {medical_file.name}")
        
        # Look for corresponding claimant file
        base_name = medical_file.stem.replace("medical_record", "claimant")
        claimant_file = input_dir / f"{base_name}.json"
        
        try:
            if mode == 'files':
                claim = await process_single_claim_from_file(str(medical_file), str(claimant_file) if claimant_file.exists() else None)
            else:
                claim = await process_single_claim(str(medical_file), str(claimant_file) if claimant_file.exists() else None)
            processed_claims.append(claim)
            print(f"  ✅ {claim.underwriter_recommendation.recommendation}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    # Generate summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"output/batch_summary_{mode}_{timestamp}.json"
    
    summary = {
        "processing_timestamp": datetime.now().isoformat(),
        "total_claims": len(processed_claims),
        "approved": sum(1 for c in processed_claims if c.underwriter_recommendation.recommendation == "APPROVE"),
        "rejected": sum(1 for c in processed_claims if c.underwriter_recommendation.recommendation == "REJECT"),
        "pending": sum(1 for c in processed_claims if c.underwriter_recommendation.recommendation == "REQUEST_MORE_INFO"),
        "claims": [
            {
                "claim_id": c.claim_id,
                "claimant": c.claimant_info.get("name", "Unknown"),
                "decision": c.underwriter_recommendation.recommendation,
                "amount": c.underwriter_recommendation.estimated_claim_amount,
                "confidence": c.underwriter_recommendation.confidence_score
            } for c in processed_claims
        ]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📊 Batch Summary:")
    print(f"  Total Claims: {summary['total_claims']}")
    print(f"  ✅ Approved: {summary['approved']}")
    print(f"  ❌ Rejected: {summary['rejected']}")
    print(f"  ⏳ Pending: {summary['pending']}")
    print(f"  📄 Summary saved to: {summary_path}")

async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Claims Health Analytics CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sample claims from text files
  python runner.py --all --mode text
  
  # Process all sample document files (PDF, DOCX, images)
  python runner.py --all --mode files
  
  # Process specific text-based claim
  python runner.py --medical input/sample_claims/sample_medical_record.txt --mode text
  
  # Process specific document file (PDF, DOCX, etc.)
  python runner.py --medical input/eyes_surgery_pre_1_4.pdf --mode files
  
  # Process with custom claimant info
  python runner.py --medical input/sample_claims/sample_medical_record.txt --claimant input/sample_claims/sample_claimant.json --mode text
  
  # Run demo
  python runner.py --demo
        """
    )
    
    parser.add_argument("--mode", choices=["text", "files"], default="text", 
                       help="Processing mode: 'text' for .txt files, 'files' for documents (PDF, DOCX, images)")
    parser.add_argument("--all", action="store_true", help="Process all medical records in input/sample_claims/")
    parser.add_argument("--medical", help="Path to medical record file (.txt) or document file (.pdf, .docx, etc.)")
    parser.add_argument("--claimant", help="Path to claimant info JSON file")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    try:
        if args.demo:
            print("🏥 Running Claims Health Analytics Demo...")
            from demo import demo_basic_functionality
            await demo_basic_functionality()
        
        elif args.all:
            await process_all_samples(args.mode)
        
        elif args.medical:
            if not os.path.exists(args.medical):
                print(f"❌ Medical record file not found: {args.medical}")
                return 1
            
            if args.mode == "files":
                await process_single_claim_from_file(args.medical, args.claimant)
            else:
                await process_single_claim(args.medical, args.claimant)
        
        else:
            print("❌ Please specify an action. Use --help for usage information.")
            print()
            print("Quick start:")
            print("  python runner.py --demo                    # Run demo")
            print("  python runner.py --all --mode text         # Process all text samples")
            print("  python runner.py --all --mode files        # Process all document samples")
            return 1
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))