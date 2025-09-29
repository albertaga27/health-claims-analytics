# Input Data Structure

This folder contains sample claim data for testing the Claims Health Analytics system.

## File Organization

### `/sample_claims/`
Contains various types of sample claims for testing different scenarios:

#### Medical Records (Text Files)
- `medical_record_doe.txt` - Basic diabetes/hypertension case
- `medical_record_johnson.txt` - Complex COPD respiratory case
- `medical_record_chen.txt` - Routine preventive care visit

#### Claimant Information (JSON Files)
- `claimant_doe.json` - Basic claimant data
- `claimant_johnson.json` - Complex respiratory claim
- `claimant_chen.json` - Routine care claim

#### Batch Processing
- `sample_batch_claims.json` - Multiple claims for batch processing

## Usage Examples

### Single Claim Processing
```python
# Process a single claim with medical record i.e. John Doe data:
python main.py --medical-record input/sample_claims/medical_record_doe.txt --claimant input/sample_claims/claimant_doe.json

# Process using text content directly
python demo.py
```

### Batch Processing
```python
# Process multiple claims
python main.py --batch input/sample_claims/sample_batch_claims.json
```

## Adding New Test Data

To add new test claims:

1. **Medical Records**: Create `.txt` files with medical history, symptoms, medications, etc.
2. **Claimant Data**: Create `.json` files with policy information, claim amounts, diagnosis codes
3. **Document Files**: Add `.pdf`, `.docx`, or image files to test document extraction

## Data Privacy

All sample data in this folder contains fictional patient information and should not contain any real PHI (Protected Health Information).