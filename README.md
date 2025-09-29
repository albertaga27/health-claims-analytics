# Claims Health Analytics System

A comprehensive multi-agent solution for automating health insurance claims processing using Azure AI services and OpenAI GPT models.

## 📁 Project Structure

```
claims-health-analytics/
├── input/                          # Input data and sample claims
│   ├── sample_claims/              # Sample medical records and claimant data
│   │   ├── sample_medical_record.txt
│   │   ├── medical_record_complex.txt
│   │   ├── medical_record_routine.txt
│   │   ├── sample_claimant.json
│   │   ├── claimant_complex.json
│   │   ├── claimant_routine.json
│   │   └── sample_batch_claims.json
│   └── README.md                   # Input data documentation
├── output/                         # Processing results and reports
│   ├── claim_report_*.json         # Individual claim reports
│   ├── batch_summary_*.json        # Batch processing summaries
│   └── README.md                   # Output documentation
├── agents/                         # AI agent implementations
│   ├── base_agent.py              # Base agent class and orchestrator
│   ├── document_extraction_agent.py
│   ├── health_analytics_agent.py
│   └── underwriting_agent.py
├── models.py                       # Data models and schemas
├── config.py                       # Configuration management
├── utils.py                        # Utility functions
├── main.py                         # Main application
├── demo.py                         # Simple demo script
├── runner.py                          # Command-line interface
├── .env.sample                     # Environment configuration template
└── README.md                       # This file
```

## Overview

This system uses three specialized AI agents to automate the claims processing workflow:

1. **Document Extraction Agent** - Extracts and analyzes medical documents using Azure Document Intelligence and GPT models
2. **Health Analytics Agent** - Performs medical entity extraction and health analysis using Azure Text Analytics for Health
3. **Underwriting Reasoning Agent** - Provides intelligent underwriting recommendations using GPT reasoning capabilities

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Document      │    │   Health         │    │   Underwriting      │
│   Extraction    │───▶│   Analytics      │───▶│   Reasoning         │
│   Agent         │    │   Agent          │    │   Agent             │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Azure Document  │    │ Azure Text       │    │ OpenAI GPT Models   │
│ Intelligence    │    │ Analytics for    │    │ (Reasoning &        │
│ + GPT Models    │    │ Health           │    │ Decision Making)    │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

## Features

### Document Extraction Agent
- **Multi-format Support**: PDF, DOCX, TXT, and image files (PNG, JPG, TIFF)
- **Azure Document Intelligence**: Advanced OCR and document understanding
- **GPT-Enhanced Analysis**: Intelligent entity extraction and categorization
- **Fallback Processing**: Graceful degradation when services are unavailable

### Health Analytics Agent
- **Azure Text Analytics for Health**: Medical entity recognition and relationships
- **Medical Entity Extraction**: Conditions, medications, symptoms, procedures
- **Risk Factor Identification**: Automatic detection of high-risk conditions
- **Confidence Scoring**: Reliability assessment for extracted information

### Underwriting Reasoning Agent
- **Intelligent Decision Making**: GPT-powered underwriting recommendations
- **Risk Assessment**: Comprehensive evaluation of medical and financial risk
- **Automated Recommendations**: APPROVE, REJECT, or REQUEST_MORE_INFO
- **Detailed Justification**: Clear reasoning for underwriting decisions

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /home/aga/azureai/ai-agents/Labfiles/13-claims-health-analytics
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.sample .env
   # Edit .env with your Azure service credentials
   ```

## Configuration

### Required Azure Services

1. **Azure OpenAI Service**:
   - Endpoint URL
   - API Key or Managed Identity
   - Deployed models (gpt-4.1, gpt-4.1-mini recommended)

2. **Azure Text Analytics for Health** (optional):
   - Endpoint URL
   - API Key
   - Enables advanced medical entity extraction

3. **Azure Document Intelligence** (optional):
   - Endpoint URL
   - API Key
   - Enables advanced document processing

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# Required - Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini
AZURE_OPENAI_API_KEY=your-api-key

# Optional - Text Analytics for Health
AZURE_TEXT_ANALYTICS_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_TEXT_ANALYTICS_KEY=your-key

# Optional - Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key

# Agent Configuration
REASONING_MODEL=gpt-4.1
CONFIDENCE_THRESHOLD=0.7
```

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
# Navigate to project directory
cd /home/aga/azureai/ai-agents/Labfiles/13-claims-health-analytics

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.sample .env
# Edit .env with your Azure credentials
```

### 2. **Run Demo**
```bash
# Quick demo with sample data
python runner.py --demo

# Or use the standalone demo
python demo.py
```

### 3. **Process Sample Claims**
```bash
# Process all sample claims
python runner.py --all

# Process specific claim
python runner.py --medical input/sample_claims/sample_medical_record.txt
```

## Usage

### Command Line Interface (Recommended)

The new CLI provides an easy-to-use interface:

```bash
# Process all sample claims in input/sample_claims/
python runner.py --all

# Process specific medical record
python runner.py --medical input/sample_claims/medical_record_complex.txt

# Process with custom claimant info
python runner.py --medical input/sample_claims/sample_medical_record.txt --claimant input/sample_claims/sample_claimant.json

# Run interactive demo
python runner.py --demo
```

### Traditional Interface

#### Demo Mode (Recommended for First Run)
```bash
python main.py --mode demo
```

#### Process Text Content
```bash
python main.py --mode text --text "Patient medical history..." --claimant input/sample_claims/sample_claimant.json
```

#### Process Document Files
```bash
python main.py --mode files --documents medical_record.pdf --claimant input/sample_claims/sample_claimant.json
```

#### Batch Processing
```bash
python main.py --batch input/sample_claims/sample_batch_claims.json
```


## Output Format

The system generates comprehensive reports in JSON format:

```json
{
  "claim_summary": {
    "claim_id": "CLM-20241223-A1B2C3D4",
    "status": "approved",
    "processing_duration": 5.23
  },
  "health_analysis": {
    "medical_conditions": ["Type 2 diabetes", "Hypertension"],
    "medications": ["Metformin", "Lisinopril"],
    "risk_factors": ["High-risk condition: diabetes"],
    "confidence_level": "high"
  },
  "underwriter_recommendation": {
    "recommendation": "APPROVE",
    "confidence_score": 0.85,
    "risk_assessment": "Medium risk profile",
    "estimated_claim_amount": 12500.00
  }
}
```

## Error Handling

The system includes comprehensive error handling:

- **Graceful Degradation**: Falls back to alternative processing methods
- **Detailed Logging**: Comprehensive logs for debugging and monitoring
- **Exception Management**: Custom exceptions for different error types
- **Retry Logic**: Automatic retry with exponential backoff

## Performance Considerations

- **Async Processing**: Non-blocking operations for better throughput
- **Batch Processing**: Efficient handling of multiple claims
- **Token Management**: Automatic text truncation to stay within API limits
- **Caching**: Intelligent caching of results where appropriate

## Security

- **Credential Management**: Uses Azure managed identities when possible
- **Data Sanitization**: Automatic cleaning of sensitive information
- **Audit Trail**: Complete processing history for compliance
- **Access Control**: Configurable access controls for different components

## Monitoring and Observability

- **Processing History**: Complete audit trail for each claim
- **Performance Metrics**: Timing and success rate tracking
- **Health Checks**: System status monitoring
- **Error Tracking**: Detailed error logging and reporting

## Customization

### Adding New Agents
```python
from base_agent import BaseAgent

class CustomAgent(BaseAgent):
    async def process(self, input_data):
        # Your custom processing logic
        return result

# Register with orchestrator
orchestrator.register_agent("custom", CustomAgent(config))
```

### Custom Models
Update the configuration to use different AI models:
```bash
DOCUMENT_EXTRACTION_MODEL=gpt-4.1-mini
HEALTH_ANALYTICS_MODEL=text-analytics-health
REASONING_MODEL=gpt-4.1
```


### Debugging

Enable detailed logging:
```bash
LOG_LEVEL=DEBUG python main.py --mode demo
```

Check log files:
- `claims_analytics.log` - Application logs
- `demo_claim_report.json` - Detailed processing results

## Contributing

To extend the system:

1. Follow the existing agent pattern in `base_agent.py`
2. Add comprehensive error handling
3. Include unit tests for new functionality
4. Update documentation and examples

## License

This project is licensed under the MIT License. See the LICENSE file for details.

