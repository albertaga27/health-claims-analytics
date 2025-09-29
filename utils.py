"""
Utility functions for Claims Health Analytics system.
"""
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
from functools import wraps
import json

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('claims_analytics.log')
        ]
    )
    return logging.getLogger(__name__)

def generate_claim_id() -> str:
    """Generate a unique claim ID."""
    return f"CLM-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

def generate_document_id() -> str:
    """Generate a unique document ID."""
    return f"DOC-{uuid.uuid4().hex[:12].upper()}"

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            if hasattr(result, '__dict__') and hasattr(result, 'processing_time'):
                result.processing_time = execution_time
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {str(e)}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            if hasattr(result, '__dict__') and hasattr(result, 'processing_time'):
                result.processing_time = execution_time
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {str(e)}")
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def validate_file_type(file_path: str, allowed_types: List[str]) -> bool:
    """Validate if file type is allowed."""
    file_extension = file_path.lower().split('.')[-1]
    return file_extension in [ext.lower() for ext in allowed_types]

def sanitize_text(text: str) -> str:
    """Sanitize extracted text for processing."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '').replace('\ufffd', '')
    
    return text.strip()

def calculate_confidence_level(score: float) -> str:
    """Calculate confidence level based on numeric score."""
    if score >= 0.8:
        return "high"
    elif score >= 0.6:
        return "medium"
    else:
        return "low"

def extract_key_health_indicators(text: str) -> Dict[str, List[str]]:
    """Extract basic health indicators from text using simple pattern matching."""
    import re
    
    indicators = {
        "conditions": [],
        "medications": [],
        "symptoms": [],
        "procedures": []
    }
    
    # Simple pattern matching for common health terms
    condition_patterns = [
        r'\b(diabetes|hypertension|cancer|heart disease|stroke|asthma|copd)\b',
        r'\b(\w+itis)\b',  # conditions ending in -itis
        r'\b(\w+osis)\b',  # conditions ending in -osis
    ]
    
    medication_patterns = [
        r'\b(\w+cillin)\b',  # antibiotics
        r'\b(\w+pril)\b',    # ACE inhibitors
        r'\b(\w+ide)\b',     # diuretics
        r'\b(insulin|metformin|aspirin|ibuprofen)\b'
    ]
    
    symptom_patterns = [
        r'\b(pain|ache|fever|nausea|fatigue|shortness of breath|chest pain)\b',
        r'\b(headache|dizziness|cough|rash)\b'
    ]
    
    procedure_patterns = [
        r'\b(surgery|operation|biopsy|x-ray|mri|ct scan|blood test)\b',
        r'\b(\w+ectomy)\b',  # surgical procedures ending in -ectomy
        r'\b(\w+oscopy)\b'   # diagnostic procedures ending in -oscopy
    ]
    
    text_lower = text.lower()
    
    for pattern in condition_patterns:
        indicators["conditions"].extend(re.findall(pattern, text_lower, re.IGNORECASE))
    
    for pattern in medication_patterns:
        indicators["medications"].extend(re.findall(pattern, text_lower, re.IGNORECASE))
    
    for pattern in symptom_patterns:
        indicators["symptoms"].extend(re.findall(pattern, text_lower, re.IGNORECASE))
    
    for pattern in procedure_patterns:
        indicators["procedures"].extend(re.findall(pattern, text_lower, re.IGNORECASE))
    
    # Remove duplicates and clean up
    for key in indicators:
        indicators[key] = list(set([item.strip() for item in indicators[key] if item.strip()]))
    
    return indicators

def format_currency(amount: Optional[float]) -> str:
    """Format currency amount."""
    if amount is None:
        return "N/A"
    return f"${amount:,.2f}"

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

class HealthAnalyticsException(Exception):
    """Custom exception for health analytics errors."""
    pass

class DocumentProcessingException(Exception):
    """Custom exception for document processing errors."""
    pass

class UnderwritingException(Exception):
    """Custom exception for underwriting errors."""
    pass