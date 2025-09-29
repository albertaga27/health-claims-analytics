"""
Data models for Claims Health Analytics system.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import json

class ClaimStatus(Enum):
    """Claim processing status."""
    SUBMITTED = "submitted"
    DOCUMENT_EXTRACTION = "document_extraction"
    HEALTH_ANALYSIS = "health_analysis"
    UNDERWRITER_REVIEW = "underwriter_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_ADDITIONAL_INFO = "requires_additional_info"

class HealthEntityType(Enum):
    """Health entity types from Azure Text Analytics for Health."""
    CONDITION_QUALIFIER = "ConditionQualifier"
    DIAGNOSIS = "Diagnosis"
    SYMPTOM_OR_SIGN = "SymptomOrSign"
    BODY_STRUCTURE = "BodyStructure"
    MEDICATION_CLASS = "MedicationClass"
    MEDICATION_NAME = "MedicationName"
    DOSAGE = "Dosage"
    MEDICATION_FORM = "MedicationForm"
    MEDICATION_ROUTE = "MedicationRoute"
    FREQUENCY = "Frequency"
    RELATION_TYPE = "RelationType"
    TIME = "Time"
    GENE_OR_PROTEIN = "GeneOrProtein"
    VARIANT = "Variant"

class ConfidenceLevel(Enum):
    """Confidence levels for analysis results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class DocumentMetadata:
    """Metadata for uploaded documents."""
    file_name: str
    file_type: str
    file_size: int
    upload_time: datetime
    document_id: str
    pages: int = 1

@dataclass
class ExtractedEntity:
    """Extracted entity from document."""
    text: str
    category: str
    confidence_score: float
    offset: int
    length: int
    subcategory: Optional[str] = None

@dataclass
class HealthEntity:
    """Health-specific entity from Azure Text Analytics for Health."""
    text: str
    category: HealthEntityType
    confidence_score: float
    offset: int
    length: int
    is_negated: bool = False
    subcategory: Optional[str] = None
    assertion: Optional[Dict[str, Any]] = None
    data_sources: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class HealthRelation:
    """Relationship between health entities."""
    relation_type: str
    source_entity: HealthEntity
    target_entity: HealthEntity
    confidence_score: float

@dataclass
class DocumentExtractionResult:
    """Result from document extraction agent."""
    document_metadata: DocumentMetadata
    extracted_text: str
    extracted_entities: List[ExtractedEntity]
    key_value_pairs: Dict[str, str]
    tables: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    errors: List[str] = field(default_factory=list)

@dataclass
class HealthAnalysisResult:
    """Result from health analytics agent."""
    health_entities: List[HealthEntity]
    health_relations: List[HealthRelation]
    medical_conditions: List[str]
    medications: List[str]
    symptoms: List[str]
    procedures: List[str]
    overall_health_assessment: str
    risk_factors: List[str]
    confidence_level: ConfidenceLevel
    processing_time: float

@dataclass
class UnderwriterRecommendation:
    """Recommendation from reasoning agent for underwriters."""
    recommendation: str  # APPROVE, REJECT, REQUEST_MORE_INFO
    confidence_score: float
    reasoning: str
    risk_assessment: str
    required_actions: List[str]
    red_flags: List[str]
    supporting_evidence: List[str]
    estimated_claim_amount: Optional[float] = None
    approval_conditions: List[str] = field(default_factory=list)

@dataclass
class ClaimSubmission:
    """Complete claim submission with all analysis results."""
    claim_id: str
    status: ClaimStatus
    submitted_at: datetime
    claimant_info: Dict[str, Any]
    documents: List[DocumentMetadata]
    
    # Analysis results
    document_extraction_results: List[DocumentExtractionResult] = field(default_factory=list)
    health_analysis_result: Optional[HealthAnalysisResult] = None
    underwriter_recommendation: Optional[UnderwriterRecommendation] = None
    
    # Processing metadata
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    assigned_underwriter: Optional[str] = None
    
    def add_processing_step(self, step_name: str, result: str, duration: float):
        """Add a processing step to the history."""
        self.processing_history.append({
            "step": step_name,
            "result": result,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert claim to dictionary for serialization."""
        return {
            "claim_id": self.claim_id,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat(),
            "claimant_info": self.claimant_info,
            "documents": [doc.__dict__ for doc in self.documents],
            "document_extraction_results": [result.__dict__ for result in self.document_extraction_results],
            "health_analysis_result": self.health_analysis_result.__dict__ if self.health_analysis_result else None,
            "underwriter_recommendation": self.underwriter_recommendation.__dict__ if self.underwriter_recommendation else None,
            "processing_history": self.processing_history,
            "last_updated": self.last_updated.isoformat(),
            "assigned_underwriter": self.assigned_underwriter
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimSubmission":
        """Create claim from dictionary."""
        # This would need proper deserialization logic
        # For now, basic implementation
        return cls(
            claim_id=data["claim_id"],
            status=ClaimStatus(data["status"]),
            submitted_at=datetime.fromisoformat(data["submitted_at"]),
            claimant_info=data["claimant_info"],
            documents=[DocumentMetadata(**doc) for doc in data.get("documents", [])],
            processing_history=data.get("processing_history", []),
            last_updated=datetime.fromisoformat(data.get("last_updated", datetime.now().isoformat())),
            assigned_underwriter=data.get("assigned_underwriter")
        )