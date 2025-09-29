"""
Health Analytics Agent for Claims Health Analytics.
Uses Azure Text Analytics for Health to extract medical entities and relationships.
"""
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

from base_agent import BaseAgent
from models import (
    HealthEntity, HealthRelation, HealthAnalysisResult, HealthEntityType, ConfidenceLevel
)
from utils import (
    timing_decorator, calculate_confidence_level, extract_key_health_indicators,
    HealthAnalyticsException
)
from config import AzureConfig, AgentConfig

class HealthAnalyticsAgent(BaseAgent):
    """Agent responsible for health-specific text analytics and medical entity extraction."""
    
    def __init__(self, azure_config: AzureConfig, agent_config: AgentConfig):
        """Initialize the Health Analytics Agent."""
        super().__init__(azure_config, agent_config)
        self.text_analytics_client = self._setup_text_analytics_client()
    
    def _setup_text_analytics_client(self) -> Optional[TextAnalyticsClient]:
        """Set up the Text Analytics client."""
        try:
            if (self.azure_config.text_analytics_endpoint and 
                self.azure_config.text_analytics_key):
                return TextAnalyticsClient(
                    endpoint=self.azure_config.text_analytics_endpoint,
                    credential=AzureKeyCredential(self.azure_config.text_analytics_key)
                )
            else:
                self.logger.warning("Text Analytics for Health not configured, using fallback processing")
                return None
        except Exception as e:
            self.logger.warning(f"Text Analytics setup failed: {e}, using fallback processing")
            return None
    
    @timing_decorator
    async def process(self, input_data: str) -> HealthAnalysisResult:
        """Process text and extract health-specific information."""
        try:
            if not input_data or not input_data.strip():
                raise HealthAnalyticsException("Input text is empty or invalid")
            
            # Limit text size to avoid API limits
            text = input_data[:5000] if len(input_data) > 5000 else input_data
            
            if self.text_analytics_client:
                return await self._process_with_text_analytics(text)
            else:
                return await self._process_with_fallback(text)
                
        except Exception as e:
            self.logger.error(f"Health analytics processing failed: {e}")
            raise HealthAnalyticsException(f"Health analytics processing failed: {e}")
    
    async def _process_with_text_analytics(self, text: str) -> HealthAnalysisResult:
        """Process text using Azure Text Analytics for Health."""
        try:
            # Analyze text for healthcare entities
            documents = [text]
            poller = self.text_analytics_client.begin_analyze_healthcare_entities(documents)
            healthcare_results = poller.result()
            
            health_entities = []
            health_relations = []
            medical_conditions = []
            medications = []
            symptoms = []
            procedures = []
            risk_factors = []
            
            for doc_result in healthcare_results:
                if doc_result.is_error:
                    raise HealthAnalyticsException(f"Text Analytics error: {doc_result.error}")
                
                # Process healthcare entities
                for entity in doc_result.entities:
                    health_entity = self._convert_healthcare_entity(entity)
                    health_entities.append(health_entity)
                    
                    # Categorize entities
                    self._categorize_entity(entity, medical_conditions, medications, symptoms, procedures, risk_factors)
                
                # Process healthcare relations
                for relation in doc_result.entity_relations:
                    health_relation = self._convert_healthcare_relation(relation, health_entities)
                    if health_relation:
                        health_relations.append(health_relation)
            
            # Generate overall health assessment
            overall_assessment = await self._generate_health_assessment(
                medical_conditions, medications, symptoms, procedures, risk_factors
            )
            
            # Calculate confidence level
            avg_confidence = sum(entity.confidence_score for entity in health_entities) / len(health_entities) if health_entities else 0.0
            confidence_level = ConfidenceLevel(calculate_confidence_level(avg_confidence))
            
            return HealthAnalysisResult(
                health_entities=health_entities,
                health_relations=health_relations,
                medical_conditions=list(set(medical_conditions)),
                medications=list(set(medications)),
                symptoms=list(set(symptoms)),
                procedures=list(set(procedures)),
                overall_health_assessment=overall_assessment,
                risk_factors=list(set(risk_factors)),
                confidence_level=confidence_level,
                processing_time=0.0  # Set by timing decorator
            )
            
        except Exception as e:
            raise HealthAnalyticsException(f"Text Analytics processing failed: {e}")
    
    async def _process_with_fallback(self, text: str) -> HealthAnalysisResult:
        """Process text using fallback pattern-based extraction."""
        try:
            # Use simple pattern matching as fallback
            health_indicators = extract_key_health_indicators(text)
            
            # Create basic health entities from patterns
            health_entities = []
            offset = 0
            
            for category, items in health_indicators.items():
                for item in items:
                    entity = HealthEntity(
                        text=item,
                        category=self._map_category_to_health_type(category),
                        confidence_score=0.6,  # Lower confidence for pattern matching
                        offset=text.lower().find(item.lower(), offset),
                        length=len(item),
                        is_negated=False
                    )
                    health_entities.append(entity)
                    offset += len(item)
            
            # Generate assessment
            overall_assessment = await self._generate_fallback_assessment(health_indicators)
            
            return HealthAnalysisResult(
                health_entities=health_entities,
                health_relations=[],  # No relations in fallback mode
                medical_conditions=health_indicators.get("conditions", []),
                medications=health_indicators.get("medications", []),
                symptoms=health_indicators.get("symptoms", []),
                procedures=health_indicators.get("procedures", []),
                overall_health_assessment=overall_assessment,
                risk_factors=self._identify_risk_factors(health_indicators),
                confidence_level=ConfidenceLevel.MEDIUM,
                processing_time=0.0  # Set by timing decorator
            )
            
        except Exception as e:
            raise HealthAnalyticsException(f"Fallback processing failed: {e}")
    
    def _convert_healthcare_entity(self, entity: Any) -> HealthEntity:
        """Convert Azure healthcare entity to our health entity model."""
        # Map Azure entity categories to our enum
        category_mapping = {
            "ConditionQualifier": HealthEntityType.CONDITION_QUALIFIER,
            "Diagnosis": HealthEntityType.DIAGNOSIS,
            "SymptomOrSign": HealthEntityType.SYMPTOM_OR_SIGN,
            "BodyStructure": HealthEntityType.BODY_STRUCTURE,
            "MedicationClass": HealthEntityType.MEDICATION_CLASS,
            "MedicationName": HealthEntityType.MEDICATION_NAME,
            "Dosage": HealthEntityType.DOSAGE,
            "MedicationForm": HealthEntityType.MEDICATION_FORM,
            "MedicationRoute": HealthEntityType.MEDICATION_ROUTE,
            "Frequency": HealthEntityType.FREQUENCY,
            "RelationType": HealthEntityType.RELATION_TYPE,
            "Time": HealthEntityType.TIME,
            "GeneOrProtein": HealthEntityType.GENE_OR_PROTEIN,
            "Variant": HealthEntityType.VARIANT,
        }
        
        entity_type = category_mapping.get(getattr(entity, 'category', 'Diagnosis'), HealthEntityType.DIAGNOSIS)
        
        # Extract assertion information
        assertion = {}
        if hasattr(entity, 'assertion') and entity.assertion:
            assertion = {
                "certainty": getattr(entity.assertion, 'certainty', None),
                "conditionality": getattr(entity.assertion, 'conditionality', None),
                "association": getattr(entity.assertion, 'association', None)
            }
        
        # Extract data sources
        data_sources = []
        if hasattr(entity, 'data_sources') and entity.data_sources:
            data_sources = [{"name": getattr(source, 'name', ''), "entity_id": getattr(source, 'entity_id', '')} for source in entity.data_sources]
        
        return HealthEntity(
            text=getattr(entity, 'text', ''),
            category=entity_type,
            confidence_score=getattr(entity, 'confidence_score', 0.0),
            offset=getattr(entity, 'offset', 0),
            length=getattr(entity, 'length', 0),
            is_negated=assertion.get("certainty") == "negative" if assertion else False,
            subcategory=getattr(entity, 'subcategory', None),
            assertion=assertion,
            data_sources=data_sources
        )
    
    def _convert_healthcare_relation(self, relation: Any, entities: List[HealthEntity]) -> Optional[HealthRelation]:
        """Convert Azure healthcare relation to our health relation model."""
        try:
            # Azure healthcare relations use roles instead of source/target
            if not hasattr(relation, 'roles') or len(relation.roles) < 2:
                return None
            
            # Extract entities from roles
            role_entities = []
            for role in relation.roles:
                # Find matching entity in our converted list
                for entity in entities:
                    if (entity.offset == role.entity.offset and 
                        entity.length == role.entity.length and
                        entity.text == role.entity.text):
                        role_entities.append(entity)
                        break
            
            # We need at least 2 entities to form a relation
            if len(role_entities) >= 2:
                return HealthRelation(
                    relation_type=relation.relation_type,
                    source_entity=role_entities[0],  # First role as source
                    target_entity=role_entities[1],  # Second role as target
                    confidence_score=relation.confidence_score
                )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to convert healthcare relation: {e}")
            return None
    
    def _categorize_entity(self, entity: Any, conditions: List[str], 
                          medications: List[str], symptoms: List[str], 
                          procedures: List[str], risk_factors: List[str]):
        """Categorize healthcare entity into appropriate lists."""
        entity_text = entity.text.lower()
        
        if entity.category in ["Diagnosis", "ConditionQualifier"]:
            conditions.append(entity.text)
            
            # Check for high-risk conditions
            high_risk_keywords = ["cancer", "heart disease", "diabetes", "stroke", "chronic"]
            if any(keyword in entity_text for keyword in high_risk_keywords):
                risk_factors.append(f"High-risk condition: {entity.text}")
        
        elif entity.category in ["MedicationName", "MedicationClass"]:
            medications.append(entity.text)
        
        elif entity.category == "SymptomOrSign":
            symptoms.append(entity.text)
            
            # Check for severe symptoms
            severe_symptoms = ["severe pain", "chest pain", "difficulty breathing", "loss of consciousness"]
            if any(symptom in entity_text for symptom in severe_symptoms):
                risk_factors.append(f"Severe symptom: {entity.text}")
        
        elif "procedure" in entity.category.lower() or "surgery" in entity_text:
            procedures.append(entity.text)
    
    def _map_category_to_health_type(self, category: str) -> HealthEntityType:
        """Map simple category to health entity type."""
        mapping = {
            "conditions": HealthEntityType.DIAGNOSIS,
            "medications": HealthEntityType.MEDICATION_NAME,
            "symptoms": HealthEntityType.SYMPTOM_OR_SIGN,
            "procedures": HealthEntityType.BODY_STRUCTURE
        }
        return mapping.get(category, HealthEntityType.DIAGNOSIS)
    
    async def _generate_health_assessment(self, conditions: List[str], medications: List[str], 
                                        symptoms: List[str], procedures: List[str], 
                                        risk_factors: List[str]) -> str:
        """Generate overall health assessment based on extracted entities."""
        assessment_parts = []
        
        if conditions:
            severity = "high" if len(conditions) > 3 else "moderate" if len(conditions) > 1 else "low"
            assessment_parts.append(f"Medical conditions identified: {len(conditions)} ({severity} complexity)")
        
        if medications:
            assessment_parts.append(f"Medications documented: {len(medications)}")
        
        if symptoms:
            assessment_parts.append(f"Symptoms reported: {len(symptoms)}")
        
        if procedures:
            assessment_parts.append(f"Medical procedures: {len(procedures)}")
        
        if risk_factors:
            assessment_parts.append(f"Risk factors identified: {len(risk_factors)}")
        
        if not assessment_parts:
            return "Limited health information available for assessment"
        
        return ". ".join(assessment_parts) + "."
    
    async def _generate_fallback_assessment(self, health_indicators: Dict[str, List[str]]) -> str:
        """Generate assessment using fallback method."""
        total_indicators = sum(len(items) for items in health_indicators.values())
        
        if total_indicators == 0:
            return "No significant health indicators identified in the document"
        elif total_indicators < 3:
            return "Limited health information extracted from document"
        elif total_indicators < 6:
            return "Moderate amount of health information identified"
        else:
            return "Comprehensive health information extracted from document"
    
    def _identify_risk_factors(self, health_indicators: Dict[str, List[str]]) -> List[str]:
        """Identify risk factors from health indicators."""
        risk_factors = []
        
        # Check for high-risk conditions
        high_risk_conditions = ["cancer", "heart disease", "diabetes", "stroke", "copd"]
        for condition in health_indicators.get("conditions", []):
            if any(risk in condition.lower() for risk in high_risk_conditions):
                risk_factors.append(f"High-risk condition: {condition}")
        
        # Check for polypharmacy
        if len(health_indicators.get("medications", [])) > 5:
            risk_factors.append("Polypharmacy (multiple medications)")
        
        # Check for severe symptoms
        severe_symptoms = ["chest pain", "difficulty breathing", "severe pain"]
        for symptom in health_indicators.get("symptoms", []):
            if any(severe in symptom.lower() for severe in severe_symptoms):
                risk_factors.append(f"Severe symptom: {symptom}")
        
        return risk_factors
    
    async def batch_analyze_texts(self, texts: List[str]) -> List[HealthAnalysisResult]:
        """Analyze multiple texts concurrently."""
        tasks = [self.process(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to analyze text {i}: {result}")
                # Create error result
                error_result = HealthAnalysisResult(
                    health_entities=[],
                    health_relations=[],
                    medical_conditions=[],
                    medications=[],
                    symptoms=[],
                    procedures=[],
                    overall_health_assessment=f"Analysis failed: {str(result)}",
                    risk_factors=[],
                    confidence_level=ConfidenceLevel.LOW,
                    processing_time=0.0
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results