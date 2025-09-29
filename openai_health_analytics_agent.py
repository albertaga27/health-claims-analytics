"""
OpenAI-based Health Analytics Agent for Claims Health Analytics.
Uses OpenAI system prompts to extract medical entities and relationships instead of Azure Text Analytics for Health.
"""
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncAzureOpenAI

from base_agent import BaseAgent
from models import (
    HealthEntity, HealthRelation, HealthAnalysisResult, HealthEntityType, ConfidenceLevel
)
from utils import (
    timing_decorator, calculate_confidence_level, HealthAnalyticsException
)
from config import AzureConfig, AgentConfig

class OpenAIHealthAnalyticsAgent(BaseAgent):
    """Agent responsible for health-specific text analytics using OpenAI system prompts."""
    
    def __init__(self, azure_config: AzureConfig, agent_config: AgentConfig):
        """Initialize the OpenAI Health Analytics Agent."""
        super().__init__(azure_config, agent_config)
        self.openai_client = self._setup_openai_client()
        self.system_prompt = self._create_health_analysis_system_prompt()
    
    def _setup_openai_client(self) -> Optional[AsyncAzureOpenAI]:
        """Set up the OpenAI client."""
        try:
            return AsyncAzureOpenAI(
                api_version=self.agent_config.openai_api_version,
                azure_endpoint=self.azure_config.openai_endpoint,
                api_key=self.azure_config.openai_api_key,
            )
        except Exception as e:
            self.logger.error(f"OpenAI client setup failed: {e}")
            raise HealthAnalyticsException(f"OpenAI client setup failed: {e}")
    
    def _create_health_analysis_system_prompt(self) -> str:
        """Create the system prompt for health text analysis."""
        return """You are a medical text analysis expert specializing in extracting structured health information from clinical documents, medical reports, insurance claims, and healthcare texts.

Your task is to analyze the provided text and extract health-related entities and relationships in a structured JSON format that matches Azure Text Analytics for Health output.

ENTITY TYPES TO IDENTIFY:
1. DIAGNOSIS - Medical conditions, diseases, disorders
2. SYMPTOM_OR_SIGN - Symptoms, signs, clinical observations
3. MEDICATION_NAME - Specific medication names
4. MEDICATION_CLASS - Drug classes (e.g., antibiotics, antihypertensives)
5. DOSAGE - Medication dosages and strengths
6. BODY_STRUCTURE - Anatomical parts, organs, body systems
7. CONDITION_QUALIFIER - Qualifiers that describe conditions (e.g., chronic, acute, severe)
8. TIME - Temporal expressions related to health events
9. FREQUENCY - How often medications are taken or symptoms occur
10. MEDICATION_ROUTE - Route of administration (oral, IV, etc.)

RELATIONSHIPS TO IDENTIFY:
- DOSAGE_OF_MEDICATION: Links dosage to specific medications
- ROUTE_OR_MODE_OF_MEDICATION: Links administration route to medications
- FREQUENCY_OF_MEDICATION: Links frequency to medications
- QUALIFIER_OF_CONDITION: Links qualifiers to medical conditions
- TIME_OF_CONDITION: Links temporal information to conditions
- TIME_OF_MEDICATION: Links temporal information to medications

ANALYSIS REQUIREMENTS:
1. Extract all relevant health entities with their exact text, position, and confidence
2. Identify relationships between entities
3. Determine if entities are negated (e.g., "no history of diabetes")
4. Categorize entities by medical significance
5. Assess overall health complexity and risk factors
6. Provide confidence scores (0.0-1.0) for each entity and relationship

OUTPUT FORMAT:
Return a JSON object with the following structure:
{
    "entities": [
        {
            "text": "exact text from document",
            "category": "DIAGNOSIS|SYMPTOM_OR_SIGN|MEDICATION_NAME|etc",
            "confidence_score": 0.95,
            "offset": 123,
            "length": 8,
            "is_negated": false,
            "subcategory": "optional subcategory"
        }
    ],
    "relations": [
        {
            "relation_type": "DOSAGE_OF_MEDICATION",
            "source_entity_index": 0,
            "target_entity_index": 1,
            "confidence_score": 0.90
        }
    ],
    "medical_conditions": ["condition1", "condition2"],
    "medications": ["medication1", "medication2"],
    "symptoms": ["symptom1", "symptom2"],
    "procedures": ["procedure1", "procedure2"],
    "risk_factors": ["risk1", "risk2"],
    "overall_assessment": "Brief assessment of health complexity and key findings",
    "confidence_level": "HIGH|MEDIUM|LOW"
}

GUIDELINES:
- Be precise and conservative with entity extraction
- Only extract medically relevant information
- Use appropriate medical terminology
- Consider context to avoid false positives
- Assign confidence scores based on clarity and certainty
- Mark entities as negated when appropriate (e.g., "denies chest pain")
- Focus on clinically significant information
"""

    @timing_decorator
    async def process(self, input_data: str) -> HealthAnalysisResult:
        """Process text and extract health-specific information using OpenAI."""
        try:
            if not input_data or not input_data.strip():
                raise HealthAnalyticsException("Input text is empty or invalid")
            
            # Limit text size to avoid token limits
            text = input_data[:8000] if len(input_data) > 8000 else input_data
            
            # Analyze with OpenAI
            analysis_result = await self._analyze_with_openai(text)
            
            # Convert to our standard format
            return self._convert_openai_result_to_health_analysis(analysis_result, text)
                
        except Exception as e:
            self.logger.error(f"OpenAI health analytics processing failed: {e}")
            raise HealthAnalyticsException(f"Health analytics processing failed: {e}")
    
    async def _analyze_with_openai(self, text: str) -> Dict[str, Any]:
        """Analyze text using OpenAI with specialized health prompts."""
        try:
            user_prompt = f"""Analyze the following medical/health text and extract structured health information:

TEXT TO ANALYZE:
{text}

Please provide a comprehensive analysis following the JSON format specified in the system prompt. Focus on accuracy and medical relevance."""

            response = await self.openai_client.chat.completions.create(
                model=self.azure_config.openai_deployment_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.agent_config.temperature,
                max_tokens=self.agent_config.max_tokens,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content
            return json.loads(result_text)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse OpenAI JSON response: {e}")
            # Fallback to basic extraction
            return await self._fallback_analysis(text)
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            return await self._fallback_analysis(text)
    
    async def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback analysis using simpler pattern matching."""
        from utils import extract_key_health_indicators
        
        indicators = extract_key_health_indicators(text)
        
        return {
            "entities": [],
            "relations": [],
            "medical_conditions": indicators.get("conditions", []),
            "medications": indicators.get("medications", []),
            "symptoms": indicators.get("symptoms", []),
            "procedures": indicators.get("procedures", []),
            "risk_factors": self._identify_risk_factors_from_indicators(indicators),
            "overall_assessment": f"Fallback analysis identified {sum(len(v) for v in indicators.values())} health indicators",
            "confidence_level": "LOW"
        }
    
    def _convert_openai_result_to_health_analysis(self, openai_result: Dict[str, Any], original_text: str) -> HealthAnalysisResult:
        """Convert OpenAI analysis result to HealthAnalysisResult format."""
        # Convert entities
        health_entities = []
        for entity_data in openai_result.get("entities", []):
            try:
                entity = HealthEntity(
                    text=entity_data.get("text", ""),
                    category=self._map_category_string_to_enum(entity_data.get("category", "DIAGNOSIS")),
                    confidence_score=entity_data.get("confidence_score", 0.7),
                    offset=entity_data.get("offset", 0),
                    length=entity_data.get("length", len(entity_data.get("text", ""))),
                    is_negated=entity_data.get("is_negated", False),
                    subcategory=entity_data.get("subcategory"),
                    assertion={"certainty": "negative" if entity_data.get("is_negated", False) else "positive"},
                    data_sources=[{"name": "OpenAI", "entity_id": f"openai_{len(health_entities)}"}]
                )
                health_entities.append(entity)
            except Exception as e:
                self.logger.warning(f"Failed to convert entity: {e}")
                continue
        
        # Convert relations
        health_relations = []
        for relation_data in openai_result.get("relations", []):
            try:
                source_idx = relation_data.get("source_entity_index", 0)
                target_idx = relation_data.get("target_entity_index", 0)
                
                if (source_idx < len(health_entities) and target_idx < len(health_entities)):
                    relation = HealthRelation(
                        relation_type=relation_data.get("relation_type", "RELATED_TO"),
                        source_entity=health_entities[source_idx],
                        target_entity=health_entities[target_idx],
                        confidence_score=relation_data.get("confidence_score", 0.7)
                    )
                    health_relations.append(relation)
            except Exception as e:
                self.logger.warning(f"Failed to convert relation: {e}")
                continue
        
        # Extract other data with defaults
        medical_conditions = openai_result.get("medical_conditions", [])
        medications = openai_result.get("medications", [])
        symptoms = openai_result.get("symptoms", [])
        procedures = openai_result.get("procedures", [])
        risk_factors = openai_result.get("risk_factors", [])
        overall_assessment = openai_result.get("overall_assessment", "Analysis completed using OpenAI")
        
        # Map confidence level
        confidence_str = openai_result.get("confidence_level", "MEDIUM")
        confidence_level = self._map_confidence_string_to_enum(confidence_str)
        
        return HealthAnalysisResult(
            health_entities=health_entities,
            health_relations=health_relations,
            medical_conditions=medical_conditions,
            medications=medications,
            symptoms=symptoms,
            procedures=procedures,
            overall_health_assessment=overall_assessment,
            risk_factors=risk_factors,
            confidence_level=confidence_level,
            processing_time=0.0  # Set by timing decorator
        )
    
    def _map_category_string_to_enum(self, category_str: str) -> HealthEntityType:
        """Map category string to HealthEntityType enum."""
        mapping = {
            "DIAGNOSIS": HealthEntityType.DIAGNOSIS,
            "SYMPTOM_OR_SIGN": HealthEntityType.SYMPTOM_OR_SIGN,
            "MEDICATION_NAME": HealthEntityType.MEDICATION_NAME,
            "MEDICATION_CLASS": HealthEntityType.MEDICATION_CLASS,
            "DOSAGE": HealthEntityType.DOSAGE,
            "BODY_STRUCTURE": HealthEntityType.BODY_STRUCTURE,
            "CONDITION_QUALIFIER": HealthEntityType.CONDITION_QUALIFIER,
            "TIME": HealthEntityType.TIME,
            "FREQUENCY": HealthEntityType.FREQUENCY,
            "MEDICATION_ROUTE": HealthEntityType.MEDICATION_ROUTE,
            "MEDICATION_FORM": HealthEntityType.MEDICATION_FORM,
            "RELATION_TYPE": HealthEntityType.RELATION_TYPE,
            "GENE_OR_PROTEIN": HealthEntityType.GENE_OR_PROTEIN,
            "VARIANT": HealthEntityType.VARIANT
        }
        return mapping.get(category_str.upper(), HealthEntityType.DIAGNOSIS)
    
    def _map_confidence_string_to_enum(self, confidence_str: str) -> ConfidenceLevel:
        """Map confidence string to ConfidenceLevel enum."""
        mapping = {
            "HIGH": ConfidenceLevel.HIGH,
            "MEDIUM": ConfidenceLevel.MEDIUM,
            "LOW": ConfidenceLevel.LOW
        }
        return mapping.get(confidence_str.upper(), ConfidenceLevel.MEDIUM)
    
    def _identify_risk_factors_from_indicators(self, indicators: Dict[str, List[str]]) -> List[str]:
        """Identify risk factors from basic health indicators."""
        risk_factors = []
        
        # High-risk conditions
        high_risk_conditions = ["cancer", "heart disease", "diabetes", "stroke", "copd", "kidney disease"]
        for condition in indicators.get("conditions", []):
            if any(risk in condition.lower() for risk in high_risk_conditions):
                risk_factors.append(f"High-risk condition: {condition}")
        
        # Polypharmacy
        if len(indicators.get("medications", [])) > 5:
            risk_factors.append("Polypharmacy (multiple medications)")
        
        # Severe symptoms
        severe_symptoms = ["chest pain", "difficulty breathing", "severe pain", "loss of consciousness"]
        for symptom in indicators.get("symptoms", []):
            if any(severe in symptom.lower() for severe in severe_symptoms):
                risk_factors.append(f"Severe symptom: {symptom}")
        
        return risk_factors
    
    async def batch_analyze_texts(self, texts: List[str]) -> List[HealthAnalysisResult]:
        """Analyze multiple texts concurrently."""
        # Limit concurrency to avoid rate limits
        semaphore = asyncio.Semaphore(3)
        
        async def analyze_with_semaphore(text: str) -> HealthAnalysisResult:
            async with semaphore:
                return await self.process(text)
        
        tasks = [analyze_with_semaphore(text) for text in texts]
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
    
    async def enhanced_analysis_with_followup(self, text: str) -> HealthAnalysisResult:
        """Perform enhanced analysis with follow-up questions for more detailed extraction."""
        # First pass - standard analysis
        initial_result = await self.process(text)
        
        # If the initial analysis found significant health information, do a follow-up analysis
        total_entities = len(initial_result.health_entities)
        
        if total_entities > 0:
            try:
                # Follow-up analysis for more details
                followup_prompt = f"""Based on the initial analysis of this medical text, please provide additional insights:

Original text: {text[:2000]}

Initial findings:
- Conditions: {', '.join(initial_result.medical_conditions)}
- Medications: {', '.join(initial_result.medications)}
- Symptoms: {', '.join(initial_result.symptoms)}

Please identify:
1. Any missed medical entities or relationships
2. Additional risk factors or complications
3. Severity indicators or urgency markers
4. Drug interactions or contraindications (if applicable)
5. Relevant family history or genetic factors

Return only the additional findings in the same JSON format."""

                response = await self.openai_client.chat.completions.create(
                    model=self.azure_config.openai_deployment_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": followup_prompt}
                    ],
                    temperature=self.agent_config.temperature,
                    max_tokens=1500,
                    response_format={"type": "json_object"}
                )
                
                followup_data = json.loads(response.choices[0].message.content)
                
                # Merge follow-up findings with initial results
                return self._merge_analysis_results(initial_result, followup_data, text)
                
            except Exception as e:
                self.logger.warning(f"Follow-up analysis failed: {e}")
                return initial_result
        
        return initial_result
    
    def _merge_analysis_results(self, initial: HealthAnalysisResult, followup: Dict[str, Any], text: str) -> HealthAnalysisResult:
        """Merge initial and follow-up analysis results."""
        # Merge entities (avoid duplicates)
        additional_entities = []
        existing_texts = {entity.text.lower() for entity in initial.health_entities}
        
        for entity_data in followup.get("entities", []):
            if entity_data.get("text", "").lower() not in existing_texts:
                try:
                    entity = HealthEntity(
                        text=entity_data.get("text", ""),
                        category=self._map_category_string_to_enum(entity_data.get("category", "DIAGNOSIS")),
                        confidence_score=entity_data.get("confidence_score", 0.6),
                        offset=entity_data.get("offset", 0),
                        length=entity_data.get("length", len(entity_data.get("text", ""))),
                        is_negated=entity_data.get("is_negated", False),
                        subcategory=entity_data.get("subcategory"),
                        assertion={"certainty": "negative" if entity_data.get("is_negated", False) else "positive"},
                        data_sources=[{"name": "OpenAI-Followup", "entity_id": f"followup_{len(additional_entities)}"}]
                    )
                    additional_entities.append(entity)
                except Exception as e:
                    self.logger.warning(f"Failed to merge entity: {e}")
        
        # Merge other fields
        merged_conditions = list(set(initial.medical_conditions + followup.get("medical_conditions", [])))
        merged_medications = list(set(initial.medications + followup.get("medications", [])))
        merged_symptoms = list(set(initial.symptoms + followup.get("symptoms", [])))
        merged_procedures = list(set(initial.procedures + followup.get("procedures", [])))
        merged_risk_factors = list(set(initial.risk_factors + followup.get("risk_factors", [])))
        
        # Enhanced assessment
        enhanced_assessment = f"{initial.overall_health_assessment} Enhanced analysis identified additional relevant findings."
        
        return HealthAnalysisResult(
            health_entities=initial.health_entities + additional_entities,
            health_relations=initial.health_relations,  # Keep original relations
            medical_conditions=merged_conditions,
            medications=merged_medications,
            symptoms=merged_symptoms,
            procedures=merged_procedures,
            overall_health_assessment=enhanced_assessment,
            risk_factors=merged_risk_factors,
            confidence_level=initial.confidence_level,
            processing_time=initial.processing_time
        )