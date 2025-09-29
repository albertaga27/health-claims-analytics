"""
Underwriting Reasoning Agent for Claims Health Analytics.
Uses GPT models for intelligent reasoning and underwriting guidance.
"""
import asyncio
from typing import Any, Dict, List, Optional, Union
import json
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from base_agent import BaseAgent
from models import (
    ClaimSubmission, HealthAnalysisResult, DocumentExtractionResult,
    UnderwriterRecommendation, ConfidenceLevel
)
from utils import (
    timing_decorator, format_currency, truncate_text, UnderwritingException
)
from config import AzureConfig, AgentConfig

class UnderwritingReasoningAgent(BaseAgent):
    """Agent responsible for analyzing claims and providing underwriting recommendations."""
    
    def __init__(self, azure_config: AzureConfig, agent_config: AgentConfig):
        """Initialize the Underwriting Reasoning Agent."""
        super().__init__(azure_config, agent_config)
        self.chat_client = self._setup_chat_client()
    
    def _setup_chat_client(self) -> AzureOpenAI:
        """Set up the chat client for GPT processing."""
        try:
            token_provider = get_bearer_token_provider(
                self.credential, 
                'https://cognitiveservices.azure.com/.default'
            )
            return AzureOpenAI(
                azure_endpoint=self.azure_config.openai_endpoint,
                azure_ad_token_provider=token_provider,
                api_version=self.agent_config.openai_api_version
            )
        except Exception as e:
            self.logger.error(f"Failed to setup chat client: {e}")
            raise UnderwritingException(f"Chat client setup failed: {e}")
    
    @timing_decorator
    async def process(self, input_data: Dict[str, Any]) -> UnderwriterRecommendation:
        """Process claim data and generate underwriting recommendation."""
        try:
            # Validate input
            required_fields = ["claim", "health_analysis"]
            self.validate_input(input_data, required_fields)
            
            claim: ClaimSubmission = input_data["claim"]
            health_analysis: HealthAnalysisResult = input_data["health_analysis"]
            documents: List[DocumentExtractionResult] = input_data.get("documents", [])
            
            # Generate comprehensive analysis
            return await self._generate_underwriting_recommendation(claim, health_analysis, documents)
            
        except Exception as e:
            self.logger.error(f"Underwriting reasoning failed: {e}")
            raise UnderwritingException(f"Underwriting reasoning failed: {e}")
    
    async def _generate_underwriting_recommendation(
        self, 
        claim: ClaimSubmission, 
        health_analysis: HealthAnalysisResult, 
        documents: List[DocumentExtractionResult]
    ) -> UnderwriterRecommendation:
        """Generate detailed underwriting recommendation using GPT reasoning."""
        
        # Prepare context for GPT
        context = await self._prepare_underwriting_context(claim, health_analysis, documents)
        
        # Create system prompt for underwriting
        system_prompt = self._create_underwriting_system_prompt()
        
        # Create user prompt with claim context
        user_prompt = self._create_underwriting_user_prompt(context)
        
        try:
            # Get GPT recommendation
            response = await asyncio.to_thread(
                self.chat_client.chat.completions.create,
                model=self.agent_config.reasoning_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # Note: o4-mini only supports temperature=1 (default)
                max_completion_tokens=self.agent_config.max_tokens
            )
            
            recommendation_text = response.choices[0].message.content
            
            # Parse GPT response into structured recommendation
            return await self._parse_recommendation_response(recommendation_text, context)
            
        except Exception as e:
            raise UnderwritingException(f"GPT reasoning failed: {e}")
    
    async def _prepare_underwriting_context(
        self, 
        claim: ClaimSubmission, 
        health_analysis: HealthAnalysisResult, 
        documents: List[DocumentExtractionResult]
    ) -> Dict[str, Any]:
        """Prepare comprehensive context for underwriting analysis."""
        
        # Extract key information
        context = {
            "claim_id": claim.claim_id,
            "claimant_info": claim.claimant_info,
            "submission_date": claim.submitted_at.isoformat(),
            "document_count": len(documents),
            "health_summary": {
                "conditions": health_analysis.medical_conditions,
                "medications": health_analysis.medications,
                "symptoms": health_analysis.symptoms,
                "procedures": health_analysis.procedures,
                "risk_factors": health_analysis.risk_factors,
                "overall_assessment": health_analysis.overall_health_assessment,
                "confidence_level": health_analysis.confidence_level.value
            },
            "document_summaries": [],
            "extracted_text_summary": ""
        }
        
        # Summarize documents
        for doc in documents:
            doc_summary = {
                "file_name": doc.document_metadata.file_name,
                "file_type": doc.document_metadata.file_type,
                "confidence": doc.confidence_score,
                "key_entities": [entity.text for entity in doc.extracted_entities[:5]],  # Top 5 entities
                "text_preview": truncate_text(doc.extracted_text, 200)
            }
            context["document_summaries"].append(doc_summary)
        
        # Create combined text summary
        all_text = " ".join([doc.extracted_text for doc in documents])
        context["extracted_text_summary"] = truncate_text(all_text, 1000)
        
        # Calculate risk indicators
        context["risk_indicators"] = await self._calculate_risk_indicators(health_analysis, documents)
        
        return context
    
    def _create_underwriting_system_prompt(self) -> str:
        """Create system prompt for underwriting analysis."""
        return """You are an expert medical underwriter with 20+ years of experience in health insurance claims analysis. Your role is to analyze medical claims and provide professional underwriting recommendations.

Key responsibilities:
1. Assess medical risk based on documented conditions, treatments, and procedures
2. Identify potential red flags or inconsistencies
3. Evaluate completeness and quality of submitted documentation
4. Provide clear recommendations: APPROVE, REJECT, or REQUEST_MORE_INFO
5. Justify decisions with specific medical and underwriting rationale

Guidelines:
- Be thorough but fair in your assessment
- Consider industry standards and best practices
- Flag high-risk conditions or treatments that require special attention
- Identify missing documentation that could impact the decision
- Provide specific, actionable recommendations
- Maintain professional tone and objectivity

Output your analysis in a structured format with:
- Primary recommendation (APPROVE/REJECT/REQUEST_MORE_INFO)
- Confidence level (HIGH/MEDIUM/LOW)
- Detailed reasoning
- Risk assessment
- Required actions (if any)
- Red flags identified
- Supporting evidence
- Estimated claim amount (if applicable)
- Approval conditions (if applicable)"""
    
    def _create_underwriting_user_prompt(self, context: Dict[str, Any]) -> str:
        """Create user prompt with claim context."""
        prompt = f"""Please analyze the following health insurance claim and provide your underwriting recommendation:

CLAIM INFORMATION:
- Claim ID: {context['claim_id']}
- Submission Date: {context['submission_date']}
- Number of Documents: {context['document_count']}

CLAIMANT INFORMATION:
{json.dumps(context['claimant_info'], indent=2)}

HEALTH ANALYSIS SUMMARY:
- Medical Conditions: {', '.join(context['health_summary']['conditions']) if context['health_summary']['conditions'] else 'None documented'}
- Medications: {', '.join(context['health_summary']['medications']) if context['health_summary']['medications'] else 'None documented'}
- Symptoms: {', '.join(context['health_summary']['symptoms']) if context['health_summary']['symptoms'] else 'None documented'}
- Procedures: {', '.join(context['health_summary']['procedures']) if context['health_summary']['procedures'] else 'None documented'}
- Risk Factors: {', '.join(context['health_summary']['risk_factors']) if context['health_summary']['risk_factors'] else 'None identified'}
- Overall Assessment: {context['health_summary']['overall_assessment']}
- Analysis Confidence: {context['health_summary']['confidence_level']}

DOCUMENT SUMMARIES:
{json.dumps(context['document_summaries'], indent=2)}

RISK INDICATORS:
{json.dumps(context['risk_indicators'], indent=2)}

EXTRACTED TEXT SUMMARY:
{context['extracted_text_summary']}

Based on this information, provide your professional underwriting recommendation with detailed justification."""
    
        return prompt
    
    async def _calculate_risk_indicators(
        self, 
        health_analysis: HealthAnalysisResult, 
        documents: List[DocumentExtractionResult]
    ) -> Dict[str, Any]:
        """Calculate risk indicators for underwriting."""
        
        indicators = {
            "high_risk_conditions": [],
            "medication_complexity": "low",
            "documentation_quality": "good",
            "consistency_issues": [],
            "severity_indicators": [],
            "cost_indicators": []
        }
        
        # Analyze high-risk conditions
        high_risk_keywords = ["cancer", "heart disease", "diabetes", "stroke", "chronic kidney", "liver disease"]
        for condition in health_analysis.medical_conditions:
            if any(keyword in condition.lower() for keyword in high_risk_keywords):
                indicators["high_risk_conditions"].append(condition)
        
        # Assess medication complexity
        med_count = len(health_analysis.medications)
        if med_count > 10:
            indicators["medication_complexity"] = "high"
        elif med_count > 5:
            indicators["medication_complexity"] = "moderate"
        else:
            indicators["medication_complexity"] = "low"
        
        # Evaluate documentation quality
        avg_confidence = sum(doc.confidence_score for doc in documents) / len(documents) if documents else 0
        if avg_confidence < 0.6:
            indicators["documentation_quality"] = "poor"
        elif avg_confidence < 0.8:
            indicators["documentation_quality"] = "fair"
        else:
            indicators["documentation_quality"] = "good"
        
        # Check for severity indicators
        severe_symptoms = ["severe pain", "emergency", "critical", "life-threatening"]
        for symptom in health_analysis.symptoms:
            if any(severe in symptom.lower() for severe in severe_symptoms):
                indicators["severity_indicators"].append(symptom)
        
        # Identify potential cost indicators
        expensive_procedures = ["surgery", "transplant", "radiation", "chemotherapy", "dialysis"]
        for procedure in health_analysis.procedures:
            if any(expensive in procedure.lower() for expensive in expensive_procedures):
                indicators["cost_indicators"].append(procedure)
        
        return indicators
    
    async def _parse_recommendation_response(
        self, 
        gpt_response: str, 
        context: Dict[str, Any]
    ) -> UnderwriterRecommendation:
        """Parse GPT response into structured underwriting recommendation."""
        
        # Extract recommendation (simple parsing - could be more sophisticated)
        response_lower = gpt_response.lower()
        
        if "approve" in response_lower and "reject" not in response_lower:
            recommendation = "APPROVE"
        elif "reject" in response_lower:
            recommendation = "REJECT"
        else:
            recommendation = "REQUEST_MORE_INFO"
        
        # Extract confidence level
        if "high confidence" in response_lower or "highly confident" in response_lower:
            confidence_score = 0.9
        elif "medium confidence" in response_lower or "moderate confidence" in response_lower:
            confidence_score = 0.7
        elif "low confidence" in response_lower:
            confidence_score = 0.5
        else:
            confidence_score = 0.6
        
        # Extract key information (simplified extraction)
        red_flags = await self._extract_red_flags(gpt_response, context)
        required_actions = await self._extract_required_actions(gpt_response)
        supporting_evidence = await self._extract_supporting_evidence(gpt_response)
        approval_conditions = await self._extract_approval_conditions(gpt_response, recommendation)
        
        # Generate risk assessment
        risk_assessment = await self._generate_risk_assessment(context, gpt_response)
        
        # Estimate claim amount (basic estimation)
        estimated_amount = await self._estimate_claim_amount(context)
        
        return UnderwriterRecommendation(
            recommendation=recommendation,
            confidence_score=confidence_score,
            reasoning=gpt_response,
            risk_assessment=risk_assessment,
            required_actions=required_actions,
            red_flags=red_flags,
            supporting_evidence=supporting_evidence,
            estimated_claim_amount=estimated_amount,
            approval_conditions=approval_conditions
        )
    
    async def _extract_red_flags(self, response: str, context: Dict[str, Any]) -> List[str]:
        """Extract red flags from GPT response and context."""
        red_flags = []
        
        # Check for explicit red flags in response
        if "red flag" in response.lower():
            # Simple extraction - could be more sophisticated
            red_flags.append("Potential inconsistencies identified in documentation")
        
        # Check context for red flags
        risk_indicators = context.get("risk_indicators", {})
        if risk_indicators.get("high_risk_conditions"):
            red_flags.append(f"High-risk conditions present: {', '.join(risk_indicators['high_risk_conditions'])}")
        
        if risk_indicators.get("documentation_quality") == "poor":
            red_flags.append("Poor documentation quality detected")
        
        if risk_indicators.get("severity_indicators"):
            red_flags.append("Severe symptoms or critical conditions identified")
        
        return red_flags
    
    async def _extract_required_actions(self, response: str) -> List[str]:
        """Extract required actions from GPT response."""
        actions = []
        
        response_lower = response.lower()
        
        if "additional documentation" in response_lower or "more information" in response_lower:
            actions.append("Request additional medical documentation")
        
        if "medical examination" in response_lower or "independent medical exam" in response_lower:
            actions.append("Schedule independent medical examination")
        
        if "specialist review" in response_lower:
            actions.append("Obtain specialist medical review")
        
        if "clarification" in response_lower:
            actions.append("Request clarification on medical history")
        
        return actions
    
    async def _extract_supporting_evidence(self, response: str) -> List[str]:
        """Extract supporting evidence from GPT response."""
        evidence = []
        
        # Look for specific medical terms or evidence mentioned
        if "documented" in response.lower():
            evidence.append("Medical conditions properly documented")
        
        if "consistent" in response.lower():
            evidence.append("Medical history appears consistent")
        
        if "complete" in response.lower():
            evidence.append("Documentation appears complete")
        
        return evidence
    
    async def _extract_approval_conditions(self, response: str, recommendation: str) -> List[str]:
        """Extract approval conditions if applicable."""
        conditions = []
        
        if recommendation == "APPROVE":
            response_lower = response.lower()
            
            if "monitoring" in response_lower:
                conditions.append("Regular medical monitoring required")
            
            if "follow-up" in response_lower:
                conditions.append("Periodic follow-up examinations")
            
            if "treatment compliance" in response_lower:
                conditions.append("Compliance with prescribed treatment plan")
        
        return conditions
    
    async def _generate_risk_assessment(self, context: Dict[str, Any], gpt_response: str) -> str:
        """Generate risk assessment summary."""
        risk_indicators = context.get("risk_indicators", {})
        
        risk_level = "LOW"
        if risk_indicators.get("high_risk_conditions") or risk_indicators.get("severity_indicators"):
            risk_level = "HIGH"
        elif risk_indicators.get("medication_complexity") == "high":
            risk_level = "MEDIUM"
        
        assessment = f"Risk Level: {risk_level}. "
        
        if risk_indicators.get("high_risk_conditions"):
            assessment += f"High-risk conditions identified: {len(risk_indicators['high_risk_conditions'])}. "
        
        if risk_indicators.get("cost_indicators"):
            assessment += f"Potential high-cost procedures: {len(risk_indicators['cost_indicators'])}. "
        
        assessment += f"Documentation quality: {risk_indicators.get('documentation_quality', 'unknown')}."
        
        return assessment
    
    async def _estimate_claim_amount(self, context: Dict[str, Any]) -> Optional[float]:
        """Estimate potential claim amount based on context."""
        # Simple estimation logic - in practice, this would be more sophisticated
        base_amount = 5000.0  # Base claim amount
        
        risk_indicators = context.get("risk_indicators", {})
        
        # Adjust based on conditions
        high_risk_count = len(risk_indicators.get("high_risk_conditions", []))
        cost_procedures = len(risk_indicators.get("cost_indicators", []))
        
        multiplier = 1.0
        multiplier += high_risk_count * 0.5  # 50% increase per high-risk condition
        multiplier += cost_procedures * 1.0   # 100% increase per costly procedure
        
        estimated = base_amount * multiplier
        
        # Cap the estimate
        return min(estimated, 100000.0)
    
    async def process_multiple_claims(
        self, 
        claims_data: List[Dict[str, Any]]
    ) -> List[UnderwriterRecommendation]:
        """Process multiple claims for underwriting recommendations."""
        tasks = [self.process(claim_data) for claim_data in claims_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to process claim {i}: {result}")
                # Create error recommendation
                error_recommendation = UnderwriterRecommendation(
                    recommendation="REQUEST_MORE_INFO",
                    confidence_score=0.0,
                    reasoning=f"Processing failed: {str(result)}",
                    risk_assessment="Unable to assess due to processing error",
                    required_actions=["Review claim manually due to processing error"],
                    red_flags=[str(result)],
                    supporting_evidence=[],
                    estimated_claim_amount=None,
                    approval_conditions=[]
                )
                processed_results.append(error_recommendation)
            else:
                processed_results.append(result)
        
        return processed_results