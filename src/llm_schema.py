"""Pydantic models for LLM JSON schema validation."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal, Union
import logging

logger = logging.getLogger(__name__)

class KidneySide(BaseModel):
    """Model for kidney-specific information (right or left)."""
    stone_status: Literal["present", "absent", "unclear"] = Field(
        description="Stone presence status for this kidney side"
    )
    stone_size_cm: Optional[float] = Field(
        None, 
        description="Stone size in centimeters, null if no stone or size not mentioned"
    )
    kidney_size_cm: Optional[str] = Field(
        None,
        description="Kidney dimensions in format 'L x W x AP cm' or 'L x W cm', null if not mentioned"
    )
    hydronephrosis_status: Literal["present", "absent", "unclear"] = Field(
        default="unclear",
        description="Hydronephrosis presence status for this kidney side (present/absent/unclear)"
    )
    
    @validator('stone_size_cm')
    def validate_stone_size(cls, v):
        if v is not None and (v < 0 or v > 20):
            logger.warning(f"Unusual stone size detected: {v} cm")
        return v
    
    @validator('kidney_size_cm')
    def validate_kidney_size(cls, v):
        if v is not None:
            # Basic validation for kidney size format
            if not any(char in v for char in ['x', 'X', 'cm']):
                logger.warning(f"Unusual kidney size format: {v}")
        return v

class Bladder(BaseModel):
    """Model for bladder information."""
    volume_ml: Optional[float] = Field(
        None,
        description="Bladder volume in milliliters, null if not mentioned"
    )
    wall: Optional[Literal["normal", "abnormal"]] = Field(
        None,
        description="Bladder wall status, null if not mentioned"
    )
    
    @validator('volume_ml')
    def validate_volume(cls, v):
        if v is not None and (v < 0 or v > 1000):
            logger.warning(f"Unusual bladder volume detected: {v} ml")
        return v

class RadiologyExtraction(BaseModel):
    """Main model for radiology narrative extraction."""
    right: KidneySide = Field(description="Right kidney information")
    left: KidneySide = Field(description="Left kidney information")
    bladder: Bladder = Field(description="Bladder information")
    history_summary: Optional[str] = Field(
        None,
        description="Summary of relevant medical history mentioned in the narrative"
    )
    key_sentences: Optional[List[str]] = Field(
        None,
        description="Key sentences that support the extracted findings"
    )
    
    @validator('history_summary')
    def validate_history_summary(cls, v):
        if v is not None and len(v) > 500:
            logger.warning(f"History summary is very long: {len(v)} characters")
        return v
    
    @validator('key_sentences')
    def validate_key_sentences(cls, v):
        if v is not None:
            if len(v) > 10:
                logger.warning(f"Many key sentences extracted: {len(v)}")
            # Limit sentence length
            v = [s[:200] for s in v if len(s) > 200]
        return v

class ExtractionResult(BaseModel):
    """Wrapper for extraction results with metadata."""
    extraction: RadiologyExtraction
    confidence: Optional[float] = Field(
        None,
        description="Confidence score for the extraction (0-1)"
    )
    extraction_method: Literal["llm", "regex", "hybrid"] = Field(
        description="Method used for extraction"
    )
    processing_time_ms: Optional[int] = Field(
        None,
        description="Processing time in milliseconds"
    )
    model_used: Optional[str] = Field(
        None,
        description="Model name used for LLM extraction"
    )
    tokens_used: Optional[dict] = Field(
        None,
        description="Token usage statistics"
    )

class StructuredOutput(BaseModel):
    """Final structured output format for the application."""
    recordid: str = Field(description="Patient record ID")
    imaging_date: str = Field(description="Imaging date")
    right_stone: Literal["present", "absent", "unclear"] = Field(
        description="Right kidney stone status"
    )
    right_stone_size_cm: Optional[float] = Field(
        None,
        description="Right kidney stone size in cm"
    )
    right_kidney_size_cm: Optional[str] = Field(
        None,
        description="Right kidney dimensions"
    )
    right_hydronephrosis: Literal["present", "absent", "unclear"] = Field(
        default="unclear",
        description="Right kidney hydronephrosis status"
    )
    left_stone: Literal["present", "absent", "unclear"] = Field(
        description="Left kidney stone status"
    )
    left_stone_size_cm: Optional[float] = Field(
        None,
        description="Left kidney stone size in cm"
    )
    left_kidney_size_cm: Optional[str] = Field(
        None,
        description="Left kidney dimensions"
    )
    left_hydronephrosis: Literal["present", "absent", "unclear"] = Field(
        default="unclear",
        description="Left kidney hydronephrosis status"
    )
    bladder_volume_ml: Optional[float] = Field(
        None,
        description="Bladder volume in ml"
    )
    history_summary: Optional[str] = Field(
        None,
        description="Medical history summary"
    )
    matched_reason: str = Field(
        description="Brief explanation of why this row matched the user's filters"
    )

class UserQuery(BaseModel):
    """Model for parsed user queries."""
    goal: str = Field(description="Main goal of the query")
    input_fields: List[str] = Field(
        default_factory=list,
        description="Input fields detected in the query"
    )
    filters: dict = Field(
        default_factory=dict,
        description="Extracted filters (side, size, date range, etc.)"
    )
    outputs: List[str] = Field(
        default_factory=list,
        description="Desired output columns"
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions made during parsing"
    )

class PlanSummary(BaseModel):
    """Model for user confirmation plan."""
    goal: str = Field(description="Main goal")
    input_fields: List[str] = Field(description="Detected input fields")
    filters: dict = Field(description="Applied filters")
    outputs: List[str] = Field(description="Output columns")
    assumptions: List[str] = Field(description="Assumptions")
    estimated_rows: Optional[int] = Field(
        None,
        description="Estimated number of matching rows"
    )
    processing_time_estimate: Optional[str] = Field(
        None,
        description="Estimated processing time"
    )

def validate_extraction_json(json_data: dict) -> RadiologyExtraction:
    """
    Validate and parse JSON data into RadiologyExtraction model.
    
    Args:
        json_data: Raw JSON data from LLM
        
    Returns:
        Validated RadiologyExtraction object
        
    Raises:
        ValueError: If validation fails
    """
    try:
        return RadiologyExtraction(**json_data)
    except Exception as e:
        logger.error(f"JSON validation failed: {e}")
        raise ValueError(f"Invalid extraction JSON: {e}")

def create_empty_extraction() -> RadiologyExtraction:
    """
    Create an empty extraction result for fallback cases.
    
    Returns:
        Empty RadiologyExtraction object
    """
    return RadiologyExtraction(
        right=KidneySide(stone_status="unclear", hydronephrosis_status="unclear"),
        left=KidneySide(stone_status="unclear", hydronephrosis_status="unclear"),
        bladder=Bladder(),
        history_summary=None,
        key_sentences=None
    )

def extraction_to_structured(extraction: RadiologyExtraction, 
                           recordid: str, 
                           imaging_date: str,
                           matched_reason: str = "No specific filters applied") -> StructuredOutput:
    """
    Convert RadiologyExtraction to StructuredOutput format.
    
    Args:
        extraction: RadiologyExtraction object
        recordid: Patient record ID
        imaging_date: Imaging date
        matched_reason: Reason for matching filters
        
    Returns:
        StructuredOutput object
    """
    return StructuredOutput(
        recordid=recordid,
        imaging_date=str(imaging_date),
        right_stone=extraction.right.stone_status,
        right_stone_size_cm=extraction.right.stone_size_cm,
        right_kidney_size_cm=extraction.right.kidney_size_cm,
        right_hydronephrosis=extraction.right.hydronephrosis_status,
        left_stone=extraction.left.stone_status,
        left_stone_size_cm=extraction.left.stone_size_cm,
        left_kidney_size_cm=extraction.left.kidney_size_cm,
        left_hydronephrosis=extraction.left.hydronephrosis_status,
        bladder_volume_ml=extraction.bladder.volume_ml,
        history_summary=extraction.history_summary,
        matched_reason=matched_reason
    )



