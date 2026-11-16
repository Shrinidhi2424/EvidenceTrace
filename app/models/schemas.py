from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    claim: str = Field(min_length=10, max_length=300)
    max_papers: int = Field(default=8, ge=1, le=15)

    @field_validator("claim")
    @classmethod
    def claim_must_be_statement(cls, v: str) -> str:
        if v.endswith("?"):
            raise ValueError(
                "Phrase the claim as a statement, e.g. 'Sleep deprivation impairs working memory.'"
            )
        return v


class PaperResult(BaseModel):
    title: str
    year: Optional[int] = None
    source: str
    citation_count: int
    paper_id: str
    quality_score: float
    score_components: Dict[str, float]
    stance: str
    stance_confidence: float
    evidence_sentence: str


class Conflict(BaseModel):
    paper_a: str
    paper_b: str
    severity: str


class AnalyzeResponse(BaseModel):
    claim: str
    verdict: str
    confidence_score: float
    confidence_label: str
    evidence_distribution: Dict[str, float]
    evidence_counts: Dict[str, int]
    conflicts: List[Conflict]
    papers: List[PaperResult]
    timings_ms: Dict[str, float]
