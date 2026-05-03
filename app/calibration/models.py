from dataclasses import dataclass
from typing import Optional


@dataclass
class EvidenceItem:
    """
    Standardized evidence input for calibration layer.
    """

    stance: str
    confidence: float
    paper_id: str
    citation_count: int
    year: Optional[int]
    score: float
