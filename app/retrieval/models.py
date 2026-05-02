from dataclasses import dataclass
from typing import Optional


@dataclass
class Paper:
    """
    Standardized paper object across all sources.
    """

    title: str
    abstract: str
    year: Optional[int]
    citation_count: int
    paper_id: str
    source: str

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "citation_count": self.citation_count,
            "paper_id": self.paper_id,
            "source": self.source,
        }
