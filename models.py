
from pydantic import BaseModel
from typing import List, Optional

class CyberCrimeSection(BaseModel):
    section: str
    act: str
    description: str
    punishment: Optional[str] = None
    crime_type: str
    keywords: List[str]

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class SearchResponse(BaseModel):
    sections: List[CyberCrimeSection]
    similarity_scores: List[float]
