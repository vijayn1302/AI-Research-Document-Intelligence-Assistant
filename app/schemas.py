from typing import List, Optional
from pydantic import BaseModel, Field

class Source(BaseModel):
    source: str
    page: Optional[int] = None
    score: Optional[float] = None
    snippet: str

class AskRequest(BaseModel):
    question: str = Field(min_length = 2)
    top_k: int = Field(default = 4, ge = 1, le = 10)

class AskResponse(BaseModel):
    answer: str
    grounded: bool
    grounding_score: float
    sources: List[Source]

class GroundingCheck(BaseModel):
    grounded: bool
    score: float = Field(ge = 0, le = 1)
    reason: str

class RelevanceCheck(BaseModel):
    relevant: bool
    score: float = Field(ge = 0, le = 1)
    reason: str
