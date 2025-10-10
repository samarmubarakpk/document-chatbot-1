from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: int = 10

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float

class DocumentResponse(BaseModel):
    document_id: str
    document_name: str
    status: str
    chunks_created: int