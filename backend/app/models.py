# models.py - FIXED with diagram support
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentMetadata(BaseModel):
    """Enhanced document metadata for better multi-doc support"""
    category: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = []
    upload_date: datetime
    file_size: int
    page_count: int

class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: int = 10
    search_mode: str = "all"  # "all", "selected"

class SourceInfo(BaseModel):
    """Enhanced source information with clear document attribution"""
    document_id: str
    document_name: str
    page: int
    type: str
    relevance_score: float
    visual_types: Optional[List[str]] = None
    excerpt: str
    
class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    confidence: float
    documents_used: List[str]
    image_url: Optional[str] = None  # ← ADDED for diagram support
    diagram_generated: bool = False   # ← ADDED to indicate if diagram was created
    
class DocumentResponse(BaseModel):
    document_id: str
    document_name: str
    status: str
    chunks_created: int
    metadata: Optional[DocumentMetadata] = None

class DocumentListResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total_count: int