# main.py - Enhanced Multi-Document Support
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from openai import AsyncOpenAI
from typing import List, Optional, Dict, Any
import tempfile
import os
from datetime import datetime
from collections import defaultdict

from app.config import settings
from app.document_processor import DocumentProcessor
from app.retrieval_engine import HybridRetriever
from app.models import (
    QueryRequest, QueryResponse, DocumentResponse, 
    SourceInfo, DocumentMetadata, DocumentListResponse
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Document Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_processor = DocumentProcessor()
retriever = HybridRetriever()

# Document registry (use database in production)
document_registry: Dict[str, Dict[str, Any]] = {}

@app.on_event("startup")
async def startup_event():
    await document_processor.initialize_collections()

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    category: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[str] = None
):
    """Upload and process a document with metadata"""
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"Processing file: {file.filename}")
        result = await document_processor.process_document(tmp_path, file.filename)
        
        # Store metadata
        metadata = DocumentMetadata(
            category=category,
            description=description,
            tags=tags.split(",") if tags else [],
            upload_date=datetime.now(),
            file_size=len(content),
            page_count=result.get("page_count", 0)
        )
        
        document_registry[result["document_id"]] = {
            "document_id": result["document_id"],
            "document_name": file.filename,
            "metadata": metadata.dict(),
            "chunks_created": result["chunks_created"],
            "upload_date": datetime.now().isoformat()
        }
        
        return DocumentResponse(**result, metadata=metadata)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    category: Optional[str] = None,
    search: Optional[str] = None
):
    """List all documents with filtering"""
    documents = list(document_registry.values())
    
    if category:
        documents = [d for d in documents if d.get("metadata", {}).get("category") == category]
    
    if search:
        search_lower = search.lower()
        documents = [
            d for d in documents 
            if search_lower in d["document_name"].lower() or
               search_lower in d.get("metadata", {}).get("description", "").lower()
        ]
    
    return DocumentListResponse(documents=documents, total_count=len(documents))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    if document_id not in document_registry:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        await retriever.delete_document(document_id)
        del document_registry[document_id]
        return {"status": "success", "message": "Document deleted"}
    except Exception as e:
        logger.error(f"Error deleting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query with multi-document support"""
    try:
        logger.info(f"\nQUERY: {request.query}")
        logger.info(f"MODE: {request.search_mode}")
        
        document_filter = None
        if request.search_mode == "selected" and request.document_ids:
            document_filter = request.document_ids
            logger.info(f"Searching {len(document_filter)} documents")
        
        results = await retriever.retrieve(
            request.query,
            request.top_k,
            document_filter
        )
        
        logger.info(f"Retrieved {len(results)} results")
        response = await generate_multi_document_answer(request.query, results)
        
        return QueryResponse(**response)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def generate_multi_document_answer(query: str, sources: List[Dict]) -> Dict:
    """Generate answer with CLEAR document attribution"""
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Group by document
    doc_groups = defaultdict(lambda: {'text': [], 'hardware': [], 'visuals': [], 'doc_name': ''})
    
    for source in sources[:15]:
        doc_id = source['metadata'].get('document_id', 'unknown')
        doc_name = source['metadata'].get('document_name', 'Unknown')
        source_type = source['metadata'].get('type', 'text')
        page = source['metadata'].get('page', 0)
        
        doc_groups[doc_id]['doc_name'] = doc_name
        source_with_page = {**source, 'page': page}
        
        if source_type == 'hardware_spec':
            doc_groups[doc_id]['hardware'].append(source_with_page)
        elif source_type == 'visual':
            doc_groups[doc_id]['visuals'].append(source_with_page)
        else:
            doc_groups[doc_id]['text'].append(source_with_page)
    
    # Build context with document separation
    context = f"=== INFORMATION FROM {len(doc_groups)} DOCUMENT(S) ===\n\n"
    context += "⚠️ IMPORTANT: Always specify which document information comes from.\n\n"
    
    documents_used = []
    
    for doc_idx, (doc_id, content) in enumerate(doc_groups.items(), 1):
        doc_name = content['doc_name']
        documents_used.append(doc_name)
        
        context += f"\n{'='*80}\n📄 DOCUMENT {doc_idx}: {doc_name}\n{'='*80}\n\n"
        
        if content['hardware']:
            context += "🔧 HARDWARE SPECIFICATIONS:\n"
            for hw in content['hardware']:
                context += f"[Page {hw['page']}] {hw['text']}\n\n"
        
        if content['text']:
            context += "📄 TEXT CONTENT:\n"
            for text in content['text'][:3]:  # Limit per doc
                context += f"[Page {text['page']}] {text['text'][:500]}...\n\n"
        
        if content['visuals']:
            context += "🖼️ VISUAL ELEMENTS:\n"
            for visual in content['visuals']:
                types = visual['metadata'].get('metadata', {}).get('visual_types', [])
                context += f"[Page {visual['page']}] Types: {', '.join(types)}\n{visual['text']}\n\n"
    
    prompt = f"""Analyze information from MULTIPLE documents.

CRITICAL RULES:
1. ALWAYS specify which document each fact comes from: "According to [Document Name], ..."
2. If multiple documents have info, note it: "Both [Doc A] and [Doc B] mention..."
3. If documents conflict, acknowledge: "[Doc A] says X while [Doc B] says Y"
4. Include page numbers for specific details
5. Start by listing which documents you're referencing

QUESTION: {query}

CONTEXT:
{context}

YOUR ANSWER (with document attribution):"""
    
    response = await openai_client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You analyze multiple documents and ALWAYS clearly attribute information to specific documents. Never confuse sources."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=2000
    )
    
    # Build sources
    source_info = []
    seen = set()
    
    for source in sources[:10]:
        doc_name = source['metadata'].get('document_name', 'Unknown')
        page = source['metadata'].get('page', 0)
        source_type = source['metadata'].get('type', 'text')
        key = f"{doc_name}_{page}_{source_type}"
        
        if key not in seen:
            seen.add(key)
            source_info.append(SourceInfo(
                document_id=source['metadata'].get('document_id', ''),
                document_name=doc_name,
                page=page,
                type=source_type,
                relevance_score=float(source.get('final_score', 0)),
                visual_types=source['metadata'].get('metadata', {}).get('visual_types') if source_type == 'visual' else None,
                excerpt=source['text'][:200] + "..."
            ))
    
    scores = [s.get("final_score", 0) for s in sources[:5]]
    confidence = float(np.mean(scores)) if scores else 0.0
    if np.isnan(confidence):
        confidence = 0.0
    
    return {
        "answer": response.choices[0].message.content,
        "sources": source_info,
        "confidence": confidence,
        "documents_used": documents_used
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "documents": len(document_registry)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)