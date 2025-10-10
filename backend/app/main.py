# main.py - FastAPI Application
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import tempfile
import os

from app.config import settings
from app.document_processor import DocumentProcessor
from app.retrieval_engine import HybridRetriever

app = FastAPI(title="Intelligent Document Chatbot API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_processor = DocumentProcessor()
retriever = HybridRetriever()

# Request/Response models
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

@app.on_event("startup")
async def startup_event():
    """Initialize collections on startup"""
    await document_processor.initialize_collections()

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    tmp_path = None
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"Processing file: {file.filename} at {tmp_path}")
        
        # Process document
        result = await document_processor.process_document(
            tmp_path,
            file.filename
        )
        
        return DocumentResponse(**result)
    
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document knowledge base"""
    try:
        # Retrieve relevant chunks
        results = await retriever.retrieve(
            request.query,
            request.top_k,
            request.document_ids
        )
        
        # Build context
        context = "\n\n".join([r["text"] for r in results])
        
        # Generate response using GPT-4
        response = await generate_answer(request.query, context, results)
        
        return QueryResponse(
            answer=response["answer"],
            sources=response["sources"],
            confidence=response["confidence"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_answer(query: str, context: str, sources: List[Dict]) -> Dict:
    """Generate answer using GPT-4"""
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    prompt = f"""You are an intelligent assistant helping users understand technical documents. 
    Use the following context to answer the question. Be precise and cite sources.
    
    Context:
    {context}
    
    Question: {query}
    
    Instructions:
    1. Provide a comprehensive answer based on the context
    2. If the context doesn't contain enough information, say so
    3. Reference specific documents and page numbers when possible
    4. For technical diagrams or images mentioned, describe their relevance
    """
    
    response = await openai_client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a technical documentation expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    # Extract source information
    source_info = []
    for source in sources[:5]:  # Top 5 sources
        source_info.append({
            "document": source["metadata"]["document_name"],
            "page": source["metadata"]["page"],
            "type": source["metadata"]["type"],
            "relevance_score": source["final_score"]
        })
    
    return {
        "answer": response.choices[0].message.content,
        "sources": source_info,
        "confidence": float(np.mean([s["final_score"] for s in sources[:5]]))
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)