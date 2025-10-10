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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
        logger.info(f"\n{'='*60}")
        logger.info(f"QUERY: {request.query}")
        logger.info(f"{'='*60}")
        
        # Retrieve relevant chunks
        results = await retriever.retrieve(
            request.query,
            request.top_k,
            request.document_ids
        )
        
        logger.info(f"\nRetrieved {len(results)} results")
        
        # Build context
        context = "\n\n".join([r["text"] for r in results])
        
        # Generate response using GPT-4
        response = await generate_answer(request.query, context, results)
        
        logger.info(f"\nGenerated answer ({len(response['answer'])} chars)")
        logger.info(f"Confidence: {response['confidence']:.2%}")
        
        return QueryResponse(
            answer=response["answer"],
            sources=response["sources"],
            confidence=response["confidence"]
        )
    
    except Exception as e:
        logger.error(f"Error in query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    
async def generate_answer(query: str, context: str, sources: List[Dict]) -> Dict:
    """Generate answer using GPT-4 with explicit text+image fusion"""
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Separate text and image sources
    text_sources = [s for s in sources if s['metadata'].get('type') == 'text']
    image_sources = [s for s in sources if s['metadata'].get('type') == 'image']
    
    # Build enhanced context
    enhanced_context = "=== DOCUMENT CONTENT ===\n\n"
    
    # Group by page for better context
    page_content = {}
    for source in sources[:10]:  # Top 10 sources
        page = source['metadata'].get('page', 0)
        doc_name = source['metadata'].get('document_name', 'Unknown')
        source_type = source['metadata'].get('type', 'text')
        
        page_key = f"{doc_name}_Page{page}"
        if page_key not in page_content:
            page_content[page_key] = {
                'text': [],
                'images': []
            }
        
        if source_type == 'text':
            page_content[page_key]['text'].append(source['text'])
        else:
            page_content[page_key]['images'].append(source['text'])
    
    # Build context with page grouping
    for page_key, content in page_content.items():
        enhanced_context += f"\n## {page_key}\n"
        
        if content['text']:
            enhanced_context += "\n**Text Content:**\n"
            for text in content['text']:
                enhanced_context += f"- {text}\n"
        
        if content['images']:
            enhanced_context += "\n**Images/Diagrams on this page:**\n"
            for img_desc in content['images']:
                enhanced_context += f"- {img_desc}\n"
        
        enhanced_context += "\n"
    
    prompt = f"""You are a technical documentation expert helping users understand network design documents.

CRITICAL INSTRUCTIONS:
1. Prioritize SPECIFIC details (model numbers, exact specs) over generic descriptions
2. When both text and image descriptions exist for the same page, COMBINE them
3. For hardware questions, always mention the EXACT model numbers if available
4. Be concise but complete - include all relevant technical details

Context:
{enhanced_context}

Question: {query}

Instructions:
- Provide a direct, accurate answer based on the context
- Cite specific page numbers when referencing information
- If technical specs are mentioned (model numbers, speeds, etc.), include them
- If images contain additional details, mention what they show
- Keep the answer focused and avoid unnecessary elaboration
"""
    
    response = await openai_client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise technical documentation expert. Always include specific model numbers, specs, and technical details when available."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,  # Lower temperature for more factual responses
        max_tokens=1000
    )
    
    # Extract source information
    source_info = []
    seen_pages = set()
    
    for source in sources[:5]:  # Top 5 sources
        page = source['metadata'].get('page', 0)
        doc_name = source['metadata'].get('document_name', 'Unknown')
        page_key = f"{doc_name}_Page{page}"
        
        if page_key not in seen_pages:
            seen_pages.add(page_key)
            source_info.append({
                "document": doc_name,
                "page": page,
                "type": source['metadata'].get('type', 'text'),
                "relevance_score": source.get("final_score", 0)
            })
    
    return {
        "answer": response.choices[0].message.content,
        "sources": source_info,
        "confidence": float(np.mean([s.get("final_score", 0) for s in sources[:5]]))
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)