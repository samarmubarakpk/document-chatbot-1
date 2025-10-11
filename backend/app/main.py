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
    text_sources = [s for s in sources if s['metadata'].get('type') in ['text', 'hardware_spec']]
    image_sources = [s for s in sources if s['metadata'].get('type') == 'image']
    hardware_sources = [s for s in sources if s['metadata'].get('section_type') == 'hardware']
    
    # Extract ALL model numbers from sources
    all_model_numbers = []
    for source in sources:
        models = source['metadata'].get('model_numbers', [])
        all_model_numbers.extend(models)
    
    # Remove duplicates
    all_model_numbers = list(set(all_model_numbers))
    
    # Build enhanced context
    enhanced_context = "=== DOCUMENT CONTENT ===\n\n"
    
    # CRITICAL: Show model numbers upfront if they exist
    if all_model_numbers:
        enhanced_context += "⚠️ IMPORTANT - MODEL NUMBERS FOUND IN DOCUMENT:\n"
        for model in all_model_numbers:
            enhanced_context += f"  • {model}\n"
        enhanced_context += "\n"
    
    # Group by page for better context
    page_content = {}
    for source in sources[:10]:  # Top 10 sources
        page = source['metadata'].get('page', 0)
        doc_name = source['metadata'].get('document_name', 'Unknown')
        source_type = source['metadata'].get('type', 'text')
        is_hardware = source['metadata'].get('section_type') == 'hardware'
        
        page_key = f"{doc_name}_Page{page}"
        if page_key not in page_content:
            page_content[page_key] = {
                'text': [],
                'hardware': [],
                'images': []
            }
        
        if is_hardware or source_type == 'hardware_spec':
            page_content[page_key]['hardware'].append(source['text'])
        elif source_type == 'text':
            page_content[page_key]['text'].append(source['text'])
        else:
            page_content[page_key]['images'].append(source['text'])
    
    # Build context with clear separation
    for page_key, content in page_content.items():
        enhanced_context += f"\n## {page_key}\n"
        
        # HARDWARE SPECS FIRST (highest priority)
        if content['hardware']:
            enhanced_context += "\n**🔧 HARDWARE SPECIFICATIONS:**\n"
            for hw_text in content['hardware']:
                enhanced_context += f"{hw_text}\n\n"
        
        # Then text content
        if content['text']:
            enhanced_context += "\n**📄 Text Content:**\n"
            for text in content['text']:
                enhanced_context += f"{text}\n\n"
        
        # Images last
        if content['images']:
            enhanced_context += "\n**🖼️ Images/Diagrams on this page:**\n"
            for img_desc in content['images']:
                enhanced_context += f"- {img_desc}\n"
        
        enhanced_context += "\n" + "-"*80 + "\n"
    
    # Build strict prompt
    prompt = f"""You are a technical documentation expert analyzing network design documents.

⚠️ CRITICAL INSTRUCTIONS - READ CAREFULLY:

1. The context below contains REAL information extracted from the document
2. If model numbers, specifications, or technical details appear in the context, YOU MUST use them
3. NEVER say information is "not provided" if it EXISTS in the context
4. When hardware specifications are marked with 🔧, they contain exact technical details
5. Prioritize hardware specifications over image descriptions
6. Always cite the specific page where you found information

QUESTION: {query}

CONTEXT FROM DOCUMENT:
{enhanced_context}

YOUR TASK:
- Answer the question using ONLY the information in the context above
- If model numbers or specifications are shown in the context, include them in your answer
- If images describe hardware, cross-reference with hardware specifications from the same page
- Be specific and technical - use exact model numbers and specifications
- Cite page numbers

Answer:"""
    
    response = await openai_client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {
                "role": "system", 
                "content": "You are a precise technical expert. When analyzing context, you extract and use ALL available technical specifications. You NEVER claim information is missing if it exists in the provided context. You prioritize explicit specifications over generic descriptions."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        temperature=0.0,  # Maximum factual accuracy
        max_tokens=1500
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