# main.py - Updated to handle new "visual" chunk type
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
        
        logger.info(f"Processing file: {file.filename} at {tmp_path}")
        
        # Process document
        result = await document_processor.process_document(
            tmp_path,
            file.filename
        )
        
        return DocumentResponse(**result)
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
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
        
        # Generate response using GPT-4
        response = await generate_answer(request.query, results)
        
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
    
    
async def generate_answer(query: str, sources: List[Dict]) -> Dict:
    """
    Generate answer using GPT-4 with page-aware visual context
    Updated to handle new "visual" chunk type
    """
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Separate sources by type
    text_sources = [s for s in sources if s['metadata'].get('type') in ['text']]
    hardware_sources = [s for s in sources if s['metadata'].get('type') == 'hardware_spec']
    visual_sources = [s for s in sources if s['metadata'].get('type') == 'visual']
    
    # Extract ALL model numbers from hardware sources
    all_model_numbers = []
    for source in hardware_sources:
        models = source['metadata'].get('metadata', {}).get('model_numbers', [])
        all_model_numbers.extend(models)
    
    # Remove duplicates
    all_model_numbers = list(set(all_model_numbers))
    
    # Build enhanced context organized by page
    enhanced_context = "=== DOCUMENT CONTENT ===\n\n"
    
    # Show model numbers upfront if they exist
    if all_model_numbers:
        enhanced_context += "⚠️ IMPORTANT - MODEL NUMBERS FOUND IN DOCUMENT:\n"
        for model in all_model_numbers:
            enhanced_context += f"  • {model}\n"
        enhanced_context += "\n"
    
    # Group sources by page for better context
    page_content = {}
    for source in sources[:10]:  # Top 10 sources
        page = source['metadata'].get('page', 0)
        doc_name = source['metadata'].get('document_name', 'Unknown')
        source_type = source['metadata'].get('type', 'text')
        
        page_key = f"{doc_name}_Page{page}"
        if page_key not in page_content:
            page_content[page_key] = {
                'text': [],
                'hardware': [],
                'visuals': []
            }
        
        if source_type == 'hardware_spec':
            page_content[page_key]['hardware'].append(source)
        elif source_type == 'visual':
            page_content[page_key]['visuals'].append(source)
        else:
            page_content[page_key]['text'].append(source)
    
    # Build context with clear separation
    for page_key, content in page_content.items():
        enhanced_context += f"\n## {page_key}\n"
        
        # HARDWARE SPECS FIRST (highest priority)
        if content['hardware']:
            enhanced_context += "\n**🔧 HARDWARE SPECIFICATIONS:**\n"
            for hw_source in content['hardware']:
                enhanced_context += f"{hw_source['text']}\n\n"
        
        # Then regular text content
        if content['text']:
            enhanced_context += "\n**📄 Text Content:**\n"
            for text_source in content['text']:
                enhanced_context += f"{text_source['text']}\n\n"
        
        # Visual content - with metadata
        if content['visuals']:
            enhanced_context += "\n**🖼️ Visual Elements on this page:**\n"
            for visual_source in content['visuals']:
                visual_types = visual_source['metadata'].get('metadata', {}).get('visual_types', [])
                key_elements = visual_source['metadata'].get('metadata', {}).get('key_elements', [])
                
                enhanced_context += f"\n📊 Visual Type(s): {', '.join(visual_types)}\n"
                if key_elements:
                    enhanced_context += f"🔑 Key Elements: {', '.join(key_elements)}\n"
                enhanced_context += f"Description: {visual_source['text']}\n\n"
        
        enhanced_context += "\n" + "-"*80 + "\n"
    
    # Build strict prompt
    prompt = f"""You are a technical documentation expert analyzing network design documents.

⚠️ CRITICAL INSTRUCTIONS - READ CAREFULLY:

1. The context below contains REAL information extracted from the document (both text and visual analysis)
2. If model numbers, specifications, or technical details appear in the context, YOU MUST use them
3. NEVER say information is "not provided" if it EXISTS in the context
4. When hardware specifications are marked with 🔧, they contain exact technical details
5. When visual elements are marked with 🖼️, they describe diagrams, charts, or tables from the document
6. Visual descriptions often contain critical information like network topology, connections, or data relationships
7. Always cite the specific page where you found information
8. Combine information from text, hardware specs, and visual descriptions to give complete answers

QUESTION: {query}

CONTEXT FROM DOCUMENT:
{enhanced_context}

YOUR TASK:
- Answer the question using ONLY the information in the context above
- If model numbers or specifications are shown in the context, include them in your answer
- If visual descriptions provide relevant information (diagrams, charts, tables), incorporate that into your answer
- Be specific and technical - use exact model numbers and specifications
- Cite page numbers for your sources
- If information comes from a visual element, mention that (e.g., "as shown in the network diagram on page 4")

Answer:"""
    
    response = await openai_client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {
                "role": "system", 
                "content": "You are a precise technical expert. When analyzing context, you extract and use ALL available information from text, specifications, and visual descriptions. You NEVER claim information is missing if it exists in the provided context. You synthesize information from multiple sources (text + visuals) to provide complete answers."
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
        source_type = source['metadata'].get('type', 'text')
        page_key = f"{doc_name}_Page{page}"
        
        if page_key not in seen_pages:
            seen_pages.add(page_key)
            
            # Add visual types if it's a visual source
            extra_info = {}
            if source_type == 'visual':
                visual_types = source['metadata'].get('metadata', {}).get('visual_types', [])
                if visual_types:
                    extra_info['visual_types'] = visual_types
            
            source_info.append({
                "document": doc_name,
                "page": page,
                "type": source_type,
                "relevance_score": source.get("final_score", 0),
                **extra_info
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