# main.py - IMPROVED WITH BETTER LLM PROMPTING
"""
🚀 UNIVERSAL MULTI-DOCUMENT CHATBOT API - PRODUCTION VERSION
Enhanced prompting for accurate answer extraction
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
import tempfile
import os
from datetime import datetime
import uuid
import logging

from app.document_processor import GPT5UniversalProcessor
from app.retrieval_engine import HybridRetriever
from app.models import (
    QueryRequest, QueryResponse, DocumentResponse,
    SourceInfo, DocumentMetadata, DocumentListResponse
)
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Universal AI Document Intelligence API",
    description="Production-ready document Q&A with hybrid retrieval"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
gpt5_processor = GPT5UniversalProcessor(
    openai_api_key=settings.OPENAI_API_KEY,
    qdrant_host=settings.QDRANT_HOST,
    qdrant_port=settings.QDRANT_PORT
)

retriever = HybridRetriever()

# Document registry
document_registry: Dict[str, Dict[str, Any]] = {}

# Ensure outputs directory
OUTPUTS_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    await gpt5_processor.initialize_collections()
    logger.info("✅ Universal API initialized")

@app.get("/")
async def root():
    return {
        "service": "Universal AI Document Intelligence",
        "version": "3.0 - Production",
        "status": "operational",
        "features": [
            "Hybrid retrieval (semantic + keyword)",
            "Smart ranking and scoring",
            "Works with any document type",
            "No hardcoding"
        ]
    }

@app.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    category: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[str] = None
):
    """Upload and process documents"""
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"📄 Processing: {file.filename}")
        
        result = await gpt5_processor.process_document(
            file_path=tmp_path,
            document_name=file.filename
        )
        
        metadata = DocumentMetadata(
            category=category,
            description=description,
            tags=tags.split(",") if tags else [],
            upload_date=datetime.now(),
            file_size=len(content),
            page_count=result["page_count"]
        )
        
        document_registry[result["document_id"]] = {
            "document_id": result["document_id"],
            "document_name": file.filename,
            "metadata": metadata.dict(),
            "chunks_created": result["chunks_created"],
            "page_count": result["page_count"],
            "upload_date": datetime.now().isoformat()
        }
        
        return DocumentResponse(
            document_id=result["document_id"],
            document_name=file.filename,
            status="success",
            chunks_created=result["chunks_created"],
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Query documents with improved LLM prompting
    """
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 NEW QUERY: {request.query}")
        logger.info(f"{'='*80}")
        
        # Retrieve relevant chunks using hybrid search
        document_filter = None
        if request.search_mode == "selected" and request.document_ids:
            document_filter = request.document_ids
        
        results = await retriever.retrieve(
            request.query,
            request.top_k,
            document_filter
        )
        
        if not results:
            logger.warning("⚠️ No results found")
            return {
                "answer": "No relevant information found in the uploaded documents.",
                "sources": [],
                "confidence": 0.0,
                "documents_used": []
            }
        
        logger.info(f"✅ Retrieved {len(results)} chunks")
        
        # Log what we're sending to the LLM
        logger.info(f"\n{'='*80}")
        logger.info(f"📤 CHUNKS BEING SENT TO LLM:")
        logger.info(f"{'='*80}")
        
        for idx, result in enumerate(results, 1):
            logger.info(f"\n[Chunk {idx}]")
            logger.info(f"  Page: {result['page']} | Type: {result['type']} | Score: {result.get('final_score', 0):.4f}")
            logger.info(f"  Preview: {result['text'][:150]}...")
        
        # Generate answer using GPT-5
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Build context with clear source attribution
        context_parts = []
        for idx, result in enumerate(results[:10], 1):
            doc_name = result.get('document_name', 'Unknown')
            page = result.get('page', 'N/A')
            chunk_type = result.get('type', 'text')
            text = result.get('text', '')
            
            # Limit text length for context window
            if len(text) > 800:
                text = text[:800] + "..."
            
            context_parts.append(
                f"[SOURCE {idx} - {doc_name}, Page {page}, Type: {chunk_type}]\n{text}\n"
            )
        
        context = "\n".join(context_parts)
        
        logger.info(f"\n📝 Context length: {len(context)} characters")
        
        # IMPROVED PROMPT - More explicit instructions
        prompt = f"""You are analyzing technical documentation to answer a specific question.

QUESTION:
{request.query}

AVAILABLE SOURCES:
{context}

INSTRUCTIONS:
1. READ EVERY SOURCE CAREFULLY - scan all sources before answering
2. Look for EXACT information requested (numbers, ratios, calculations, specifications)
3. If you find the answer, extract it PRECISELY and cite the page number
4. DO NOT say information is "not specified" if it exists in any source
5. DO NOT make calculations from unrelated hardware specifications
6. DO NOT assume or infer - only use explicitly stated information
7. If genuinely not found, say "This specific information is not present in the provided excerpts"

FORMAT YOUR ANSWER:
- Start with a direct answer to the question
- Include specific values, numbers, or details from the sources
- Cite page numbers in format: (Page X)
- If calculations are requested, show them clearly
- Be concise but complete

ANSWER:"""
        
        logger.info(f"🤖 Calling GPT-5...")
        
        # Call OpenAI with proper parameters for reasoning model
        response = await openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a precise technical document analyst. You extract information exactly as stated in sources and cite page numbers. You never claim information is missing if it exists in the provided text."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            
            max_completion_tokens=8000  # High limit for reasoning model
        )
        
        # Extract answer with detailed logging
        logger.info(f"✅ API Response received")
        logger.info(f"   Model: {response.model}")
        logger.info(f"   Tokens: {response.usage.total_tokens} (completion: {response.usage.completion_tokens})")
        logger.info(f"   Finish reason: {response.choices[0].finish_reason}")
        
        try:
            answer = response.choices[0].message.content
            
            if answer:
                logger.info(f"✅ Answer extracted ({len(answer)} chars)")
                logger.info(f"   Preview: {answer[:200]}...")
            else:
                logger.error(f"❌ Answer is empty!")
                answer = "Error: Received empty response from AI model."
                
        except (IndexError, AttributeError) as e:
            logger.error(f"❌ Failed to extract answer: {e}")
            answer = "Error: Could not extract answer from API response."
        
        # Build sources for response
        sources = []
        documents_used = set()
        
        for result in results[:5]:
            doc_id = result.get('document_id', '')
            doc_name = result.get('document_name', 'Unknown')
            page = result.get('page', 0)
            chunk_type = result.get('type', 'text')
            score = result.get('final_score', 0)
            text = result.get('text', '')
            
            documents_used.add(doc_name)
            
            sources.append(SourceInfo(
                document_id=doc_id,
                document_name=doc_name,
                page=page,
                type=chunk_type,
                relevance_score=float(score) if score else 0.0,
                excerpt=text[:200] + "..." if len(text) > 200 else text
            ))
        
        # Calculate confidence based on top score
        confidence = min(0.95, max(0.5, float(results[0].get('final_score', 0.7)))) if results else 0.5
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ QUERY COMPLETE")
        logger.info(f"   Answer length: {len(answer)} chars")
        logger.info(f"   Sources: {len(sources)}")
        logger.info(f"   Confidence: {confidence:.2f}")
        logger.info(f"{'='*80}\n")
        
        return {
            "answer": answer if answer else "No answer generated.",
            "sources": sources,
            "confidence": confidence,
            "documents_used": list(documents_used)
        }
    
    except Exception as e:
        logger.error(f"❌ Query Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all documents"""
    documents = list(document_registry.values())
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
        logger.error(f"❌ Error deleting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "3.0 - Production",
        "documents": len(document_registry),
        "models": {
            "analysis": "gpt-5-nano-2025-08-07",
            "embedding": "text-embedding-3-large"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)