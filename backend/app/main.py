# main.py - FIXED v2.1 with Image Serving and Better Logic
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from openai import AsyncOpenAI, OpenAI
from typing import List, Optional, Dict, Any
import tempfile
import os
from datetime import datetime
from collections import defaultdict
import uuid
import base64
import requests

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

app = FastAPI(title="Multi-Document Chatbot API - FIXED v2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_processor = DocumentProcessor()
retriever = HybridRetriever()

# Document registry
document_registry: Dict[str, Dict[str, Any]] = {}

# Ensure outputs directory exists
OUTPUTS_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    await document_processor.initialize_collections()

# NEW: Image serving endpoint
@app.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve generated diagram images"""
    file_path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path, media_type="image/png")

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

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query with FIXED answer generation and diagram support"""
    try:
        logger.info(f"\nQUERY: {request.query}")
        logger.info(f"MODE: {request.search_mode}")
        
        # Check if user wants a diagram
        diagram_keywords = ["draw", "diagram", "visualize", "create diagram", "show diagram", 
                          "generate diagram", "site diagram", "network diagram", "topology"]
        wants_diagram = any(keyword in request.query.lower() for keyword in diagram_keywords)
        
        document_filter = None
        if request.search_mode == "selected" and request.document_ids:
            document_filter = request.document_ids
        
        results = await retriever.retrieve(
            request.query,
            request.top_k,
            document_filter
        )
        
        logger.info(f"Retrieved {len(results)} results")
        
        if wants_diagram:
            # Generate diagram
            response = await generate_diagram_response(request.query, results)
        else:
            # Generate text answer (FIXED version)
            response = await generate_precise_answer(request.query, results)
        
        return response
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def generate_diagram_response(query: str, sources: List[Dict]) -> Dict:
    """Generate a diagram using DALL-E 3"""
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Build diagram prompt from sources
    diagram_context = ""
    for source in sources[:5]:
        if source['metadata'].get('type') == 'visual':
            diagram_context += f"\n{source['text']}\n"
    
    # If no visual context, use text descriptions
    if not diagram_context:
        for source in sources[:3]:
            diagram_context += f"\n{source['text'][:200]}...\n"
    
    # Create enhanced prompt
    enhanced_prompt = f"""Create a professional technical network diagram based on this description:

{diagram_context[:2000]}

Style: Clean, professional, technical documentation style
Background: White
Lines: Clear black lines for connections, use arrows for direction
Components: Use standard shapes (rectangles for switches/routers, clouds for external connections, cylinders for databases)
Text: Clear, readable labels on all components with device names and interface identifiers
Include: Device names (like L35AC111, L35EXR1), interface labels (like Et0/1), connection types

Make it look like a professional network architecture diagram from technical documentation, similar to Cisco or Juniper network designs."""
    
    try:
        logger.info("Generating diagram with DALL-E 3...")
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt[:4000],  # DALL-E has limits
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        # Download image
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        image_bytes = image_response.content
        
        # Save to outputs
        image_filename = f"diagram_{uuid.uuid4()}.png"
        image_path = os.path.join(OUTPUTS_DIR, image_filename)
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        logger.info(f"Diagram saved to {image_path}")
        
        # FIXED: Use HTTP endpoint instead of computer:// protocol
        http_image_url = f"http://localhost:8000/images/{image_filename}"
        
        return {
            "answer": f"I've generated the network diagram based on the document. The diagram shows the site topology with devices interconnected by Layer 2 and Layer 3 technologies.\n\nYou can download the diagram using the link below.",
            "sources": [],
            "confidence": 1.0,
            "documents_used": [s['metadata'].get('document_name', 'Unknown') for s in sources[:3]],
            "image_url": http_image_url,  # FIXED: Use HTTP URL
            "diagram_generated": True
        }
        
    except Exception as e:
        logger.error(f"Error generating diagram: {str(e)}")
        return {
            "answer": f"I attempted to generate a diagram but encountered an error: {str(e)}\n\nHowever, I can describe the network topology from the documents:\n\n{diagram_context[:500]}",
            "sources": [],
            "confidence": 0.5,
            "documents_used": [],
            "image_url": None,
            "diagram_generated": False
        }


async def generate_precise_answer(query: str, sources: List[Dict]) -> Dict:
    """
    FIXED v2.1: Generate PRECISE, CONCISE answers with improved logic
    """
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Group by document and organize by type
    doc_groups = defaultdict(lambda: {
        'doc_name': '',
        'text': [],
        'hardware': [],
        'visuals': [],
        'tables': []
    })
    
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
            # Check if it contains table-like data
            text = source['text']
            if any(indicator in text.lower() for indicator in ['table', '|', 'hostname', 'interface', 'priority', 'state']):
                doc_groups[doc_id]['tables'].append(source_with_page)
            else:
                doc_groups[doc_id]['text'].append(source_with_page)
    
    # Build STRUCTURED context
    context = f"=== RETRIEVED INFORMATION FROM {len(doc_groups)} DOCUMENT(S) ===\n\n"
    
    documents_used = []
    
    for doc_idx, (doc_id, content) in enumerate(doc_groups.items(), 1):
        doc_name = content['doc_name']
        documents_used.append(doc_name)
        
        context += f"\n{'='*80}\n📄 DOCUMENT {doc_idx}: {doc_name}\n{'='*80}\n\n"
        
        # Tables (most important for technical questions)
        if content['tables']:
            context += "📊 TABLE DATA (CRITICAL - READ CAREFULLY):\n"
            for table in content['tables']:
                context += f"[Page {table['page']}]\n{table['text']}\n\n"
        
        # Hardware specs
        if content['hardware']:
            context += "🔧 HARDWARE SPECIFICATIONS:\n"
            for hw in content['hardware']:
                context += f"[Page {hw['page']}] {hw['text']}\n\n"
        
        # Visual descriptions
        if content['visuals']:
            context += "🖼️ DIAGRAMS/VISUALS:\n"
            for visual in content['visuals']:
                types = visual['metadata'].get('metadata', {}).get('visual_types', [])
                context += f"[Page {visual['page']}] Types: {', '.join(types)}\n{visual['text']}\n\n"
        
        # Regular text
        if content['text']:
            context += "📝 TEXT CONTENT:\n"
            for text in content['text'][:2]:
                context += f"[Page {text['page']}] {text['text'][:400]}...\n\n"
    
    # IMPROVED PROMPT v2.1: Better logical reasoning for HSRP questions
    prompt = f"""You are a precise technical documentation analyst. Answer the question using ONLY the provided context.

🎯 CRITICAL RULES:
1. BE CONCISE: Maximum 200 words unless question requires detail
2. CITE SPECIFICALLY: Always cite "Page X, Table Y" or "Page X"
3. USE EXACT DATA: Quote table values, IP addresses, configurations exactly
4. THINK LOGICALLY: 
   - If a device has NO standby configured, it HAS NO REDUNDANCY
   - If only ONE device is active and NO standby exists, then FAILURE = LOSS OF SERVICE
   - Don't assume redundancy exists if not shown in tables
5. NO SPECULATION: If info not in context, say "Not found in provided documents"
6. STRUCTURE: Use bullet points for technical details
7. BE DIRECT: Start with the answer, not background

QUESTION: {query}

CONTEXT:
{context}

LOGICAL REASONING EXAMPLE:
If Table shows:
  VLAN 110: L35EXR1 (active), L35EXR2 (standby) → Has redundancy ✓
  VLAN 6: L35EXR2 (active), NO L35EXR1 → NO redundancy ✗
  
Then: If L35EXR2 fails on VLAN 6 → Gateway LOST (no standby)

FORMAT YOUR ANSWER LIKE THIS:
[Direct answer in 1-2 sentences]

Key Details:
• Point 1 (Page X, Table Y)
• Point 2 (Page X)
• Point 3 (Page X)

[Brief conclusion with logical reasoning]

YOUR PRECISE ANSWER:"""
    
    response = await openai_client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a precise technical analyst who:
1. Gives short, accurate answers (max 200 words)
2. Cites specific pages and tables
3. Uses logical reasoning
4. NEVER assumes redundancy if not shown in data
5. If only one device is active with NO standby, states clearly: "No redundancy - failure will cause outage"
Never speculate. Use exact data from tables."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,  # Maximum precision
        max_tokens=800,  # Limit verbosity
        presence_penalty=0.3,  # Reduce repetition
        frequency_penalty=0.3
    )
    
    # Build sources
    source_info = []
    seen = set()
    
    for source in sources[:8]:
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
                excerpt=source['text'][:150] + "..."
            ))
    
    scores = [s.get("final_score", 0) for s in sources[:5]]
    confidence = float(np.mean(scores)) if scores else 0.0
    if np.isnan(confidence):
        confidence = 0.0
    
    return {
        "answer": response.choices[0].message.content,
        "sources": source_info,
        "confidence": confidence,
        "documents_used": documents_used,
        "image_url": None,
        "diagram_generated": False
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "documents": len(document_registry),
        "version": "FIXED_v2.1"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)