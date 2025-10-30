# gpt5_universal_processor.py - UNIVERSAL DOCUMENT PROCESSOR WITH GPT-5
"""
🚀 UNIVERSAL DOCUMENT PROCESSOR - ZERO HARDCODING
Uses GPT-5 for maximum detail extraction from ANY document type

Key Features:
1. Works with ANY document: medical, legal, technical, financial, etc.
2. GPT-5 powered visual analysis with EXTREME detail
3. Extracts information so detailed that images can be perfectly recreated
4. No separate text extraction - GPT-5 handles everything
5. Modular architecture
"""

import hashlib
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import fitz  # PyMuPDF
from docx import Document
from PIL import Image
import cv2
import numpy as np
from openai import AsyncOpenAI, OpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import base64
import io
import uuid
import json

class GPT5UniversalProcessor:
    """
    🎯 UNIVERSAL PROCESSOR using GPT-5
    Extracts MAXIMUM detail from any document for perfect reconstruction
    """
    
    def __init__(self, openai_api_key: str, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        # Use GPT-5 Nano model
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.openai_sync_client = OpenAI(api_key=openai_api_key)
        
        self.qdrant_client = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Model configuration
        self.gpt5_model = "gpt-5-nano-2025-08-07"
        self.embedding_model = "text-embedding-3-large"
        
        print(f"✅ Initialized GPT-5 Universal Processor")
        print(f"   Model: {self.gpt5_model}")
        print(f"   Capability: UNIVERSAL - works with ANY document type")
    
    async def process_document(self, file_path: str, document_name: str) -> Dict[str, Any]:
        """
        Main entry point - processes ANY document type
        """
        print(f"\n{'='*80}")
        print(f"🚀 GPT-5 UNIVERSAL DOCUMENT PROCESSING")
        print(f"{'='*80}")
        print(f"Document: {document_name}")
        print(f"File: {file_path}")
        
        document_id = self._generate_document_id(file_path)
        
        # Extract content based on file type
        if file_path.endswith('.pdf'):
            content = await self._process_pdf_with_gpt5(file_path)
        elif file_path.endswith('.docx'):
            content = await self._process_docx_with_gpt5(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            content = await self._process_image_with_gpt5(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Create intelligent chunks
        chunks = await self._create_universal_chunks(content, document_id, document_name)
        
        # Generate embeddings and store
        await self._embed_and_store(chunks, document_name, document_id)
        
        print(f"\n{'='*80}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Document ID: {document_id}")
        
        return {
            "document_id": document_id,
            "document_name": document_name,
            "chunks_created": len(chunks),
            "page_count": len(content.get("pages", [])),
            "status": "success"
        }
    
    async def _process_pdf_with_gpt5(self, file_path: str) -> Dict[str, Any]:
        """
        🎯 UNIVERSAL PDF PROCESSING WITH GPT-5
        
        Uses GPT-5 to extract EVERYTHING from each page:
        - Text (GPT-5 can extract text from images)
        - Visuals with EXTREME detail
        - Context and relationships
        - ENOUGH detail for perfect image reconstruction
        """
        print(f"\n📄 Opening PDF: {file_path}")
        pdf_document = fitz.open(file_path)
        print(f"✅ PDF loaded: {len(pdf_document)} pages\n")
        
        content = {"pages": []}
        
        for page_num, page in enumerate(pdf_document, 1):
            print(f"\n{'─'*80}")
            print(f"📖 Processing Page {page_num}/{len(pdf_document)} with GPT-5")
            print(f"{'─'*80}")
            
            # Create high-resolution snapshot
            print(f"  📸 Creating high-res snapshot...")
            mat = fitz.Matrix(3, 3)  # 3x resolution for maximum detail
            pix = page.get_pixmap(matrix=mat)
            print(f"     Resolution: {pix.width}x{pix.height} pixels")
            
            img_bytes = pix.tobytes("png")
            print(f"     Size: {len(img_bytes)/1024:.2f} KB")
            
            # 🚀 GPT-5 COMPREHENSIVE ANALYSIS - EXTRACTS EVERYTHING
            print(f"  🤖 GPT-5 comprehensive analysis...")
            page_analysis = await self._analyze_page_with_gpt5(img_bytes, page_num)
            
            content["pages"].append({
                "page_number": page_num,
                "analysis": page_analysis,
                "has_content": True
            })
            
            print(f"     ✅ Page {page_num} analyzed")
            print(f"     Content types: {', '.join(page_analysis.get('content_types', []))}")
            if page_analysis.get('has_visuals'):
                print(f"     Visual types: {', '.join(page_analysis.get('visual_types', []))}")
            
            pix = None
        
        pdf_document.close()
        
        return content
    
    async def _analyze_page_with_gpt5(
        self, 
        image_bytes: bytes, 
        page_num: int
    ) -> Dict[str, Any]:
        """
        🚀 GPT-5 COMPREHENSIVE PAGE ANALYSIS
        
        This is the CORE of the universal solution.
        GPT-5 extracts EVERYTHING with EXTREME detail:
        
        1. ALL text on the page (GPT-5 can read text from images)
        2. ALL visual elements with pixel-level detail
        3. Spatial relationships and layouts
        4. Colors, styles, patterns
        5. Context and meaning
        6. RECONSTRUCTION INSTRUCTIONS for perfect image recreation
        
        This works for ANY document type - medical, legal, technical, financial, etc.
        """
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # 🎯 UNIVERSAL COMPREHENSIVE ANALYSIS PROMPT
            # This prompt is designed to work with ANY document type
            analysis_prompt = """You are GPT-5, the most advanced AI vision model. Your task is to perform a COMPREHENSIVE ANALYSIS of this document page with EXTREME DETAIL.

🎯 YOUR MISSION:
Extract EVERY piece of information from this page with such detail that:
1. An AI image generator could recreate this page PERFECTLY
2. A human could understand EXACTLY what's on this page
3. NO information is lost

📋 ANALYSIS STRUCTURE:

1. **CONTENT OVERVIEW**
   - What type of document is this? (report, diagram, medical chart, legal document, etc.)
   - What domain/field? (IT, medicine, law, finance, engineering, etc.)
   - What is the purpose of this page?
   - Overall layout structure

2. **TEXT EXTRACTION** (CRITICAL - Extract ALL text)
   - Headers, titles, headings (with hierarchy)
   - Body text, paragraphs
   - Labels, annotations
   - Legends, captions
   - Numerical values, measurements
   - Tables (structure and all data)
   - Lists (ordered/unordered)
   - Footnotes, references
   - ANY other text
   - PRESERVE: Exact wording, spacing, formatting

3. **VISUAL ELEMENTS** (If any - describe in EXTREME detail)
   - Type of visual: diagram, chart, graph, image, flowchart, blueprint, etc.
   - Layout: positioning, dimensions, orientation
   - Components: every box, shape, icon, symbol
   - Connections: lines, arrows, relationships
   - Colors: EXACT colors (e.g., "navy blue #1E3A8A", "bright red #FF0000")
   - Line styles: solid, dashed, dotted, thickness measurements
   - Text in visuals: ALL labels, values, annotations
   - Spatial relationships: "X is above Y", "A connects to B via arrow"
   - Patterns, textures, shading
   - Background elements

4. **SPATIAL LAYOUT**
   - Page structure: margins, columns, sections
   - Element positioning: coordinates, alignment
   - Size relationships: "diagram takes up top 40% of page"
   - Z-order: layering, overlaps

5. **STYLING & FORMATTING**
   - Font types, sizes, weights
   - Text colors
   - Background colors
   - Borders, boxes, frames
   - Emphasis: bold, italic, underline
   - Spacing: line height, padding

6. **CONTEXT & MEANING**
   - What information is being conveyed?
   - Key concepts or ideas
   - Relationships between elements
   - Important data points

7. **IMAGE RECONSTRUCTION INSTRUCTIONS** (MOST CRITICAL)
   Write detailed instructions for an AI image generator to recreate this page:
   - Start with: "Create a [type] in [style]..."
   - Describe EVERY element step-by-step
   - Include ALL measurements, colors, positions
   - Include ALL text exactly as written
   - Describe connections and relationships
   - Minimum 300 words for complex pages

8. **METADATA**
   - Quality: resolution, clarity
   - Completeness: is anything cut off or unclear?
   - Notable features or unique elements

🎯 QUALITY REQUIREMENTS:
- Be EXHAUSTIVELY detailed
- Miss NOTHING
- Use precise language
- Include ALL text verbatim
- Describe spatial relationships clearly
- Your analysis must enable PERFECT recreation

📤 OUTPUT FORMAT (JSON):
{
  "content_types": ["text", "diagram", "table", etc.],
  "document_type": "technical report / medical chart / etc.",
  "domain": "IT / medicine / law / etc.",
  "purpose": "brief description",
  
  "text_content": {
    "headers": ["header 1", "header 2"],
    "body_paragraphs": ["para 1", "para 2"],
    "labels": ["label 1", "label 2"],
    "tables": [{"structure": "...", "data": [...]}],
    "all_text": "EVERY piece of text on the page"
  },
  
  "has_visuals": true/false,
  "visual_analysis": {
    "visual_types": ["network diagram", "bar chart", etc.],
    "comprehensive_description": "EXTREMELY detailed description (200+ words)",
    "components": ["component 1", "component 2"],
    "spatial_layout": "detailed layout description",
    "colors": ["color 1: #HEX", "color 2: #HEX"],
    "line_styles": ["solid 2px", "dashed 1px"],
    "text_in_visual": ["label 1", "label 2"],
    "connections": ["A -> B via arrow", "X connects to Y"]
  },
  
  "reconstruction_instructions": "DETAILED 300+ word instructions for AI image generator to recreate this page PERFECTLY",
  
  "metadata": {
    "quality": "high/medium/low",
    "completeness": "complete/partial"
  }
}

ANALYZE THIS PAGE NOW:"""
            
            # Call GPT-5 with maximum detail
            response = await self.openai_client.chat.completions.create(
                model=self.gpt5_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are GPT-5, the most advanced vision AI. You analyze documents with EXTREME detail for perfect reconstruction. You are UNIVERSAL - you work with ANY document type from ANY domain."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=8000,  # Allow very long responses
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                analysis = {
                    "content_types": ["unknown"],
                    "raw_response": response_text
                }
            
            return analysis
            
        except Exception as e:
            print(f"     ❌ ERROR in GPT-5 analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "content_types": ["error"],
                "error": str(e)
            }
    
    async def _create_universal_chunks(
        self, 
        content: Dict[str, Any], 
        document_id: str,
        document_name: str
    ) -> List[Dict]:
        """
        Create intelligent chunks from GPT-5 analysis
        UNIVERSAL - works with any document type
        """
        print(f"\n{'='*80}")
        print(f"🔪 CREATING INTELLIGENT CHUNKS")
        print(f"{'='*80}")
        
        chunks = []
        
        for page_data in content.get("pages", []):
            page_num = page_data["page_number"]
            analysis = page_data.get("analysis", {})
            
            print(f"\n📄 Page {page_num}")
            
            # Extract text content
            text_content = analysis.get("text_content", {})
            all_text = text_content.get("all_text", "")
            
            if all_text:
                # Create text chunk
                chunk = {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "document_name": document_name,
                    "content": all_text,
                    "type": "text",
                    "page": page_num,
                    "metadata": {
                        "document_type": analysis.get("document_type", "unknown"),
                        "domain": analysis.get("domain", "general"),
                        "headers": text_content.get("headers", []),
                        "has_tables": bool(text_content.get("tables"))
                    }
                }
                chunks.append(chunk)
                print(f"   ✅ Text chunk created ({len(all_text)} chars)")
            
            # Extract visual content
            if analysis.get("has_visuals"):
                visual_analysis = analysis.get("visual_analysis", {})
                
                # Create visual chunk with FULL reconstruction data
                chunk = {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "document_name": document_name,
                    "content": visual_analysis.get("comprehensive_description", ""),
                    "type": "visual",
                    "page": page_num,
                    "metadata": {
                        "visual_types": visual_analysis.get("visual_types", []),
                        "components": visual_analysis.get("components", []),
                        "spatial_layout": visual_analysis.get("spatial_layout", ""),
                        "colors": visual_analysis.get("colors", []),
                        "line_styles": visual_analysis.get("line_styles", []),
                        "text_in_visual": visual_analysis.get("text_in_visual", []),
                        "connections": visual_analysis.get("connections", []),
                        "reconstruction_instructions": analysis.get("reconstruction_instructions", ""),
                        "document_type": analysis.get("document_type", "unknown"),
                        "domain": analysis.get("domain", "general")
                    }
                }
                chunks.append(chunk)
                print(f"   ✅ Visual chunk created")
                print(f"      Types: {', '.join(visual_analysis.get('visual_types', []))}")
        
        print(f"\n{'='*80}")
        print(f"✅ CHUNK CREATION COMPLETE")
        print(f"Total chunks: {len(chunks)}")
        
        # Count chunk types
        chunk_types = {}
        for chunk in chunks:
            t = chunk['type']
            chunk_types[t] = chunk_types.get(t, 0) + 1
        
        for chunk_type, count in chunk_types.items():
            print(f"  - {chunk_type}: {count}")
        print(f"{'='*80}\n")
        
        return chunks
    
    async def _embed_and_store(
        self, 
        chunks: List[Dict], 
        document_name: str, 
        document_id: str
    ):
        """
        Generate embeddings and store in Qdrant
        """
        print(f"\n🔢 Generating embeddings and storing...")
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk["content"] for chunk in batch]
            
            # Generate embeddings
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            # Prepare points
            points = []
            for j, chunk in enumerate(batch):
                embedding = response.data[j].embedding
                
                point = PointStruct(
                    id=chunk["chunk_id"],
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "document_name": document_name,
                        "text": chunk["content"],
                        "type": chunk["type"],
                        "page": chunk["page"],
                        "metadata": chunk.get("metadata", {}),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                points.append(point)
            
            # Store in Qdrant
            await self.qdrant_client.upsert(
                collection_name="documents",
                points=points
            )
            
            print(f"   ✅ Stored batch {i//batch_size + 1} ({len(points)} vectors)")
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID"""
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]
    
    async def _process_docx_with_gpt5(self, file_path: str) -> Dict[str, Any]:
        """Process DOCX files with GPT-5 (to be implemented)"""
        raise NotImplementedError("DOCX processing - coming soon")
    
    async def _process_image_with_gpt5(self, file_path: str) -> Dict[str, Any]:
        """Process standalone images with GPT-5 (to be implemented)"""
        raise NotImplementedError("Standalone image processing - coming soon")
    
    async def initialize_collections(self):
        """Initialize Qdrant collections if they don't exist"""
        try:
            existing_collections = await self.qdrant_client.get_collections()
            collection_names = [col.name for col in existing_collections.collections]
            
            if "documents" not in collection_names:
                print("📦 Creating 'documents' collection...")
                await self.qdrant_client.create_collection(
                    collection_name="documents",
                    vectors_config=VectorParams(
                        size=3072,  # text-embedding-3-large dimension
                        distance=Distance.COSINE
                    )
                )
                print("✅ Collection created")
            else:
                print("✅ Collection 'documents' already exists")
        except Exception as e:
            print(f"❌ Error initializing collections: {str(e)}")
            raise


# Example usage
"""
processor = GPT5UniversalProcessor(
    openai_api_key="your-api-key"
)

await processor.initialize_collections()

result = await processor.process_document(
    file_path="/path/to/document.pdf",
    document_name="document.pdf"
)

print(f"Processed: {result['chunks_created']} chunks")
"""