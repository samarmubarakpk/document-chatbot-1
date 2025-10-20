# document_processor_universal.py - ZERO HARDCODING, UNIVERSAL FOR ALL DOCUMENTS
"""
Universal Document Processor with:
1. ZERO hardcoding - works with ANY document type
2. AI-powered content classification (no keyword matching)
3. Top-tier visual analysis using GPT-4 Vision
4. Comprehensive diagram understanding
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
import requests
import tiktoken
from minio import Minio
import json
import uuid
import base64
import io

from app.config import settings

class UniversalDocumentProcessor:
    """
    Universal document processor with ZERO hardcoding.
    Works with ANY document type: medical, legal, technical, financial, etc.
    """
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.qdrant_client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE
        )
        self.tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL)
        
    async def initialize_collections(self):
        """Initialize Qdrant collections - only create if they don't exist"""
        collections = {
            "documents": {
                "size": 3072,
                "distance": Distance.COSINE
            }
        }
        
        for collection_name, config in collections.items():
            try:
                existing_collections = await self.qdrant_client.get_collections()
                collection_names = [col.name for col in existing_collections.collections]
                
                if collection_name in collection_names:
                    print(f"✅ Collection '{collection_name}' already exists")
                    continue
                
                print(f"📦 Creating new collection '{collection_name}'")
                await self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=config["size"],
                        distance=config["distance"]
                    )
                )
                print(f"✅ Collection '{collection_name}' created")
                
            except Exception as e:
                print(f"❌ Error initializing collection '{collection_name}': {str(e)}")
                raise
    
    async def process_document(self, file_path: str, document_name: str) -> Dict[str, Any]:
        """Main document processing pipeline"""
        document_id = self.generate_document_id(file_path)
        
        # Extract content based on file type
        if file_path.endswith('.pdf'):
            content = await self.process_pdf(file_path)
        elif file_path.endswith('.docx'):
            content = await self.process_docx(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            content = await self.process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Create smart chunks (AI-powered, no hardcoding)
        chunks = await self.create_intelligent_chunks(content, document_id)
        
        # Generate embeddings and store
        await self.embed_and_store(chunks, document_name, document_id)
        
        # Store original file in MinIO
        await self.store_original_file(file_path, document_id)
        
        return {
            "document_id": document_id,
            "document_name": document_name,
            "chunks_created": len(chunks),
            "status": "success"
        }
    
    async def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Universal PDF processing with AI-powered page analysis
        NO HARDCODING - works with ANY PDF content
        """
        print(f"\n{'='*80}")
        print(f"📄 UNIVERSAL PDF PROCESSING")
        print(f"{'='*80}")
        print(f"Opening PDF: {file_path}")
        
        pdf_document = fitz.open(file_path)
        print(f"✅ PDF loaded: {len(pdf_document)} pages")
        
        content = {
            "pages": []
        }
        
        for page_num, page in enumerate(pdf_document, 1):
            print(f"\n{'─'*80}")
            print(f"📖 Processing Page {page_num}/{len(pdf_document)}")
            print(f"{'─'*80}")
            
            page_content = {
                "page_number": page_num,
                "text": None,
                "visual_analysis": None,
                "has_visuals": False
            }
            
            # Extract text
            print(f"  📝 Extracting text...")
            text = page.get_text()
            text_length = len(text.strip())
            
            if text.strip():
                page_content["text"] = text
                print(f"     ✅ Extracted {text_length} characters")
            else:
                print(f"     ⚠️  No text found")
            
            # Create high-res page snapshot
            print(f"  📸 Creating page snapshot...")
            try:
                mat = fitz.Matrix(2, 2)  # 2x resolution
                pix = page.get_pixmap(matrix=mat)
                print(f"     ✅ Snapshot: {pix.width}x{pix.height} pixels")
                
                img_bytes = pix.tobytes("png")
                print(f"     Size: {len(img_bytes)/1024:.2f} KB")
                
                # AI-POWERED VISUAL ANALYSIS (no hardcoding)
                print(f"  🔍 AI-powered visual analysis...")
                visual_analysis = await self.analyze_page_with_ai(
                    img_bytes, 
                    page_num,
                    has_text=(text_length > 0)
                )
                
                if visual_analysis and visual_analysis.get("has_visuals"):
                    page_content["visual_analysis"] = visual_analysis
                    page_content["has_visuals"] = True
                    print(f"     ✅ VISUALS DETECTED!")
                    print(f"     Types: {', '.join(visual_analysis.get('visual_types', []))}")
                else:
                    print(f"     ℹ️  No significant visuals")
                
                pix = None
                
            except Exception as e:
                print(f"     ❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
            
            content["pages"].append(page_content)
        
        pdf_document.close()
        
        print(f"\n{'='*80}")
        print(f"✅ PDF PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total pages: {len(content['pages'])}")
        print(f"Pages with text: {sum(1 for p in content['pages'] if p['text'])}")
        print(f"Pages with visuals: {sum(1 for p in content['pages'] if p['has_visuals'])}")
        print(f"{'='*80}\n")
        
        return content
    
    async def analyze_page_with_ai(
        self, 
        image_bytes: bytes, 
        page_num: int,
        has_text: bool
    ) -> Optional[Dict[str, Any]]:
        """
        🚀 TOP-TIER AI-POWERED VISUAL ANALYSIS - ZERO HARDCODING
        
        Uses GPT-4 Vision with advanced prompting to understand ANY visual content:
        - Medical diagrams, X-rays, anatomical charts
        - Legal flowcharts, organizational structures
        - Financial charts, graphs, tables
        - Technical diagrams, network topologies, architectures
        - Engineering blueprints, CAD drawings
        - Scientific plots, data visualizations
        - Any other visual content
        
        This is CRITICAL for accurate image generation later.
        """
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # ADVANCED PROMPT - Works for ANY document type
            analysis_prompt = """You are an expert visual analyst trained in ALL domains: medical, legal, technical, financial, scientific, engineering, business, etc.

Analyze this document page with EXTREME ATTENTION TO DETAIL:

🎯 PRIMARY TASK: Determine if this page contains ANY visual elements beyond plain text.

Visual elements include:
• Diagrams (network, flow, system, organizational, anatomical, etc.)
• Charts & Graphs (bar, line, pie, scatter, financial, scientific, etc.)
• Tables (data tables, comparison tables, specification tables, etc.)
• Images (photographs, illustrations, technical drawings, medical images, etc.)
• Maps (geographic, conceptual, process maps, etc.)
• Blueprints, schematics, circuit diagrams
• Infographics, timelines, hierarchies
• Screenshots, UI mockups, wireframes
• Mathematical notation, chemical structures, molecular diagrams
• Any other visual representation

📋 IF VISUALS ARE PRESENT, PROVIDE:

1. **Visual Classification** (be specific):
   - Type: [e.g., "network topology diagram", "financial bar chart", "anatomical illustration", "legal flowchart"]
   - Domain: [e.g., "IT networking", "finance", "medicine", "legal", "engineering"]
   
2. **Comprehensive Description** (for image reconstruction):
   - Overall layout and structure
   - Every component/element visible (shapes, symbols, nodes, etc.)
   - All text labels, annotations, titles, legends
   - Colors used (be specific: "dark blue", "red", "green")
   - Spatial relationships (what connects to what, positioning)
   - Size relationships (relative sizes of elements)
   - Direction indicators (arrows, flow direction)
   - Line styles (solid, dashed, dotted, thick, thin)
   - Patterns, textures, shading
   
3. **Key Elements** (critical for reconstruction):
   - List every named component/label
   - List all numerical values, measurements, data points
   - Identify key relationships or connections
   - Note any special symbols or icons
   
4. **Technical Context**:
   - What domain/field does this visual represent?
   - What is the purpose of this visual?
   - What information does it convey?
   
5. **Reconstruction Prompt** (CRITICAL):
   - Write a detailed prompt that an AI image generator could use to recreate this visual
   - Include ALL details: layout, components, colors, labels, relationships
   - Be extremely specific and comprehensive

📋 IF NO VISUALS:
Simply respond: "NO_VISUALS_FOUND"

🎯 QUALITY REQUIREMENTS:
- Be exhaustive - miss nothing
- Use precise language
- Include ALL text verbatim
- Describe spatial relationships clearly
- Your analysis will be used for image generation - it must be PERFECT

Respond in this structured format:

HAS_VISUALS: [YES/NO]
VISUAL_TYPES: [comma-separated list if YES]
DOMAIN: [domain/field if YES]
COMPREHENSIVE_DESCRIPTION:
[Extremely detailed description - minimum 200 words for complex visuals]

KEY_ELEMENTS:
• [Element 1]
• [Element 2]
• ... [all elements]

SPATIAL_LAYOUT:
[Describe how elements are arranged - top, bottom, left, right, center, etc.]

COLORS_AND_STYLES:
[All colors, line styles, patterns used]

TEXT_CONTENT:
[Every piece of text visible, including labels, legends, annotations]

RECONSTRUCTION_PROMPT:
[Detailed 300+ word prompt for image generation that captures EVERY detail]

YOUR ANALYSIS:"""
            
            # Call GPT-4 Vision with high detail
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",  # Latest GPT-4 with vision
                messages=[
                    {
                        "role": "system",
                        "content": "You are a world-class visual analyst expert in ALL domains. Your analysis must be exhaustive, precise, and detailed enough for perfect image reconstruction. Never miss any detail."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"  # Maximum detail
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,  # Allow long, detailed responses
                temperature=0.1  # Low temperature for consistency
            )
            
            response_text = response.choices[0].message.content
            
            # Parse response
            if "NO_VISUALS_FOUND" in response_text or "HAS_VISUALS: NO" in response_text:
                return {
                    "has_visuals": False,
                    "description": "This page contains only text content."
                }
            
            # Extract structured information (with robust error handling)
            parsed_data = self._parse_visual_analysis(response_text)
            
            # Ensure all required keys exist
            result = {
                "has_visuals": True,
                "visual_types": parsed_data.get("visual_types", ["general_visual"]),
                "domain": parsed_data.get("domain", "general"),
                "description": parsed_data.get("comprehensive_description", response_text[:500]),
                "key_elements": parsed_data.get("key_elements", []),
                "spatial_layout": parsed_data.get("spatial_layout", ""),
                "colors_and_styles": parsed_data.get("colors_and_styles", ""),
                "text_content": parsed_data.get("text_content", ""),
                "reconstruction_prompt": parsed_data.get("reconstruction_prompt", response_text),
                "raw_response": response_text
            }
            
            # Ensure description is not empty
            if not result["description"] or len(result["description"]) < 50:
                result["description"] = response_text[:500] if len(response_text) > 500 else response_text
            
            return result
            
        except Exception as e:
            print(f"     ❌ ERROR in AI visual analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_visual_analysis(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the structured response from AI visual analysis
        NO HARDCODING - parses AI-generated content
        ROBUST: Handles missing fields gracefully
        """
        parsed = {
            "visual_types": [],
            "domain": "general",
            "comprehensive_description": "",
            "key_elements": [],
            "spatial_layout": "",
            "colors_and_styles": "",
            "text_content": "",
            "reconstruction_prompt": ""
        }
        
        try:
            # Extract VISUAL_TYPES
            if "VISUAL_TYPES:" in response_text:
                types_line = response_text.split("VISUAL_TYPES:")[1].split("\n")[0]
                visual_types = [t.strip() for t in types_line.replace("[", "").replace("]", "").split(",") if t.strip()]
                if visual_types:
                    parsed["visual_types"] = visual_types
            
            # Extract DOMAIN
            if "DOMAIN:" in response_text:
                domain_line = response_text.split("DOMAIN:")[1].split("\n")[0]
                domain = domain_line.strip()
                if domain:
                    parsed["domain"] = domain
            
            # Extract COMPREHENSIVE_DESCRIPTION
            if "COMPREHENSIVE_DESCRIPTION:" in response_text:
                desc_start = response_text.find("COMPREHENSIVE_DESCRIPTION:") + len("COMPREHENSIVE_DESCRIPTION:")
                # Find next section or end of text
                desc_end = len(response_text)
                for next_section in ["KEY_ELEMENTS:", "SPATIAL_LAYOUT:", "COLORS_AND_STYLES:", "TEXT_CONTENT:", "RECONSTRUCTION_PROMPT:"]:
                    if next_section in response_text[desc_start:]:
                        desc_end = response_text.find(next_section, desc_start)
                        break
                
                description = response_text[desc_start:desc_end].strip()
                if description:
                    parsed["comprehensive_description"] = description
            
            # Extract KEY_ELEMENTS
            if "KEY_ELEMENTS:" in response_text:
                elem_start = response_text.find("KEY_ELEMENTS:") + len("KEY_ELEMENTS:")
                # Find next section
                elem_end = len(response_text)
                for next_section in ["SPATIAL_LAYOUT:", "COLORS_AND_STYLES:", "TEXT_CONTENT:", "RECONSTRUCTION_PROMPT:"]:
                    if next_section in response_text[elem_start:]:
                        elem_end = response_text.find(next_section, elem_start)
                        break
                
                elements_text = response_text[elem_start:elem_end].strip()
                key_elements = [e.strip("• -").strip() for e in elements_text.split("\n") if e.strip() and (e.strip().startswith("•") or e.strip().startswith("-"))]
                if key_elements:
                    parsed["key_elements"] = key_elements
            
            # Extract SPATIAL_LAYOUT
            if "SPATIAL_LAYOUT:" in response_text:
                layout_start = response_text.find("SPATIAL_LAYOUT:") + len("SPATIAL_LAYOUT:")
                layout_end = len(response_text)
                for next_section in ["COLORS_AND_STYLES:", "TEXT_CONTENT:", "RECONSTRUCTION_PROMPT:"]:
                    if next_section in response_text[layout_start:]:
                        layout_end = response_text.find(next_section, layout_start)
                        break
                
                layout = response_text[layout_start:layout_end].strip()
                if layout:
                    parsed["spatial_layout"] = layout
            
            # Extract COLORS_AND_STYLES
            if "COLORS_AND_STYLES:" in response_text:
                colors_start = response_text.find("COLORS_AND_STYLES:") + len("COLORS_AND_STYLES:")
                colors_end = len(response_text)
                for next_section in ["TEXT_CONTENT:", "RECONSTRUCTION_PROMPT:"]:
                    if next_section in response_text[colors_start:]:
                        colors_end = response_text.find(next_section, colors_start)
                        break
                
                colors = response_text[colors_start:colors_end].strip()
                if colors:
                    parsed["colors_and_styles"] = colors
            
            # Extract TEXT_CONTENT
            if "TEXT_CONTENT:" in response_text:
                text_start = response_text.find("TEXT_CONTENT:") + len("TEXT_CONTENT:")
                text_end = len(response_text)
                if "RECONSTRUCTION_PROMPT:" in response_text[text_start:]:
                    text_end = response_text.find("RECONSTRUCTION_PROMPT:", text_start)
                
                text_content = response_text[text_start:text_end].strip()
                if text_content:
                    parsed["text_content"] = text_content
            
            # Extract RECONSTRUCTION_PROMPT (MOST CRITICAL)
            if "RECONSTRUCTION_PROMPT:" in response_text:
                prompt_start = response_text.find("RECONSTRUCTION_PROMPT:") + len("RECONSTRUCTION_PROMPT:")
                reconstruction_prompt = response_text[prompt_start:].strip()
                if reconstruction_prompt:
                    parsed["reconstruction_prompt"] = reconstruction_prompt
            
            # Fallback: If no structured parsing worked, use the entire response as description
            if not parsed["comprehensive_description"] and not parsed["reconstruction_prompt"]:
                # Clean response might not have structured format, use as-is
                parsed["comprehensive_description"] = response_text[:500] if len(response_text) > 500 else response_text
                parsed["reconstruction_prompt"] = response_text
        
        except Exception as e:
            print(f"     ⚠️  Parsing error: {str(e)}, using fallback")
            # Absolute fallback
            parsed["comprehensive_description"] = response_text[:500] if len(response_text) > 500 else response_text
            parsed["reconstruction_prompt"] = response_text
        
        return parsed
    
    async def create_intelligent_chunks(self, content: Dict[str, Any], document_id: str) -> List[Dict]:
        """
        AI-POWERED CHUNKING - ZERO HARDCODING
        Uses AI to determine chunk importance and type
        """
        print(f"\n{'='*80}")
        print(f"🔪 AI-POWERED INTELLIGENT CHUNKING")
        print(f"{'='*80}")
        
        chunks = []
        
        for page_data in content.get("pages", []):
            page_num = page_data["page_number"]
            print(f"\n📄 Processing Page {page_num}")
            
            # Process text content
            if page_data.get("text"):
                text_content = page_data["text"]
                
                # AI-powered section classification (no keyword matching)
                section_type = await self._classify_section_with_ai(text_content)
                
                print(f"   🤖 AI Classification: {section_type}")
                
                # Chunk based on AI classification
                if section_type in ["specification", "technical_detail", "critical_data"]:
                    # Keep together as one chunk
                    chunk = {
                        "chunk_id": str(uuid.uuid4()),
                        "document_id": document_id,
                        "content": text_content,
                        "type": section_type,
                        "page": page_num,
                        "metadata": {
                            "token_count": len(self.tokenizer.encode(text_content)),
                            "section_type": section_type,
                            "ai_classified": True
                        }
                    }
                    chunks.append(chunk)
                    print(f"   ✅ Single chunk (type: {section_type})")
                else:
                    # Regular text chunking
                    text_chunks = self.chunk_text(text_content, page_num, document_id, section_type)
                    chunks.extend(text_chunks)
                    print(f"   ✅ {len(text_chunks)} text chunks")
            
            # Process visual content
            if page_data.get("has_visuals") and page_data.get("visual_analysis"):
                visual_analysis = page_data["visual_analysis"]
                
                print(f"   🖼️  VISUAL CONTENT")
                print(f"      Domain: {visual_analysis.get('domain', 'N/A')}")
                print(f"      Types: {', '.join(visual_analysis.get('visual_types', []))}")
                
                # Create comprehensive visual chunk
                # Get description - handle missing key gracefully
                visual_description = visual_analysis.get("description", "")
                if not visual_description:
                    visual_description = visual_analysis.get("raw_response", "Visual content detected")[:500]
                
                chunk = {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "content": visual_description,
                    "type": "visual",
                    "page": page_num,
                    "metadata": {
                        "visual_types": visual_analysis.get("visual_types", []),
                        "domain": visual_analysis.get("domain", ""),
                        "key_elements": visual_analysis.get("key_elements", []),
                        "spatial_layout": visual_analysis.get("spatial_layout", ""),
                        "colors_and_styles": visual_analysis.get("colors_and_styles", ""),
                        "text_content": visual_analysis.get("text_content", ""),
                        "reconstruction_prompt": visual_analysis.get("reconstruction_prompt", ""),
                        "has_text_on_page": bool(page_data.get("text"))
                    }
                }
                chunks.append(chunk)
                print(f"   ✅ Visual chunk created")
        
        print(f"\n{'='*80}")
        print(f"✅ INTELLIGENT CHUNKING COMPLETE")
        print(f"{'='*80}")
        print(f"Total chunks: {len(chunks)}")
        chunk_types = {}
        for chunk in chunks:
            chunk_type = chunk['type']
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        for chunk_type, count in chunk_types.items():
            print(f"  - {chunk_type}: {count}")
        print(f"{'='*80}\n")
        
        return chunks
    
    async def _classify_section_with_ai(self, text: str) -> str:
        """
        AI-POWERED SECTION CLASSIFICATION - NO KEYWORD MATCHING
        Works for ANY document type
        """
        # Sample text if too long
        text_sample = text[:1000] if len(text) > 1000 else text
        
        classification_prompt = f"""Classify this text section into ONE of these categories:

Categories:
- specification: Technical specifications, detailed product/component descriptions, requirements
- technical_detail: Implementation details, procedures, technical instructions
- critical_data: Tables, lists of important data, critical measurements
- summary: Executive summaries, overviews, abstracts
- narrative: Regular narrative text, explanations, discussions
- reference: Citations, references, appendices

Text sample:
{text_sample}

Respond with ONLY the category name, nothing else."""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Fast model for classification
                messages=[
                    {"role": "system", "content": "You are a text classifier. Respond with only the category name."},
                    {"role": "user", "content": classification_prompt}
                ],
                max_tokens=20,
                temperature=0
            )
            
            category = response.choices[0].message.content.strip().lower()
            return category if category in ["specification", "technical_detail", "critical_data", "summary", "narrative", "reference"] else "narrative"
            
        except:
            return "text"  # Fallback
    
    def chunk_text(self, text: str, page: int, document_id: str, section_type: str = "text") -> List[Dict]:
        """Universal text chunking with overlap"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), settings.CHUNK_SIZE - settings.CHUNK_OVERLAP):
            chunk_tokens = tokens[i:i + settings.CHUNK_SIZE]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "document_id": document_id,
                "content": chunk_text,
                "type": section_type,
                "page": page,
                "metadata": {
                    "token_count": len(chunk_tokens),
                    "position": i
                }
            })
        
        return chunks
    
    async def embed_and_store(self, chunks: List[Dict], document_name: str, document_id: str):
        """Generate embeddings and store in Qdrant"""
        print(f"\n{'='*80}")
        print(f"🔢 GENERATING EMBEDDINGS AND STORING")
        print(f"{'='*80}")
        
        for i in range(0, len(chunks), settings.EMBEDDING_BATCH_SIZE):
            batch = chunks[i:i + settings.EMBEDDING_BATCH_SIZE]
            texts = [chunk["content"] for chunk in batch]
            
            print(f"Processing batch {i//settings.EMBEDDING_BATCH_SIZE + 1}")
            
            # Generate embeddings
            response = await self.openai_client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
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
            
            print(f"   ✅ Stored {len(points)} vectors")
        
        print(f"{'='*80}\n")
    
    def generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID"""
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]
    
    async def store_original_file(self, file_path: str, document_id: str):
        """Store original file in MinIO"""
        bucket_name = "documents"
        
        if not self.minio_client.bucket_exists(bucket_name):
            self.minio_client.make_bucket(bucket_name)
        
        self.minio_client.fput_object(
            bucket_name,
            f"{document_id}/original",
            file_path
        )
    
    async def process_docx(self, file_path: str) -> Dict[str, Any]:
        """Process DOCX files (to be implemented)"""
        # TODO: Implement DOCX processing
        raise NotImplementedError("DOCX processing not yet implemented")
    
    async def process_image(self, file_path: str) -> Dict[str, Any]:
        """Process standalone images (to be implemented)"""
        # TODO: Implement image processing
        raise NotImplementedError("Image processing not yet implemented")