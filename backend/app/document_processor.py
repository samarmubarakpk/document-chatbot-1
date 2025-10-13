# document_processor.py - Enhanced with Full Page Visual Analysis
import hashlib
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import fitz  # PyMuPDF
from docx import Document
from PIL import Image
import cv2
import numpy as np
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import tiktoken
from minio import Minio
import json
import uuid
import base64
import io

from app.config import settings

class DocumentProcessor:
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
        """Initialize Qdrant collections"""
        collections = {
            "documents": {
                "size": 3072,  # text-embedding-3-large dimension
                "distance": Distance.COSINE
            }
        }
        
        for collection_name, config in collections.items():
            await self.qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=config["size"],
                    distance=config["distance"]
                )
            )
    
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
        
        # Create smart chunks
        chunks = await self.create_smart_chunks(content, document_id)
        
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
        NEW APPROACH: Analyze full page snapshots instead of extracting individual images
        This captures ALL visuals including tables, graphs, and layout-based diagrams
        """
        print(f"\n{'='*80}")
        print(f"📄 PROCESSING PDF WITH PAGE SNAPSHOT ANALYSIS")
        print(f"{'='*80}")
        print(f"Opening PDF: {file_path}")
        
        pdf_document = fitz.open(file_path)
        print(f"✅ PDF loaded: {len(pdf_document)} pages")
        
        content = {
            "pages": []  # Store page-by-page analysis
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
            
            # Step 1: Extract text content
            print(f"  📝 Extracting text...")
            text = page.get_text()
            text_length = len(text.strip())
            
            if text.strip():
                page_content["text"] = text
                print(f"     ✅ Extracted {text_length} characters")
                print(f"     Preview: {text.strip()[:100]}...")
            else:
                print(f"     ⚠️  No text found on this page")
            
            # Step 2: Create page snapshot and analyze with Vision API
            print(f"  📸 Creating page snapshot...")
            try:
                # Convert page to high-resolution image
                # zoom=2 means 2x resolution (144 DPI instead of 72 DPI)
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                
                print(f"     ✅ Snapshot created: {pix.width}x{pix.height} pixels")
                
                # Convert to bytes
                img_bytes = pix.tobytes("png")
                print(f"     Size: {len(img_bytes)/1024:.2f} KB")
                
                # Analyze the entire page with Vision API
                print(f"  🔍 Analyzing page with GPT-4 Vision...")
                visual_analysis = await self.analyze_page_snapshot(
                    img_bytes, 
                    page_num,
                    has_text=(text_length > 0)
                )
                
                if visual_analysis and visual_analysis.get("has_visuals"):
                    page_content["visual_analysis"] = visual_analysis
                    page_content["has_visuals"] = True
                    print(f"     ✅ VISUALS DETECTED!")
                    print(f"     Types found: {', '.join(visual_analysis.get('visual_types', []))}")
                    print(f"     Description: {visual_analysis['description'][:150]}...")
                else:
                    print(f"     ℹ️  No significant visuals detected")
                
                pix = None
                
            except Exception as e:
                print(f"     ❌ Error analyzing page snapshot: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Add page to content
            content["pages"].append(page_content)
        
        pdf_document.close()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"✅ PDF PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total pages: {len(content['pages'])}")
        print(f"Pages with text: {sum(1 for p in content['pages'] if p['text'])}")
        print(f"Pages with visuals: {sum(1 for p in content['pages'] if p['has_visuals'])}")
        print(f"{'='*80}\n")
        
        return content
    
    async def analyze_page_snapshot(
        self, 
        image_bytes: bytes, 
        page_num: int,
        has_text: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a full page snapshot to detect and describe visuals
        
        Returns:
            {
                "has_visuals": True/False,
                "visual_types": ["diagram", "table", "chart"],
                "description": "Detailed description...",
                "key_elements": ["element1", "element2"]
            }
        """
        try:
            # Convert to base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create specialized prompt for page analysis
            analysis_prompt = """Analyze this document page carefully and determine:

1. **Does this page contain ANY visual elements?** (diagrams, charts, graphs, tables, flowcharts, network diagrams, architecture diagrams, screenshots, illustrations, etc.)

2. **If YES, provide:**
   - List of visual types present (e.g., "network diagram", "table", "flowchart")
   - Detailed description of each visual element
   - Any text, labels, or annotations visible in the visuals
   - Relationships or connections shown in diagrams
   - Data or information presented in tables/charts
   - Technical specifications or model numbers visible

3. **If NO visuals**, simply respond: "NO_VISUALS_FOUND"

**Important:** 
- Look for ALL types of visuals, including tables, diagrams, charts, graphs, screenshots
- Even simple diagrams or small icons count as visuals
- Technical diagrams often have text labels - include those in your description
- If you see hardware specifications, network topologies, or system architectures, describe them in detail

Respond in this format:
HAS_VISUALS: [YES/NO]
VISUAL_TYPES: [list types if YES]
DESCRIPTION: [detailed description if YES]
KEY_ELEMENTS: [important elements, model numbers, labels if YES]"""
            
            # Call Vision API
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing technical documents and identifying visual elements like diagrams, charts, tables, and illustrations."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content
            
            # Parse response
            if "NO_VISUALS_FOUND" in response_text or "HAS_VISUALS: NO" in response_text:
                return {
                    "has_visuals": False,
                    "description": "This page contains only text content with no visual elements."
                }
            
            # Extract information from structured response
            visual_types = []
            description = ""
            key_elements = []
            
            # Parse VISUAL_TYPES
            if "VISUAL_TYPES:" in response_text:
                types_line = response_text.split("VISUAL_TYPES:")[1].split("\n")[0]
                visual_types = [t.strip() for t in types_line.replace("[", "").replace("]", "").split(",") if t.strip()]
            
            # Parse DESCRIPTION
            if "DESCRIPTION:" in response_text:
                desc_start = response_text.find("DESCRIPTION:") + len("DESCRIPTION:")
                desc_end = response_text.find("KEY_ELEMENTS:") if "KEY_ELEMENTS:" in response_text else len(response_text)
                description = response_text[desc_start:desc_end].strip()
            
            # Parse KEY_ELEMENTS
            if "KEY_ELEMENTS:" in response_text:
                elements_line = response_text.split("KEY_ELEMENTS:")[1].strip()
                key_elements = [e.strip() for e in elements_line.replace("[", "").replace("]", "").split(",") if e.strip()]
            
            # Fallback: if structured parsing fails, use the whole response
            if not description:
                description = response_text
            
            # Classify visual types if not provided
            if not visual_types:
                visual_types = self._classify_visual_types(description)
            
            return {
                "has_visuals": True,
                "visual_types": visual_types,
                "description": description,
                "key_elements": key_elements,
                "raw_response": response_text
            }
            
        except Exception as e:
            print(f"     ❌ ERROR in page snapshot analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _classify_visual_types(self, description: str) -> List[str]:
        """Classify visual types based on description"""
        description_lower = description.lower()
        types = []
        
        type_keywords = {
            "network_diagram": ["network", "topology", "router", "switch", "cisco"],
            "flowchart": ["flowchart", "flow diagram", "process flow", "workflow"],
            "architecture_diagram": ["architecture", "system diagram", "infrastructure", "design"],
            "table": ["table", "grid", "matrix", "spreadsheet"],
            "chart": ["chart", "graph", "plot", "bar chart", "pie chart"],
            "graph": ["graph", "line graph", "scatter plot"],
            "screenshot": ["screenshot", "screen capture", "interface"],
            "technical_diagram": ["technical diagram", "schematic", "blueprint"],
            "illustration": ["illustration", "drawing", "figure"]
        }
        
        for visual_type, keywords in type_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                types.append(visual_type)
        
        if not types:
            types.append("general_visual")
        
        return types
    
    async def create_smart_chunks(self, content: Dict[str, Any], document_id: str) -> List[Dict]:
        """Create intelligent chunks from page content"""
        print(f"\n{'='*80}")
        print(f"🔪 CREATING SMART CHUNKS")
        print(f"{'='*80}")
        
        chunks = []
        chunk_id = 0
        
        # Process each page
        for page_data in content.get("pages", []):
            page_num = page_data["page_number"]
            print(f"\n📄 Processing Page {page_num}")
            
            # Process text content
            if page_data.get("text"):
                text_content = page_data["text"]
                
                # Check if hardware section
                if self._is_hardware_section(text_content):
                    print(f"   ⚡ HARDWARE SECTION - keeping together")
                    chunk = {
                        "chunk_id": str(uuid.uuid4()),
                        "document_id": document_id,
                        "content": text_content,
                        "type": "hardware_spec",
                        "page": page_num,
                        "metadata": {
                            "token_count": len(self.tokenizer.encode(text_content)),
                            "section_type": "hardware",
                            "contains_specs": True,
                            "model_numbers": self._extract_model_numbers(text_content)
                        }
                    }
                    chunks.append(chunk)
                    print(f"   ✅ Hardware chunk created with {len(chunk['metadata']['model_numbers'])} models")
                else:
                    # Regular text chunking
                    text_chunks = self.chunk_text(text_content, page_num, chunk_id, document_id)
                    chunks.extend(text_chunks)
                    print(f"   ✅ Created {len(text_chunks)} text chunks")
                    chunk_id += len(text_chunks)
            
            # Process visual analysis
            if page_data.get("has_visuals") and page_data.get("visual_analysis"):
                visual_analysis = page_data["visual_analysis"]
                
                print(f"   🖼️  VISUAL CONTENT DETECTED")
                print(f"      Types: {', '.join(visual_analysis.get('visual_types', []))}")
                
                # Create chunk for visual content
                chunk = {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "content": visual_analysis["description"],
                    "type": "visual",
                    "page": page_num,
                    "metadata": {
                        "visual_types": visual_analysis.get("visual_types", []),
                        "key_elements": visual_analysis.get("key_elements", []),
                        "has_text_on_page": bool(page_data.get("text"))
                    }
                }
                chunks.append(chunk)
                print(f"   ✅ Visual chunk created")
                chunk_id += 1
        
        print(f"\n{'='*80}")
        print(f"✅ CHUNKING COMPLETE")
        print(f"{'='*80}")
        print(f"Total chunks: {len(chunks)}")
        print(f"  - Text chunks: {len([c for c in chunks if c['type'] == 'text'])}")
        print(f"  - Hardware spec chunks: {len([c for c in chunks if c['type'] == 'hardware_spec'])}")
        print(f"  - Visual chunks: {len([c for c in chunks if c['type'] == 'visual'])}")
        print(f"{'='*80}\n")
        
        return chunks
    
    def _is_hardware_section(self, text: str) -> bool:
        """Detect if text is a hardware specification section"""
        text_lower = text.lower()
        
        indicators = {
            'hardware': 'hardware' in text_lower,
            'cisco': 'cisco' in text_lower,
            'catalyst': 'catalyst' in text_lower,
            'switch': 'switch' in text_lower or 'switches' in text_lower,
            'router': 'router' in text_lower or 'routers' in text_lower,
            'model': 'model' in text_lower,
            'chassis': 'chassis' in text_lower,
            'supervisor': 'supervisor' in text_lower,
            'port': 'port' in text_lower or 'ports' in text_lower,
            'gbps': 'gbps' in text_lower or 'gigabit' in text_lower,
            'recommended': 'recommended' in text_lower,
            'specification': 'specification' in text_lower or 'specs' in text_lower
        }
        
        match_count = sum(indicators.values())
        return match_count >= 4
    
    def _extract_model_numbers(self, text: str) -> List[str]:
        """Extract Cisco model numbers from text"""
        import re
        
        models = []
        
        # Pattern 1: Catalyst followed by model number
        pattern1 = r'Catalyst\s+(\d{4}[A-Z]?(?:-\d+[A-Z]+)?)'
        matches1 = re.findall(pattern1, text, re.IGNORECASE)
        models.extend([f"Catalyst {m}" for m in matches1])
        
        # Pattern 2: Cisco followed by Catalyst and model
        pattern2 = r'Cisco\s+Catalyst\s+(\d{4}[A-Z]?(?:-\d+[A-Z]+)?)'
        matches2 = re.findall(pattern2, text, re.IGNORECASE)
        models.extend([f"Cisco Catalyst {m}" for m in matches2])
        
        # Pattern 3: Standalone model numbers
        pattern3 = r'\b(\d{4}[A-Z]?(?:-\d+[A-Z]+)?)\b'
        matches3 = re.findall(pattern3, text)
        models.extend([m for m in matches3 if m[0] in '9321' and len(m) >= 4])
        
        # Remove duplicates
        unique_models = []
        seen = set()
        for model in models:
            model_clean = model.strip()
            if model_clean and model_clean not in seen:
                unique_models.append(model_clean)
                seen.add(model_clean)
        
        return unique_models
    
    def chunk_text(self, text: str, page: int, start_chunk_id: int, document_id: str) -> List[Dict]:
        """Chunk text with overlap"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), settings.CHUNK_SIZE - settings.CHUNK_OVERLAP):
            chunk_tokens = tokens[i:i + settings.CHUNK_SIZE]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "document_id": document_id,
                "content": chunk_text,
                "type": "text",
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
        
        # Batch process embeddings
        for i in range(0, len(chunks), settings.EMBEDDING_BATCH_SIZE):
            batch = chunks[i:i + settings.EMBEDDING_BATCH_SIZE]
            texts = [chunk["content"] for chunk in batch]
            
            print(f"Processing batch {i//settings.EMBEDDING_BATCH_SIZE + 1}/{(len(chunks)-1)//settings.EMBEDDING_BATCH_SIZE + 1}")
            
            # Generate embeddings
            response = await self.openai_client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=texts
            )
            
            # Prepare points for Qdrant
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
            
            print(f"   ✅ Stored {len(points)} vectors in Qdrant")
        
        print(f"{'='*80}\n")
    
    def generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID"""
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]
    
    async def store_original_file(self, file_path: str, document_id: str):
        """Store original file in MinIO"""
        bucket_name = "documents"
        
        # Create bucket if not exists
        if not self.minio_client.bucket_exists(bucket_name):
            self.minio_client.make_bucket(bucket_name)
        
        # Upload file
        self.minio_client.fput_object(
            bucket_name,
            f"{document_id}/original",
            file_path
        )