# document_processor.py
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

from app.config import settings  # Import the instance, not the class!

class DocumentProcessor:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)  # ← lowercase!
        self.qdrant_client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,      # ← lowercase!
            port=settings.QDRANT_PORT       # ← lowercase!
        )
        self.minio_client = Minio(
            settings.MINIO_ENDPOINT,        # ← lowercase!
            access_key=settings.MINIO_ACCESS_KEY,     # ← lowercase!
            secret_key=settings.MINIO_SECRET_KEY,     # ← lowercase!
            secure=settings.MINIO_SECURE    # ← lowercase!
        )
        self.tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL)  # ← lowercase!
        
    async def initialize_collections(self):
        """Initialize Qdrant collections"""
        collections = {
            "documents": {
                "size": 3072,  # text-embedding-3-large dimension
                "distance": Distance.COSINE
            },
            "images": {
                "size": 1536,  # vision embedding dimension
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
        """Extract text and images from PDF"""
        print(f"Opening PDF: {file_path}")
        pdf_document = fitz.open(file_path)
        print(f"PDF has {len(pdf_document)} pages")
        
        content = {
            "text": [],
            "images": [],
            "diagrams": [],
            "tables": []
        }
        
        for page_num, page in enumerate(pdf_document, 1):
            print(f"Processing page {page_num}/{len(pdf_document)}...")
            
            # Extract text
            text = page.get_text()
            if text.strip():
                print(f"  - Extracted {len(text)} characters of text")
                content["text"].append({
                    "page": page_num,
                    "content": text,
                    "type": "text"
                })
            
            # Extract images
            image_list = page.get_images()
            print(f"  - Found {len(image_list)} images")

            for img_index, img in enumerate(image_list):
                print(f"    - Processing image {img_index + 1}/{len(image_list)}...")
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        print(f"      → Valid image format (colorspace: {pix.n})")
                        # Analyze image with GPT-4 Vision
                        image_analysis = await self.analyze_image_with_vision(pix.tobytes())
                        print(f"      → Analysis type: {image_analysis['type']}")
                        
                        content["images"].append({
                            "page": page_num,
                            "index": img_index,
                            "analysis": image_analysis,
                            "type": "image"
                        })
                        print(f"      → ✅ Image added to content")
                    else:
                        print(f"      → Skipping (unsupported colorspace: {pix.n})")
                    
                    pix = None
                    
                except Exception as e:
                    print(f"      → ❌ Error processing image: {str(e)}")
                    continue
        
        # CRITICAL: Return the content!
        print("PDF processing complete!")
        pdf_document.close()
        return content  # ← ADD THIS LINE!
    
    async def analyze_image_with_vision(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze image using GPT-4 Vision"""
        import base64
        
        try:
            print(f"      → Encoding image (size: {len(image_bytes)} bytes)...")
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            print(f"      → Base64 size: {len(base64_image)} chars")
            
            print(f"      → Calling GPT-4o Vision API...")
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are analyzing technical diagrams and images from documents. Describe what you see, identify any diagrams, flowcharts, network diagrams, or technical illustrations. Extract all text and relationships visible."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and describe its contents, especially if it's a technical diagram."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            description = response.choices[0].message.content
            print(f"      → ✅ Got vision response: {description[:100]}...")
            
            return {
                "description": description,
                "type": self.classify_image_type(description)
            }
            
        except Exception as e:
            print(f"      → ❌ ERROR in vision analysis: {str(e)}")
            # Return a fallback instead of failing
            return {
                "description": f"[Image could not be analyzed: {str(e)}]",
                "type": "general_image"
            }
    
    def classify_image_type(self, description: str) -> str:
        """Classify image type based on description"""
        description_lower = description.lower()
        if any(term in description_lower for term in ["flowchart", "flow diagram", "process flow"]):
            return "flowchart"
        elif any(term in description_lower for term in ["network", "architecture", "system diagram"]):
            return "architecture_diagram"
        elif any(term in description_lower for term in ["table", "grid", "matrix"]):
            return "table"
        elif any(term in description_lower for term in ["chart", "graph", "plot"]):
            return "chart"
        else:
            return "general_image"
    
    async def create_smart_chunks(self, content: Dict[str, Any], document_id: str) -> List[Dict]:
        """Create intelligent chunks with context preservation"""
        print(f"\n=== Creating Smart Chunks ===")
        chunks = []
        chunk_id = 0
        
        # Process text chunks
        text_items = content.get("text", [])
        print(f"Processing {len(text_items)} text items...")
        
        for text_item in text_items:
            page = text_item["page"]
            text_content = text_item["content"]
            
            print(f"  → Processing page {page}...")
            
            # ENHANCED: Check if this is a hardware specification section
            if self._is_hardware_section(text_content):
                print(f"    ⚡ HARDWARE SECTION DETECTED - keeping together")
                # Keep hardware sections together without splitting
                chunk = {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "content": text_content,
                    "type": "hardware_spec",  # Special type for hardware specs
                    "page": page,
                    "metadata": {
                        "token_count": len(self.tokenizer.encode(text_content)),
                        "section_type": "hardware",
                        "contains_specs": True,
                        "model_numbers": self._extract_model_numbers(text_content)
                    }
                }
                chunks.append(chunk)
                print(f"    ✅ Created hardware spec chunk with models: {chunk['metadata']['model_numbers']}")
                chunk_id += 1
            else:
                # Use standard chunking for other text
                text_chunks = self.chunk_text(
                    text_content,
                    page,
                    chunk_id,
                    document_id
                )
                print(f"    ✅ Created {len(text_chunks)} standard text chunks from page {page}")
                chunks.extend(text_chunks)
                chunk_id += len(text_chunks)
        
        print(f"\nText chunks total: {len(chunks)}")
        
        # Process image descriptions
        print(f"\n--- Processing Image Chunks ---")
        image_items = content.get("images", [])
        print(f"Found {len(image_items)} images to process")
        
        for image_item in image_items:
            print(f"  → Creating chunk for image on page {image_item['page']}...")
            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "document_id": document_id,
                "content": image_item["analysis"]["description"],
                "type": "image",
                "page": image_item["page"],
                "metadata": {
                    "image_type": image_item["analysis"]["type"],
                    "image_index": image_item["index"]
                }
            }
            chunks.append(chunk)
            print(f"    ✅ Image chunk created (content length: {len(image_item['analysis']['description'])} chars)")
            chunk_id += 1
        
        print(f"\n✅ Total chunks created: {len(chunks)}")
        print(f"   - Text chunks: {len([c for c in chunks if c['type'] == 'text'])}")
        print(f"   - Hardware spec chunks: {len([c for c in chunks if c['type'] == 'hardware_spec'])}")
        print(f"   - Image chunks: {len([c for c in chunks if c['type'] == 'image'])}")
        
        return chunks

    def _is_hardware_section(self, text: str) -> bool:
        """Detect if text is a hardware specification section"""
        text_lower = text.lower()
        
        # Count hardware-related indicators
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
        
        # If 4 or more indicators are present, it's likely a hardware section
        match_count = sum(indicators.values())
        
        if match_count >= 4:
            print(f"    🎯 Hardware section detected ({match_count}/12 indicators)")
            return True
        
        return False

    def _extract_model_numbers(self, text: str) -> List[str]:
        """Extract Cisco model numbers from text"""
        import re
        
        models = []
        
        # Pattern 1: Catalyst followed by model number (e.g., "Catalyst 9407R", "Catalyst 9300-48P")
        pattern1 = r'Catalyst\s+(\d{4}[A-Z]?(?:-\d+[A-Z]+)?)'
        matches1 = re.findall(pattern1, text, re.IGNORECASE)
        models.extend([f"Catalyst {m}" for m in matches1])
        
        # Pattern 2: Cisco followed by Catalyst and model
        pattern2 = r'Cisco\s+Catalyst\s+(\d{4}[A-Z]?(?:-\d+[A-Z]+)?)'
        matches2 = re.findall(pattern2, text, re.IGNORECASE)
        models.extend([f"Cisco Catalyst {m}" for m in matches2])
        
        # Pattern 3: Standalone model numbers (e.g., "9407R", "9300-48P")
        pattern3 = r'\b(\d{4}[A-Z]?(?:-\d+[A-Z]+)?)\b'
        matches3 = re.findall(pattern3, text)
        # Filter to likely Cisco models (4 digits starting with 9, 3, 2, or 1)
        models.extend([m for m in matches3 if m[0] in '9321' and len(m) >= 4])
        
        # Remove duplicates while preserving order
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
        # Batch process embeddings
        for i in range(0, len(chunks), settings.EMBEDDING_BATCH_SIZE):
            batch = chunks[i:i + settings.EMBEDDING_BATCH_SIZE]
            texts = [chunk["content"] for chunk in batch]
            
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