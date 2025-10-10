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
            
            # Extract images - COMMENT THIS OUT FOR NOW!
            print(f"  - Skipping images for faster processing")
            # image_list = page.get_images()
            # print(f"  - Found {len(image_list)} images")
            # for img_index, img in enumerate(image_list):
            #     print(f"    - Processing image {img_index + 1}/{len(image_list)}...")
            #     xref = img[0]
            #     pix = fitz.Pixmap(pdf_document, xref)
            #     if pix.n - pix.alpha < 4:
            #         image_analysis = await self.analyze_image_with_vision(pix.tobytes())
            #         content["images"].append({
            #             "page": page_num,
            #             "index": img_index,
            #             "analysis": image_analysis,
            #             "type": "image"
            #         })
            #     pix = None
        
        print("PDF processing complete!")
        pdf_document.close()
        return content
    
    
    async def analyze_image_with_vision(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze image using GPT-4 Vision"""
        import base64
        
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
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
        
        return {
            "description": response.choices[0].message.content,
            "type": self.classify_image_type(response.choices[0].message.content)
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
        chunks = []
        chunk_id = 0
        
        # Process text chunks
        for text_item in content.get("text", []):
            text_chunks = self.chunk_text(
                text_item["content"],
                text_item["page"],
                chunk_id,
                document_id
            )
            chunks.extend(text_chunks)
            chunk_id += len(text_chunks)
        
        # Process image descriptions
        for image_item in content.get("images", []):
            chunk = {
                "chunk_id": str(uuid.uuid4()),  # ← CHANGED: Use UUID instead
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
            chunk_id += 1
        
        return chunks
    
    def chunk_text(self, text: str, page: int, start_chunk_id: int, document_id: str) -> List[Dict]:
        """Chunk text with overlap"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), settings.CHUNK_SIZE - settings.CHUNK_OVERLAP):
            chunk_tokens = tokens[i:i + settings.CHUNK_SIZE]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                "chunk_id": str(uuid.uuid4()),  # ← CHANGED: Use UUID instead
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