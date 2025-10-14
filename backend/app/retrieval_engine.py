# retrieval_engine.py - Enhanced for multi-document handling
from typing import List, Dict, Any, Optional
import numpy as np
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, SearchParams
from sentence_transformers import CrossEncoder
import redis
import json
import hashlib
from app.config import settings

class HybridRetriever:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.qdrant_client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB
        )
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 10,
        document_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Enhanced retrieval with multi-document handling"""
        
        # Check cache
        cache_key = self._get_cache_key(query, top_k, document_filter)
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Generate embedding
        query_embedding = await self._get_embedding(query)
        
        # Build filter
        search_filter = None
        if document_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match={"any": document_filter}
                    )
                ]
            )
        
        # Dense retrieval
        search_results = await self.qdrant_client.search(
            collection_name="documents",
            query_vector=query_embedding,
            limit=top_k * 5,
            query_filter=search_filter
        )
        
        # Extract candidates
        candidates = []
        for result in search_results:
            candidates.append({
                "id": result.id,
                "text": result.payload["text"],
                "score": result.score,
                "metadata": result.payload,
                "page": result.payload.get("page", 0),
                "type": result.payload.get("type", "text"),
                "document_name": result.payload.get("document_name", ""),
                "document_id": result.payload.get("document_id", "")
            })
        
        unique_docs = len(set(c['document_name'] for c in candidates))
        print(f"\n=== Retrieved {len(candidates)} candidates from {unique_docs} documents ===")
        
        # GROUP BY DOCUMENT AND PAGE
        doc_page_groups = {}
        for candidate in candidates:
            doc_id = candidate['document_id']
            page = candidate['page']
            doc_page_key = f"{doc_id}_Page{page}"
            
            if doc_page_key not in doc_page_groups:
                doc_page_groups[doc_page_key] = {
                    'document_id': doc_id,
                    'document_name': candidate['document_name'],
                    'page': page,
                    'chunks': []
                }
            
            doc_page_groups[doc_page_key]['chunks'].append(candidate)
        
        print(f"Grouped into {len(doc_page_groups)} document-page combinations")
        
        # Calculate page scores with document diversity
        page_scores = {}
        
        for doc_page_key, page_data in doc_page_groups.items():
            chunks = page_data['chunks']
            
            # Separate by type
            text_chunks = [c for c in chunks if c['type'] in ['text', 'hardware_spec']]
            visual_chunks = [c for c in chunks if c['type'] == 'visual']
            hardware_chunks = [c for c in chunks if c['type'] == 'hardware_spec']
            
            # Base score
            avg_score = sum(c['score'] for c in chunks) / len(chunks)
            
            # Boosts
            diversity_boost = 0.15 if (text_chunks and visual_chunks) else 0
            multi_chunk_boost = min(len(chunks) * 0.05, 0.2)
            hardware_boost = 0.1 if hardware_chunks else 0
            
            visual_boost = 0.0
            if visual_chunks:
                for vc in visual_chunks:
                    if vc['metadata'].get('metadata', {}).get('key_elements', []):
                        visual_boost = 0.1
                        break
            
            final_score = avg_score + diversity_boost + multi_chunk_boost + hardware_boost + visual_boost
            
            page_scores[doc_page_key] = {
                'base_score': avg_score,
                'final_score': final_score,
                'chunks': chunks,
                'document_id': page_data['document_id'],
                'document_name': page_data['document_name'],
                'page': page_data['page']
            }
        
        # Sort pages
        sorted_pages = sorted(
            page_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )
        
        # Print distribution
        print(f"\n=== Document Distribution in Top Results ===")
        doc_count = {}
        for _, page_data in sorted_pages[:10]:
            doc_name = page_data['document_name']
            doc_count[doc_name] = doc_count.get(doc_name, 0) + 1
        
        for doc_name, count in doc_count.items():
            print(f"  📄 {doc_name}: {count} pages")
        
        # Flatten to chunks
        reordered_candidates = []
        for doc_page_key, page_data in sorted_pages:
            page_chunks = sorted(
                page_data['chunks'],
                key=lambda x: (
                    0 if x['type'] == 'hardware_spec' else (1 if x['type'] == 'text' else 2),
                    -x['score']
                )
            )
            reordered_candidates.extend(page_chunks)
        
        # Rerank
        if reordered_candidates:
            candidates_to_rerank = reordered_candidates[:top_k * 3]
            texts = [c["text"] for c in candidates_to_rerank]
            
            print(f"\n=== Reranking {len(candidates_to_rerank)} candidates ===")
            
            rerank_scores = self.reranker.predict(
                [(query, text) for text in texts]
            )
            
            for i, candidate in enumerate(candidates_to_rerank):
                candidate["rerank_score"] = float(rerank_scores[i])
                candidate["final_score"] = (
                    0.3 * candidate["score"] + 
                    0.7 * candidate["rerank_score"]
                )
            
            candidates_to_rerank.sort(key=lambda x: x["final_score"], reverse=True)
            final_candidates = candidates_to_rerank[:top_k]
        else:
            final_candidates = []
        
        # Print final results
        print(f"\n=== Final Top {len(final_candidates)} Results ===")
        current_doc = None
        for i, candidate in enumerate(final_candidates, 1):
            doc_name = candidate['document_name']
            if doc_name != current_doc:
                print(f"\n📄 {doc_name}")
                current_doc = doc_name
            
            type_emoji = "📝" if candidate['type'] == "text" else ("🔧" if candidate['type'] == "hardware_spec" else "🖼️")
            print(f"  {i}. {type_emoji} Page {candidate['page']} - Score: {candidate['final_score']:.4f}")
        
        # Cache
        self.redis_client.setex(cache_key, 3600, json.dumps(final_candidates))
        
        return final_candidates
    
    async def delete_document(self, document_id: str):
        """Delete all chunks for a document"""
        await self.qdrant_client.delete(
            collection_name="documents",
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match={"value": document_id}
                    )
                ]
            )
        )
        print(f"✅ Deleted all chunks for document {document_id}")
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding"""
        response = await self.openai_client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    def _get_cache_key(self, query: str, top_k: int, document_filter: Optional[List[str]]) -> str:
        """Generate cache key"""
        key_parts = [query, str(top_k)]
        if document_filter:
            key_parts.extend(sorted(document_filter))
        key_string = "_".join(key_parts)
        return f"query_cache:{hashlib.md5(key_string.encode()).hexdigest()}"