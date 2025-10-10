# retrieval_engine.py
from typing import List, Dict, Any, Optional
import numpy as np
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, SearchParams
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
        """Advanced hybrid retrieval with reranking"""
        
        # Check cache first
        cache_key = self._get_cache_key(query, top_k, document_filter)
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Generate query embedding
        query_embedding = await self._get_embedding(query)
        
        # Build filter if document IDs provided
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
            limit=top_k * 3,  # Get more for reranking
            query_filter=search_filter
        )
        
        # Extract texts and metadata
        candidates = []
        for result in search_results:
            candidates.append({
                "id": result.id,
                "text": result.payload["text"],
                "score": result.score,
                "metadata": result.payload
            })
        
        # Rerank with cross-encoder
        if candidates:
            texts = [c["text"] for c in candidates]
            rerank_scores = self.reranker.predict(
                [(query, text) for text in texts]
            )
            
            # Combine scores
            for i, candidate in enumerate(candidates):
                candidate["rerank_score"] = float(rerank_scores[i])
                candidate["final_score"] = (
                    0.3 * candidate["score"] + 
                    0.7 * candidate["rerank_score"]
                )
            
            # Sort by final score
            candidates.sort(key=lambda x: x["final_score"], reverse=True)
            candidates = candidates[:top_k]
        
        # Cache results
        self.redis_client.setex(
            cache_key,
            3600,  # 1 hour TTL
            json.dumps(candidates)
        )
        
        return candidates
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        response = await self.openai_client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    def _get_cache_key(self, query: str, top_k: int, document_filter: Optional[List[str]]) -> str:
        """Generate cache key for query"""
        key_parts = [query, str(top_k)]
        if document_filter:
            key_parts.extend(sorted(document_filter))
        key_string = "_".join(key_parts)
        return f"query_cache:{hashlib.md5(key_string.encode()).hexdigest()}"
