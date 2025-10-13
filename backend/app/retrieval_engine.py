# retrieval_engine.py - Updated to handle new "visual" chunk type
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
        """Advanced hybrid retrieval with page-level visual boosting"""
        
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
        
        # Dense retrieval - get MORE results for better reranking
        search_results = await self.qdrant_client.search(
            collection_name="documents",
            query_vector=query_embedding,
            limit=top_k * 5,  # Get 5x more for better page grouping
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
                "document_name": result.payload.get("document_name", "")
            })
        
        print(f"\n=== Retrieved {len(candidates)} initial candidates ===")
        
        # GROUP BY PAGE - This is critical for context!
        page_groups = {}
        for candidate in candidates:
            doc_page_key = f"{candidate['document_name']}_{candidate['page']}"
            if doc_page_key not in page_groups:
                page_groups[doc_page_key] = []
            page_groups[doc_page_key].append(candidate)
        
        print(f"Grouped into {len(page_groups)} page groups")
        
        # BOOST: Combine scores for chunks from the same page
        page_scores = {}
        for doc_page_key, chunks in page_groups.items():
            # Separate by chunk type
            text_chunks = [c for c in chunks if c['type'] in ['text', 'hardware_spec']]
            visual_chunks = [c for c in chunks if c['type'] == 'visual']  # NEW: visual type
            hardware_chunks = [c for c in chunks if c['type'] == 'hardware_spec']
            
            # Calculate combined score
            avg_score = sum(c['score'] for c in chunks) / len(chunks)
            
            # BOOST if page has BOTH text AND visuals (complementary information)
            page_diversity_boost = 0.15 if (text_chunks and visual_chunks) else 0
            
            # BOOST if multiple chunks from same page (indicates relevance)
            multi_chunk_boost = min(len(chunks) * 0.05, 0.2)  # Max 20% boost
            
            # EXTRA BOOST for hardware specs (often critical information)
            hardware_boost = 0.1 if hardware_chunks else 0
            
            # EXTRA BOOST if visual contains key technical elements
            visual_boost = 0.0
            if visual_chunks:
                for vc in visual_chunks:
                    key_elements = vc['metadata'].get('metadata', {}).get('key_elements', [])
                    if key_elements:  # Has identified key elements
                        visual_boost = 0.1
                        break
            
            page_scores[doc_page_key] = {
                'base_score': avg_score,
                'diversity_boost': page_diversity_boost,
                'multi_chunk_boost': multi_chunk_boost,
                'hardware_boost': hardware_boost,
                'visual_boost': visual_boost,
                'final_score': avg_score + page_diversity_boost + multi_chunk_boost + hardware_boost + visual_boost,
                'chunks': chunks,
                'page': chunks[0]['page'],
                'document': chunks[0]['document_name'],
                'has_visuals': len(visual_chunks) > 0,
                'has_hardware': len(hardware_chunks) > 0
            }
        
        # Sort pages by score
        sorted_pages = sorted(
            page_scores.items(), 
            key=lambda x: x[1]['final_score'], 
            reverse=True
        )
        
        print(f"\n=== Top 5 Pages by Score ===")
        for i, (doc_page_key, page_data) in enumerate(sorted_pages[:5], 1):
            print(f"{i}. {page_data['document']} Page {page_data['page']}")
            print(f"   Score: {page_data['final_score']:.4f} (base: {page_data['base_score']:.4f})")
            print(f"   Boosts: diversity +{page_data['diversity_boost']:.4f}, "
                  f"multi +{page_data['multi_chunk_boost']:.4f}, "
                  f"hardware +{page_data['hardware_boost']:.4f}, "
                  f"visual +{page_data['visual_boost']:.4f}")
            print(f"   Chunks: {len(page_data['chunks'])} "
                  f"({len([c for c in page_data['chunks'] if c['type'] in ['text', 'hardware_spec']])} text, "
                  f"{len([c for c in page_data['chunks'] if c['type']=='visual'])} visual)")
            print(f"   Has visuals: {'✅' if page_data['has_visuals'] else '❌'}")
            print(f"   Has hardware specs: {'✅' if page_data['has_hardware'] else '❌'}")
        
        # Flatten back to chunks, but in page-grouped order
        reordered_candidates = []
        for doc_page_key, page_data in sorted_pages:
            # Sort chunks within page
            # Priority: hardware_spec > text > visual
            page_chunks = sorted(
                page_data['chunks'], 
                key=lambda x: (
                    0 if x['type'] == 'hardware_spec' else (1 if x['type'] == 'text' else 2),
                    -x['score']
                )
            )
            reordered_candidates.extend(page_chunks)
        
        # Rerank with cross-encoder
        if reordered_candidates:
            # Take top candidates for reranking
            candidates_to_rerank = reordered_candidates[:top_k * 3]
            texts = [c["text"] for c in candidates_to_rerank]
            
            print(f"\n=== Reranking {len(candidates_to_rerank)} candidates ===")
            
            rerank_scores = self.reranker.predict(
                [(query, text) for text in texts]
            )
            
            # Combine scores
            for i, candidate in enumerate(candidates_to_rerank):
                candidate["rerank_score"] = float(rerank_scores[i])
                # Weight: 30% original embedding score + 70% reranker score
                candidate["final_score"] = (
                    0.3 * candidate["score"] + 
                    0.7 * candidate["rerank_score"]
                )
            
            # Final sort by combined score
            candidates_to_rerank.sort(key=lambda x: x["final_score"], reverse=True)
            final_candidates = candidates_to_rerank[:top_k]
        else:
            final_candidates = []
        
        print(f"\n=== Final Top {len(final_candidates)} Results ===")
        for i, candidate in enumerate(final_candidates, 1):
            chunk_type = candidate['type']
            type_emoji = "📝" if chunk_type == "text" else ("🔧" if chunk_type == "hardware_spec" else "🖼️")
            
            print(f"{i}. {type_emoji} [{chunk_type}] Page {candidate['page']} - "
                  f"Score: {candidate['final_score']:.4f}")
            print(f"   {candidate['text'][:100]}...")
        
        # Cache results
        self.redis_client.setex(
            cache_key,
            3600,  # 1 hour TTL
            json.dumps(final_candidates)
        )
        
        return final_candidates
    
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