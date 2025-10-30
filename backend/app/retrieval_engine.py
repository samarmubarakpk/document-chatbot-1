# retrieval_engine.py - PRODUCTION-READY HYBRID RETRIEVAL ENGINE
"""
🚀 HYBRID RETRIEVAL ENGINE
- Semantic search via embeddings
- Keyword/BM25 boosting for exact matches
- Smart scoring without hardcoding
- Works with ANY document type
"""

from typing import List, Dict, Any, Optional
import numpy as np
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, SearchParams
from sentence_transformers import CrossEncoder
import redis
import json
import hashlib
import re
from collections import Counter
from app.config import settings

class HybridRetriever:
    """
    Production-grade hybrid retrieval combining:
    1. Dense semantic search (embeddings)
    2. Sparse keyword search (BM25-style)
    3. Cross-encoder reranking
    4. Smart score fusion
    """
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.qdrant_client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Redis with error handling
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Test connection
            self.redis_client.ping()
            self.use_cache = True
            print("✅ Redis cache enabled")
        except Exception as e:
            print(f"⚠️ Redis unavailable: {e}")
            print("   Continuing without cache...")
            self.redis_client = None
            self.use_cache = False
    
    async def retrieve(
        self, 
        query: str, 
        top_k: int = 10,
        document_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval function with hybrid search
        
        Args:
            query: User's search query
            top_k: Number of results to return
            document_filter: Optional list of document IDs to filter
            
        Returns:
            List of ranked results with scores and metadata
        """
        
        print(f"\n{'='*80}")
        print(f"🔍 HYBRID RETRIEVAL STARTED")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Top K: {top_k}")
        print(f"Document filter: {document_filter}")
        
        # Check cache
        if self.use_cache:
            cache_key = self._get_cache_key(query, top_k, document_filter)
            try:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    print("✅ Cache HIT - returning cached results")
                    return json.loads(cached_result)
            except Exception as e:
                print(f"⚠️ Cache read failed: {e}")
        
        # Step 1: SEMANTIC SEARCH (Embedding-based)
        print(f"\n📊 Step 1: Semantic Search")
        semantic_results = await self._semantic_search(query, top_k * 5, document_filter)
        print(f"   Retrieved {len(semantic_results)} candidates")
        
        # Step 2: KEYWORD EXTRACTION & BOOSTING
        print(f"\n🔑 Step 2: Keyword Boosting")
        keywords = self._extract_keywords(query)
        print(f"   Extracted keywords: {keywords}")
        
        boosted_results = self._apply_keyword_boost(semantic_results, keywords, query)
        
        # Step 3: QUERY-SPECIFIC SCORING
        print(f"\n🎯 Step 3: Smart Scoring")
        scored_results = self._apply_smart_scoring(boosted_results, query, keywords)
        
        # Step 4: CROSS-ENCODER RERANKING
        print(f"\n🔄 Step 4: Reranking Top Candidates")
        reranked_results = await self._rerank_with_cross_encoder(
            query, 
            scored_results[:top_k * 3],
            top_k
        )
        
        # Step 5: FINAL RESULTS
        final_results = reranked_results[:top_k]
        
        print(f"\n{'='*80}")
        print(f"✅ RETRIEVAL COMPLETE - Returning {len(final_results)} results")
        print(f"{'='*80}")
        
        self._print_final_results(final_results)
        
        # Cache results
        if self.use_cache:
            try:
                self.redis_client.setex(
                    cache_key, 
                    3600,  # 1 hour
                    json.dumps(final_results)
                )
            except Exception as e:
                print(f"⚠️ Cache write failed: {e}")
        
        return final_results
    
    async def _semantic_search(
        self,
        query: str,
        limit: int,
        document_filter: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using embeddings
        """
        # Generate query embedding
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
        
        # Search Qdrant
        search_results = await self.qdrant_client.search(
            collection_name="documents",
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter
        )
        
        # Convert to standard format
        candidates = []
        for result in search_results:
            candidates.append({
                "id": result.id,
                "text": result.payload.get("text", ""),
                "semantic_score": float(result.score),
                "metadata": result.payload.get("metadata", {}),
                "page": result.payload.get("page", 0),
                "type": result.payload.get("type", "text"),
                "document_name": result.payload.get("document_name", ""),
                "document_id": result.payload.get("document_id", "")
            })
        
        return candidates
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query
        - Numbers (ratios, values, measurements)
        - Technical terms
        - Capitalized acronyms
        - Multi-word technical phrases
        """
        keywords = []
        
        # Extract numbers with context (e.g., "20:1", "4.8Gbps", "10GE")
        number_patterns = [
            r'\d+:\d+',           # Ratios: 20:1
            r'\d+\.?\d*\s*[A-Za-z]+',  # Numbers with units: 4.8Gbps, 10GE
            r'\d+',               # Plain numbers
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, query)
            keywords.extend(matches)
        
        # Extract capitalized acronyms (e.g., LAN, OSPF, VLAN)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', query)
        keywords.extend(acronyms)
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        keywords.extend(quoted)
        
        # Extract technical terms (words with 4+ chars that aren't common words)
        common_words = {'what', 'how', 'why', 'when', 'where', 'which', 'does', 'this', 
                       'that', 'these', 'those', 'with', 'from', 'have', 'been', 'their'}
        
        words = re.findall(r'\b\w{4,}\b', query.lower())
        technical_terms = [w for w in words if w not in common_words]
        keywords.extend(technical_terms)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for k in keywords:
            k_lower = k.lower()
            if k_lower not in seen:
                seen.add(k_lower)
                unique_keywords.append(k)
        
        return unique_keywords
    
    def _apply_keyword_boost(
        self,
        candidates: List[Dict],
        keywords: List[str],
        query: str
    ) -> List[Dict]:
        """
        Boost scores for chunks containing important keywords
        """
        for candidate in candidates:
            text = candidate['text'].lower()
            keyword_score = 0
            
            # Exact keyword matches (case-insensitive)
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Count occurrences
                count = text.count(keyword_lower)
                
                if count > 0:
                    # Base boost per occurrence
                    base_boost = 0.05
                    
                    # Extra boost for rare/specific terms
                    if len(keyword) > 6 or re.match(r'\d', keyword):
                        base_boost = 0.1  # Numbers and long terms are more important
                    
                    keyword_score += base_boost * count
            
            # Boost for exact phrase match
            if query.lower() in text:
                keyword_score += 0.2
            
            candidate['keyword_score'] = keyword_score
            candidate['boosted_score'] = candidate['semantic_score'] + keyword_score
        
        # Sort by boosted score
        candidates.sort(key=lambda x: x['boosted_score'], reverse=True)
        
        return candidates
    
    def _apply_smart_scoring(
        self,
        candidates: List[Dict],
        query: str,
        keywords: List[str]
    ) -> List[Dict]:
        """
        Apply intelligent scoring based on:
        - Chunk type (text vs visual)
        - Content density
        - Answer likelihood
        """
        
        # Detect query type
        is_asking_for_calculation = any(word in query.lower() for word in 
                                        ['calculate', 'calculation', 'show', 'compute', 'how much'])
        is_asking_for_visual = any(word in query.lower() for word in 
                                   ['diagram', 'draw', 'visualize', 'image', 'picture', 'topology'])
        is_asking_for_specs = any(word in query.lower() for word in 
                                  ['specification', 'hardware', 'model', 'equipment'])
        
        for candidate in candidates:
            chunk_type = candidate['type']
            text = candidate['text']
            
            # Base score
            score = candidate['boosted_score']
            
            # Type-based adjustments
            if is_asking_for_visual and chunk_type == 'visual':
                score += 0.15  # Boost visual chunks for diagram questions
            elif not is_asking_for_visual and chunk_type == 'text':
                score += 0.1   # Boost text chunks for factual questions
            
            # Content quality signals
            
            # Has numbers (good for calculation/spec questions)
            if re.search(r'\d+', text) and (is_asking_for_calculation or is_asking_for_specs):
                score += 0.05
            
            # Has technical terms from metadata
            if candidate['metadata']:
                metadata_str = str(candidate['metadata']).lower()
                if any(kw.lower() in metadata_str for kw in keywords):
                    score += 0.05
            
            # Length considerations
            text_length = len(text)
            if 100 < text_length < 2000:
                # Sweet spot - not too short, not too long
                score += 0.03
            elif text_length < 50:
                # Very short chunks are usually less useful
                score -= 0.05
            
            candidate['smart_score'] = score
        
        # Re-sort by smart score
        candidates.sort(key=lambda x: x['smart_score'], reverse=True)
        
        return candidates
    
    async def _rerank_with_cross_encoder(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Use cross-encoder for final reranking of top candidates
        """
        if not candidates:
            return []
        
        print(f"   Reranking {len(candidates)} candidates...")
        
        # Prepare pairs for cross-encoder
        pairs = [(query, candidate['text']) for candidate in candidates]
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update candidates with rerank scores
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = float(rerank_scores[i])
            
            # Final score: weighted combination
            candidate['final_score'] = (
                0.3 * candidate['semantic_score'] +    # Semantic similarity
                0.3 * candidate.get('keyword_score', 0) +  # Keyword match
                0.4 * candidate['rerank_score']         # Cross-encoder (most weight)
            )
        
        # Final sort
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        return candidates[:top_k]
    
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
        """Generate embedding using OpenAI"""
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
    
    def _print_final_results(self, results: List[Dict]):
        """Print final results for debugging"""
        print(f"\n📊 FINAL TOP {len(results)} RESULTS:")
        print(f"{'='*80}")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 📄 {result['document_name']} - Page {result['page']}")
            print(f"   Type: {result['type']}")
            print(f"   Semantic: {result['semantic_score']:.4f} | "
                  f"Keyword: {result.get('keyword_score', 0):.4f} | "
                  f"Rerank: {result.get('rerank_score', 0):.4f}")
            print(f"   ⭐ FINAL SCORE: {result['final_score']:.4f}")
            print(f"   Preview: {result['text'][:150]}...")
        
        print(f"\n{'='*80}\n")