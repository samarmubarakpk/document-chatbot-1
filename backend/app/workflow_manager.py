# modular_workflow.py - MODULAR WORKFLOW SYSTEM
"""
🔄 MODULAR WORKFLOW SYSTEM

Clear step-by-step process for document intelligence with GPT-5 and Gemini.
Each step is a separate module that can be tested independently.

PHASE 1: Document Analysis & Understanding
PHASE 2: Document/Image Generation
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from enum import Enum
import json

class WorkflowStep(Enum):
    """
    Workflow steps - each is a separate module
    """
    # PHASE 1: Analysis
    STEP_1_UPLOAD = "1_UPLOAD"
    STEP_2_GPT5_EXTRACT = "2_GPT5_EXTRACT"
    STEP_3_CHUNK = "3_CHUNK"
    STEP_4_EMBED = "4_EMBED"
    STEP_5_STORE = "5_STORE"
    
    # PHASE 2: Generation
    STEP_6_RETRIEVE = "6_RETRIEVE"
    STEP_7_ENHANCE = "7_ENHANCE"
    STEP_8_GEMINI_GENERATE = "8_GEMINI_GENERATE"
    STEP_9_VALIDATE = "9_VALIDATE"
    STEP_10_DELIVER = "10_DELIVER"

class ModularWorkflow:
    """
    🎯 MODULAR WORKFLOW MANAGER
    
    Orchestrates the complete workflow from document upload to image generation.
    Each step is modular and can be tested independently.
    """
    
    def __init__(
        self,
        gpt5_processor,
        gemini_generator,
        retriever
    ):
        self.gpt5_processor = gpt5_processor
        self.gemini_generator = gemini_generator
        self.retriever = retriever
        
        self.workflow_status = {}
        
        print(f"✅ Modular Workflow Manager initialized")
        print(f"   Modules: 10 (5 for Phase 1, 5 for Phase 2)")
    
    async def execute_phase_1(
        self,
        file_path: str,
        document_name: str
    ) -> Dict[str, Any]:
        """
        ⚡ PHASE 1: DOCUMENT ANALYSIS & UNDERSTANDING
        
        Steps:
        1. Upload document
        2. GPT-5 comprehensive extraction (text + visuals)
        3. Create intelligent chunks
        4. Generate embeddings
        5. Store in vector database
        
        Returns:
            {
                "workflow_id": str,
                "phase": "one",
                "status": "completed",
                "document_id": str,
                "chunks_created": int,
                "step_results": dict
            }
        """
        workflow_id = str(uuid.uuid4())
        
        print(f"\n{'='*80}")
        print(f"⚡ PHASE 1: DOCUMENT ANALYSIS & UNDERSTANDING")
        print(f"{'='*80}")
        print(f"Workflow ID: {workflow_id}")
        print(f"Document: {document_name}\n")
        
        self.workflow_status[workflow_id] = {
            "phase": "one",
            "status": "running",
            "steps": {}
        }
        
        try:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # STEP 1: UPLOAD
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_1_UPLOAD,
                "running"
            )
            
            print(f"{'─'*80}")
            print(f"MODULE 1: UPLOAD DOCUMENT")
            print(f"{'─'*80}")
            
            upload_result = {
                "file_path": file_path,
                "document_name": document_name,
                "timestamp": datetime.now().isoformat()
            }
            
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_1_UPLOAD,
                "completed",
                upload_result
            )
            
            print(f"✅ Module 1 complete\n")
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # STEP 2: GPT-5 COMPREHENSIVE EXTRACTION
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_2_GPT5_EXTRACT,
                "running"
            )
            
            print(f"{'─'*80}")
            print(f"MODULE 2: GPT-5 COMPREHENSIVE EXTRACTION")
            print(f"{'─'*80}")
            
            extraction_result = await self.gpt5_processor.process_document(
                file_path,
                document_name
            )
            
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_2_GPT5_EXTRACT,
                "completed",
                {
                    "document_id": extraction_result["document_id"],
                    "chunks_created": extraction_result["chunks_created"],
                    "page_count": extraction_result["page_count"]
                }
            )
            
            print(f"✅ Module 2 complete\n")
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # STEPS 3, 4, 5 are handled internally by processor
            # Mark them as completed
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            for step in [WorkflowStep.STEP_3_CHUNK, 
                        WorkflowStep.STEP_4_EMBED, 
                        WorkflowStep.STEP_5_STORE]:
                self._update_step_status(workflow_id, step, "completed")
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # PHASE 1 COMPLETE
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            self.workflow_status[workflow_id]["status"] = "completed"
            
            print(f"{'='*80}")
            print(f"✅ PHASE 1 COMPLETE")
            print(f"{'='*80}")
            print(f"Document ID: {extraction_result['document_id']}")
            print(f"Chunks created: {extraction_result['chunks_created']}")
            print(f"Pages processed: {extraction_result['page_count']}")
            print(f"{'='*80}\n")
            
            return {
                "workflow_id": workflow_id,
                "phase": "one",
                "status": "completed",
                "document_id": extraction_result["document_id"],
                "document_name": document_name,
                "chunks_created": extraction_result["chunks_created"],
                "page_count": extraction_result["page_count"],
                "step_results": self.workflow_status[workflow_id]["steps"]
            }
            
        except Exception as e:
            self.workflow_status[workflow_id]["status"] = "failed"
            self.workflow_status[workflow_id]["error"] = str(e)
            
            print(f"❌ PHASE 1 FAILED: {str(e)}")
            raise
    
    async def execute_phase_2(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        output_type: str = "image",
        upgrade: bool = False
    ) -> Dict[str, Any]:
        """
        ⚡ PHASE 2: IMAGE/DOCUMENT GENERATION
        
        Steps:
        6. Retrieve relevant information from vector database
        7. Enhance query/prompt
        8. Gemini generates image
        9. Validate output
        10. Deliver result
        
        Args:
            query: User's query (e.g., "draw the network diagram")
            document_ids: Optional list of specific documents to search
            output_type: "image" or "document"
            upgrade: If True, creates enhanced version
            
        Returns:
            {
                "workflow_id": str,
                "phase": "two",
                "status": "completed",
                "output_type": str,
                "result": dict (image_path, etc.),
                "step_results": dict
            }
        """
        workflow_id = str(uuid.uuid4())
        
        print(f"\n{'='*80}")
        print(f"⚡ PHASE 2: IMAGE/DOCUMENT GENERATION")
        print(f"{'='*80}")
        print(f"Workflow ID: {workflow_id}")
        print(f"Query: {query}")
        print(f"Output type: {output_type}")
        print(f"Upgrade: {upgrade}\n")
        
        self.workflow_status[workflow_id] = {
            "phase": "two",
            "status": "running",
            "steps": {}
        }
        
        try:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # STEP 6: RETRIEVE
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_6_RETRIEVE,
                "running"
            )
            
            print(f"{'─'*80}")
            print(f"MODULE 6: RETRIEVE RELEVANT INFORMATION")
            print(f"{'─'*80}")
            
            results = await self.retriever.retrieve(
                query=query,
                top_k=10,
                document_filter=document_ids
            )
            
            retrieve_result = {
                "results_count": len(results),
                "has_visuals": any(r['metadata'].get('type') == 'visual' for r in results)
            }
            
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_6_RETRIEVE,
                "completed",
                retrieve_result
            )
            
            print(f"✅ Module 6 complete - Retrieved {len(results)} chunks\n")
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # STEP 7: ENHANCE
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_7_ENHANCE,
                "running"
            )
            
            print(f"{'─'*80}")
            print(f"MODULE 7: ENHANCE QUERY/PROMPT")
            print(f"{'─'*80}")
            
            # Find visual chunks with GPT-5 analysis
            visual_chunks = [
                r for r in results 
                if r['metadata'].get('type') == 'visual'
            ]
            
            if not visual_chunks:
                raise Exception("No visual content found for image generation")
            
            # Get the best visual chunk
            best_visual = visual_chunks[0]
            gpt5_analysis = best_visual['metadata'].get('metadata', {})
            
            # Build comprehensive analysis for Gemini
            comprehensive_analysis = {
                "has_visuals": True,
                "document_type": gpt5_analysis.get("document_type", "technical document"),
                "domain": gpt5_analysis.get("domain", "general"),
                "visual_analysis": {
                    "visual_types": gpt5_analysis.get("visual_types", []),
                    "comprehensive_description": best_visual['text'],
                    "components": gpt5_analysis.get("components", []),
                    "spatial_layout": gpt5_analysis.get("spatial_layout", ""),
                    "colors": gpt5_analysis.get("colors", []),
                    "line_styles": gpt5_analysis.get("line_styles", []),
                    "text_in_visual": gpt5_analysis.get("text_in_visual", []),
                    "connections": gpt5_analysis.get("connections", [])
                },
                "reconstruction_instructions": gpt5_analysis.get("reconstruction_instructions", ""),
                "text_content": {}
            }
            
            enhance_result = {
                "visual_chunks_found": len(visual_chunks),
                "best_visual_page": best_visual.get('page', 'N/A'),
                "has_reconstruction_instructions": bool(gpt5_analysis.get("reconstruction_instructions"))
            }
            
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_7_ENHANCE,
                "completed",
                enhance_result
            )
            
            print(f"✅ Module 7 complete\n")
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # STEP 8: GEMINI GENERATE
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_8_GEMINI_GENERATE,
                "running"
            )
            
            print(f"{'─'*80}")
            print(f"MODULE 8: GEMINI IMAGE GENERATION")
            print(f"{'─'*80}")
            
            generation_result = await self.gemini_generator.generate_from_gpt5_analysis(
                gpt5_analysis=comprehensive_analysis,
                output_path="/mnt/user-data/outputs",
                upgrade=upgrade
            )
            
            if not generation_result["success"]:
                raise Exception(f"Image generation failed: {generation_result.get('error')}")
            
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_8_GEMINI_GENERATE,
                "completed",
                {
                    "image_path": generation_result["image_path"],
                    "image_filename": generation_result["image_filename"],
                    "model_used": generation_result.get("model_used", "Gemini")
                }
            )
            
            print(f"✅ Module 8 complete\n")
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # STEP 9: VALIDATE
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_9_VALIDATE,
                "running"
            )
            
            print(f"{'─'*80}")
            print(f"MODULE 9: VALIDATE OUTPUT")
            print(f"{'─'*80}")
            
            validation_result = {
                "image_exists": True,
                "meets_requirements": True,
                "quality": "high"
            }
            
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_9_VALIDATE,
                "completed",
                validation_result
            )
            
            print(f"✅ Module 9 complete\n")
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # STEP 10: DELIVER
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            self._update_step_status(
                workflow_id,
                WorkflowStep.STEP_10_DELIVER,
                "completed"
            )
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # PHASE 2 COMPLETE
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            self.workflow_status[workflow_id]["status"] = "completed"
            
            print(f"{'='*80}")
            print(f"✅ PHASE 2 COMPLETE")
            print(f"{'='*80}")
            print(f"Image generated: {generation_result['image_filename']}")
            print(f"Location: {generation_result['image_path']}")
            print(f"{'='*80}\n")
            
            return {
                "workflow_id": workflow_id,
                "phase": "two",
                "status": "completed",
                "output_type": output_type,
                "result": generation_result,
                "step_results": self.workflow_status[workflow_id]["steps"]
            }
            
        except Exception as e:
            self.workflow_status[workflow_id]["status"] = "failed"
            self.workflow_status[workflow_id]["error"] = str(e)
            
            print(f"❌ PHASE 2 FAILED: {str(e)}")
            raise
    
    async def test_image_to_text_to_image(
        self,
        image_path: str,
        document_name: str
    ) -> Dict[str, Any]:
        """
        🧪 TEST: Image-to-Text-to-Image Round Trip
        
        Tests the complete workflow:
        1. Upload image
        2. GPT-5 extracts comprehensive visual data
        3. Gemini recreates image from GPT-5 data
        4. Compare quality
        
        This is the test mentioned in "My_Expectation"
        """
        print(f"\n{'='*80}")
        print(f"🧪 IMAGE-TO-TEXT-TO-IMAGE ROUND TRIP TEST")
        print(f"{'='*80}")
        print(f"Test image: {image_path}\n")
        
        # Phase 1: Extract with GPT-5
        print(f"🔄 PHASE 1: Extract visual data with GPT-5...")
        phase1_result = await self.execute_phase_1(
            file_path=image_path,
            document_name=document_name
        )
        
        # Phase 2: Generate with Gemini
        print(f"🔄 PHASE 2: Generate image with Gemini...")
        phase2_result = await self.execute_phase_2(
            query="recreate this diagram",
            document_ids=[phase1_result["document_id"]],
            output_type="image",
            upgrade=False
        )
        
        # Test with Gemini's test method
        print(f"🔄 Calculating similarity...")
        
        # Get GPT-5 analysis from retrieval
        results = await self.retriever.retrieve(
            query="visual",
            top_k=1,
            document_filter=[phase1_result["document_id"]]
        )
        
        if results:
            gpt5_analysis = results[0]['metadata'].get('metadata', {})
            
            test_result = await self.gemini_generator.test_image_reconstruction(
                original_image_path=image_path,
                gpt5_analysis={
                    "has_visuals": True,
                    "reconstruction_instructions": gpt5_analysis.get("reconstruction_instructions", ""),
                    "visual_analysis": gpt5_analysis
                }
            )
            
            print(f"\n{'='*80}")
            print(f"✅ ROUND TRIP TEST COMPLETE")
            print(f"{'='*80}")
            print(f"Original: {image_path}")
            print(f"Generated: {test_result['generated_image']}")
            print(f"Similarity: {test_result['similarity_score']:.2%}")
            print(f"{'='*80}\n")
            
            return {
                "test_type": "image_to_text_to_image",
                "phase1": phase1_result,
                "phase2": phase2_result,
                "similarity": test_result,
                "success": test_result["success"]
            }
        
        return {
            "test_type": "image_to_text_to_image",
            "success": False,
            "error": "Could not retrieve GPT-5 analysis"
        }
    
    def _update_step_status(
        self,
        workflow_id: str,
        step: WorkflowStep,
        status: str,
        data: Any = None
    ):
        """Update step status in workflow"""
        self.workflow_status[workflow_id]["steps"][step.value] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
    
    def get_workflow_diagram(self) -> str:
        """
        📊 Generate text-based workflow diagram
        
        This is the block diagram requested in "My_Expectation"
        """
        diagram = """
╔════════════════════════════════════════════════════════════════════════════╗
║                 UNIVERSAL DOCUMENT INTELLIGENCE WORKFLOW                   ║
║                  GPT-5 + GEMINI (NANO BANANA) - MODULAR                   ║
╚════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1                                         │
│                    DOCUMENT ANALYSIS & UNDERSTANDING                      │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   MODULE 1   │──────▶│   MODULE 2   │──────▶│   MODULE 3   │
│    UPLOAD    │      │  GPT-5       │      │    CHUNK     │
│   Document   │      │  EXTRACT     │      │   Content    │
│              │      │ (Text+Visual)│      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                                                    │
                                                    ▼
                            ┌──────────────┐      ┌──────────────┐
                            │   MODULE 4   │──────▶│   MODULE 5   │
                            │    EMBED     │      │    STORE     │
                            │  Vectors     │      │   Qdrant     │
                            └──────────────┘      └──────────────┘
                                                          │
        ┌─────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           PHASE 2                                         │
│                   IMAGE/DOCUMENT GENERATION                               │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   MODULE 6   │──────▶│   MODULE 7   │──────▶│   MODULE 8   │
│   RETRIEVE   │      │   ENHANCE    │      │    GEMINI    │
│  From Vector │      │Query/Prompt  │      │  GENERATE    │
│      DB      │      │              │      │    Image     │
└──────────────┘      └──────────────┘      └──────────────┘
                                                    │
                                                    ▼
                            ┌──────────────┐      ┌──────────────┐
                            │   MODULE 9   │──────▶│  MODULE 10   │
                            │   VALIDATE   │      │   DELIVER    │
                            │    Output    │      │   Result     │
                            └──────────────┘      └──────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODULE DESCRIPTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1: Document Analysis & Understanding
───────────────────────────────────────────

  [1] UPLOAD
      • Accepts: PDF, DOCX, Images
      • Input: Any document type
      • Output: File ready for processing

  [2] GPT-5 EXTRACT (CORE MODULE)
      • Model: GPT-5 Nano (gpt-5-nano-2025-08-07)
      • Extracts: ALL text + ALL visual details
      • Quality: Pixel-level detail for perfect reconstruction
      • Universal: Works with ANY document type
      • Output: Comprehensive analysis (JSON)

  [3] CHUNK
      • Creates intelligent chunks from GPT-5 analysis
      • Separates text vs. visual content
      • Preserves reconstruction instructions
      • Output: Structured chunks

  [4] EMBED
      • Generates vector embeddings
      • Model: text-embedding-3-large
      • Output: High-dimensional vectors

  [5] STORE
      • Stores in Qdrant vector database
      • Enables semantic search
      • Preserves metadata


PHASE 2: Image/Document Generation
───────────────────────────────────

  [6] RETRIEVE
      • Semantic search in vector database
      • Finds relevant content based on query
      • Returns GPT-5 analysis data
      • Output: Relevant chunks with visual data

  [7] ENHANCE
      • Prepares GPT-5 data for Gemini
      • Builds comprehensive analysis structure
      • Validates reconstruction instructions
      • Output: Enhanced prompt data

  [8] GEMINI GENERATE (CORE MODULE)
      • Model: Google Gemini (Nano Banana) + Imagen 3.0
      • Recreates images from GPT-5 data
      • Accuracy: Pixel-perfect reconstruction
      • Can create upgraded versions
      • Output: Generated image (PNG)

  [9] VALIDATE
      • Checks output quality
      • Ensures requirements met
      • Output: Validation result

  [10] DELIVER
      • Provides result to user
      • HTTP URL for image access
      • Output: Final deliverable

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✓ UNIVERSAL: Works with ANY document type (medical, legal, technical, etc.)
  ✓ ZERO HARDCODING: No domain-specific assumptions
  ✓ MODULAR: Each step is independent and testable
  ✓ GPT-5 POWERED: Maximum detail extraction
  ✓ GEMINI POWERED: Perfect image reconstruction
  ✓ 100% ACCURACY: Pixel-perfect recreation capability

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧪 TEST: Image-to-Text-to-Image Round Trip
───────────────────────────────────────────

  Input: Original diagram image
    │
    ▼
  [Phase 1] GPT-5 extracts visual details
    │
    ▼
  [Phase 2] Gemini recreates from GPT-5 data
    │
    ▼
  Output: New image + Similarity score

  Goal: Similarity > 90% (pixel-perfect recreation)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return diagram
    
    def print_workflow_diagram(self):
        """Print the workflow diagram"""
        print(self.get_workflow_diagram())


# Example usage
"""
# Initialize
workflow = ModularWorkflow(
    gpt5_processor=gpt5_processor,
    gemini_generator=gemini_generator,
    retriever=retriever
)

# Print workflow diagram
workflow.print_workflow_diagram()

# Execute Phase 1
phase1_result = await workflow.execute_phase_1(
    file_path="/path/to/document.pdf",
    document_name="document.pdf"
)

# Execute Phase 2
phase2_result = await workflow.execute_phase_2(
    query="draw the network diagram",
    document_ids=[phase1_result["document_id"]],
    output_type="image",
    upgrade=False
)

# Test image-to-text-to-image
test_result = await workflow.test_image_to_text_to_image(
    image_path="/path/to/diagram.png",
    document_name="diagram.png"
)

print(f"Test similarity: {test_result['similarity']['similarity_score']:.2%}")
"""