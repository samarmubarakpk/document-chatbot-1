# workflow_manager.py - NEW FILE

import datetime
from typing import Dict, List, Any
import asyncio
from enum import Enum
import uuid

from app.document_processor import EnhancedDocumentProcessor

class WorkflowStep(Enum):
    UPLOAD = "1_UPLOAD"
    EXTRACT = "2_EXTRACT"
    ANALYZE = "3_ANALYZE"
    UNDERSTAND = "4_UNDERSTAND"
    GENERATE_DOC = "5_GENERATE_DOC"
    GENERATE_IMG = "6_GENERATE_IMG"
    VALIDATE = "7_VALIDATE"

class WorkflowManager:
    def __init__(self, processor: EnhancedDocumentProcessor):
        self.processor = processor
        self.workflow_status = {}
        
    async def execute_phase_one(self, file_path: str) -> Dict[str, Any]:
        """
        Phase 1: Document Analysis and Understanding
        """
        workflow_id = str(uuid.uuid4())
        self.workflow_status[workflow_id] = {
            "status": "started",
            "steps": {}
        }
        
        try:
            # Step 1: Upload and Extract
            self._update_status(workflow_id, WorkflowStep.UPLOAD, "processing")
            extracted = await self.processor.process_document(file_path, "analysis_doc")
            self._update_status(workflow_id, WorkflowStep.UPLOAD, "completed", extracted)
            
            # Step 2: Deep Analysis
            self._update_status(workflow_id, WorkflowStep.ANALYZE, "processing")
            analysis = await self._deep_analyze(extracted)
            self._update_status(workflow_id, WorkflowStep.ANALYZE, "completed", analysis)
            
            # Step 3: Understand Context
            self._update_status(workflow_id, WorkflowStep.UNDERSTAND, "processing")
            understanding = await self._understand_context(analysis)
            self._update_status(workflow_id, WorkflowStep.UNDERSTAND, "completed", understanding)
            
            return {
                "workflow_id": workflow_id,
                "phase": "one",
                "status": "completed",
                "results": {
                    "extracted": extracted,
                    "analysis": analysis,
                    "understanding": understanding
                }
            }
            
        except Exception as e:
            self.workflow_status[workflow_id]["status"] = "failed"
            self.workflow_status[workflow_id]["error"] = str(e)
            raise
    
    async def execute_phase_two(self, 
                               phase_one_results: List[Dict],
                               output_type: str = "document") -> Dict[str, Any]:
        """
        Phase 2: Document/Image Generation
        """
        workflow_id = str(uuid.uuid4())
        self.workflow_status[workflow_id] = {
            "status": "started",
            "steps": {}
        }
        
        try:
            if output_type == "document":
                # Generate Document
                self._update_status(workflow_id, WorkflowStep.GENERATE_DOC, "processing")
                new_doc = await self.processor.generate_document_from_analysis(phase_one_results)
                self._update_status(workflow_id, WorkflowStep.GENERATE_DOC, "completed")
                
                result = {"document": new_doc}
                
            elif output_type == "image":
                # Generate Image
                self._update_status(workflow_id, WorkflowStep.GENERATE_IMG, "processing")
                # Extract diagram descriptions from phase 1
                diagram_desc = self._extract_diagram_descriptions(phase_one_results)
                image_bytes = await self.processor.generate_image_from_description(diagram_desc)
                self._update_status(workflow_id, WorkflowStep.GENERATE_IMG, "completed")
                
                result = {"image": image_bytes}
            
            # Validation Step
            self._update_status(workflow_id, WorkflowStep.VALIDATE, "processing")
            validation = await self._validate_output(result, phase_one_results)
            self._update_status(workflow_id, WorkflowStep.VALIDATE, "completed", validation)
            
            return {
                "workflow_id": workflow_id,
                "phase": "two",
                "status": "completed",
                "output_type": output_type,
                "result": result,
                "validation": validation
            }
            
        except Exception as e:
            self.workflow_status[workflow_id]["status"] = "failed"
            self.workflow_status[workflow_id]["error"] = str(e)
            raise
    
    async def test_bidirectional_conversion(self, image_path: str) -> Dict[str, Any]:
        """
        Test Image-to-Text-to-Image conversion
        """
        # Image to Text
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        extracted = await self.processor.extract_and_understand_image(image_bytes)
        
        # Text to Image
        reconstruction_prompt = extracted.get("reconstruction_prompt", "")
        if reconstruction_prompt:
            new_image = await self.processor.generate_image_from_description(reconstruction_prompt)
            
            return {
                "original_analysis": extracted,
                "reconstruction_prompt": reconstruction_prompt,
                "new_image_generated": new_image is not None,
                "similarity_score": await self._calculate_similarity(image_bytes, new_image)
            }
        
        return {"error": "Could not extract reconstruction prompt"}
    
    def generate_workflow_diagram(self) -> str:
        """
        Generate a text-based workflow diagram
        """
        diagram = """
DOCUMENT INTELLIGENCE WORKFLOW - MODULAR ARCHITECTURE
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                         PHASE ONE                                │
│                   Document Analysis & Understanding              │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  1. UPLOAD   │──────▶│  2. EXTRACT  │──────▶│  3. ANALYZE  │
│   Document   │      │  Text+Images │      │   Content    │
└──────────────┘      └──────────────┘      └──────────────┘
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │ 4. UNDERSTAND│
                                            │   Context    │
                                            └──────────────┘
                                                    │
                            ┌───────────────────────┴───────────────────────┐
                            │                                               │
                            ▼                                               ▼
┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
│            PHASE TWO-A                   │   │            PHASE TWO-B                   │
│         Document Generation              │   │          Image Generation                │
└─────────────────────────────────────────┘   └─────────────────────────────────────────┘
                    │                                               │
                    ▼                                               ▼
            ┌──────────────┐                              ┌──────────────┐
            │ 5. GENERATE  │                              │ 6. GENERATE  │
            │   Document   │                              │    Image     │
            └──────────────┘                              └──────────────┘
                    │                                               │
                    └───────────────────┬───────────────────────────┘
                                        ▼
                                ┌──────────────┐
                                │  7. VALIDATE │
                                │    Output    │
                                └──────────────┘

MODULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. UPLOAD:    Ingests PDF/DOCX/Images
2. EXTRACT:   Extracts text + identifies visuals with GPT-4V
3. ANALYZE:   Deep analysis of content structure & meaning
4. UNDERSTAND: Contextual understanding & pattern recognition
5. GENERATE (Doc): Creates new compliant document
6. GENERATE (Img): Creates technical diagrams via DALL-E 3
7. VALIDATE:  Ensures output matches requirements
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return diagram
    
    def _update_status(self, workflow_id: str, step: WorkflowStep, status: str, data: Any = None):
        """Update workflow status"""
        self.workflow_status[workflow_id]["steps"][step.value] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
    
    async def _calculate_similarity(self, img1: bytes, img2: bytes) -> float:
        """Calculate similarity between two images"""
        # This would use image similarity algorithms
        # For now, return a placeholder
        return 0.85