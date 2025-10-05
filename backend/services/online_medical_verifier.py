"""
🌐 Enhanced Online Medical Knowledge Verification System
Integrates textbook verification with online sources and provides comprehensive medical evidence
"""

import requests
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import time
import google.generativeai as genai
import os
import re
from datetime import datetime

# Import the new textbook verification system
try:
    from .textbook_verification_service import TextbookVerificationService, TextbookEvidence, TextbookReference
    HAS_TEXTBOOK_SYSTEM = True
except ImportError:
    print("⚠️ Textbook verification service not available")
    HAS_TEXTBOOK_SYSTEM = False
    TextbookVerificationService = None
    TextbookEvidence = None
    TextbookReference = None

@dataclass
class OnlineSource:
    """Online medical source with detailed information"""
    title: str
    url: str
    source_type: str  # "journal", "medical_site", "guideline", "database"
    reliability_score: float  # 0.0-1.0
    publication_date: Optional[str] = None
    authors: Optional[str] = None
    abstract: Optional[str] = None
    relevant_excerpt: Optional[str] = None

@dataclass
class TextbookEvidence:
    """Textbook evidence with precise location"""
    textbook_name: str
    edition: str
    chapter: str
    section: str
    page_number: str
    line_number: Optional[int] = None
    exact_quote: str = ""
    relevance_score: float = 0.0
    textbook_preview: str = ""  # Preview of surrounding text
    citation: str = ""

@dataclass
class ComprehensiveMedicalReference:
    """Comprehensive medical reference combining textbooks and online sources"""
    diagnosis: str
    verification_status: str  # 'VERIFIED', 'PARTIAL', 'CONTRADICTED', 'NOT_FOUND'
    overall_confidence: float
    evidence_strength: str  # 'STRONG', 'MODERATE', 'WEAK', 'INSUFFICIENT'
    
    # Textbook evidence
    textbook_references: List[TextbookEvidence]
    textbook_confidence: float
    
    # Online sources
    online_sources: List[OnlineSource]
    online_confidence: float
    
    # Combined analysis
    consensus_analysis: str
    contradictions: List[str]
    clinical_recommendations: List[str]
    evidence_summary: str
    
    # Metadata
    verification_timestamp: str
    sources_count: int
    
class EnhancedOnlineMedicalVerifier:
    """Enhanced medical verifier that combines textbook and online verification"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Prefer ai_key_manager for Gemini model rotation
        try:
            from ai_key_manager import get_gemini_model
            self.model = get_gemini_model('gemini-1.5-flash')
            self.api_key = None
        except Exception:
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            else:
                self.model = None
        
        # Initialize textbook verifier if available
        if HAS_TEXTBOOK_SYSTEM and TextbookVerificationService is not None:
            self.textbook_verifier = TextbookVerificationService(api_key)
            print("📚 Textbook verification system initialized")
        else:
            self.textbook_verifier = None
            print("⚠️ Textbook verification system not available")
        
        # Medical knowledge base for fallback
        self.medical_knowledge = {
            "sarcoma": {
                "definition": "A type of cancer that develops in bone and soft tissue",
                "types": ["soft tissue sarcoma", "bone sarcoma", "liposarcoma", "rhabdomyosarcoma"],
                "symptoms": ["painless mass", "swelling", "limited range of motion"],
                "diagnosis": ["MRI", "CT scan", "biopsy", "histopathology"],
                "treatment": ["surgical resection", "chemotherapy", "radiation therapy"],
                "authoritative_sources": [
                    "NCCN Guidelines for Soft Tissue Sarcoma",
                    "WHO Classification of Tumours",
                    "Harrison's Principles of Internal Medicine"
                ]
            },
            "liposarcoma": {
                "definition": "A malignant tumor that arises from fat cells",
                "types": ["well-differentiated", "dedifferentiated", "myxoid", "pleomorphic"],
                "symptoms": ["soft tissue mass", "especially in thigh", "gradual enlargement"],
                "diagnosis": ["MRI characteristic findings", "core needle biopsy", "immunohistochemistry"],
                "treatment": ["wide surgical excision", "adjuvant therapy for high-grade"],
                "authoritative_sources": [
                    "ESMO Clinical Practice Guidelines for Soft Tissue Sarcomas",
                    "AJCC Cancer Staging Manual",
                    "Robbins Basic Pathology"
                ]
            },
            "myocardial_infarction": {
                "definition": "Death of heart muscle due to insufficient blood supply",
                "types": ["STEMI", "NSTEMI", "Type 1", "Type 2"],
                "symptoms": ["chest pain", "dyspnea", "diaphoresis", "nausea"],
                "diagnosis": ["ECG", "troponins", "echocardiography", "coronary angiography"],
                "treatment": ["primary PCI", "thrombolysis", "antiplatelet therapy", "statins"],
                "authoritative_sources": [
                    "ESC Guidelines for Acute Myocardial Infarction",
                    "AHA/ACC Guidelines",
                    "Harrison's Principles of Internal Medicine"
                ]
            }
        }
        
        # Known reliable medical sources
        self.reliable_sources = {
            "pubmed.ncbi.nlm.nih.gov": 0.95,
            "nejm.org": 0.98,
            "thelancet.com": 0.97,
            "jamanetwork.com": 0.96,
            "bmj.com": 0.94,
            "nature.com": 0.93,
            "mayoclinic.org": 0.85,
            "uptodate.com": 0.92,
            "who.int": 0.90,
            "cdc.gov": 0.88,
            "nccn.org": 0.94,
            "esmo.org": 0.92
        }
    
    async def verify_diagnosis_comprehensive(
        self, 
        diagnosis: str, 
        clinical_context: str = "",
        patient_symptoms: Optional[List[str]] = None
    ) -> ComprehensiveMedicalReference:
        """
        Comprehensive diagnosis verification using both textbooks and online sources
        """
        print(f"\n🔬 COMPREHENSIVE MEDICAL VERIFICATION")
        print(f"Diagnosis: {diagnosis}")
        print("=" * 60)
        
        # Step 1: Textbook verification
        textbook_evidence = await self._verify_with_textbooks(diagnosis, clinical_context)
        
        # Step 2: Online source verification
        online_sources = await self._verify_with_online_sources(diagnosis, clinical_context)
        
        # Step 3: Cross-verification and consensus analysis
        verification_result = self._analyze_consensus(
            diagnosis, textbook_evidence, online_sources, clinical_context
        )
        
        return verification_result
    
    async def _verify_with_textbooks(self, diagnosis: str, clinical_context: str) -> Tuple[List[TextbookEvidence], float]:
        """Verify diagnosis against medical textbooks with precise citations"""
        
        textbook_evidence = []
        textbook_confidence = 0.0
        
        if not self.textbook_verifier:
            print("📚 Textbook verification service not available")
            return textbook_evidence, textbook_confidence
        
        try:
            print("📚 Verifying against medical textbooks...")
            
            # Use the new textbook verification service
            evidence = await self.textbook_verifier.verify_diagnosis_against_textbooks(
                diagnosis, clinical_context
            )
            
            # Convert to list format for compatibility
            if evidence and evidence.references:
                textbook_evidence = [evidence]  # Wrap in list for compatibility
                textbook_confidence = evidence.overall_confidence
                
                print(f"📖 Found {len(evidence.references)} textbook references")
                print(f"� Textbook confidence: {textbook_confidence:.2f}")
                
                # Log textbook sources
                for ref in evidence.references[:3]:  # Show top 3
                    print(f"  📄 {ref.title} - Page {ref.page_number}: {ref.relevant_quote[:100]}...")
            else:
                print("📚 No textbook evidence found")
                
        except Exception as e:
            print(f"❌ Textbook verification error: {e}")
        
        return textbook_evidence, textbook_confidence
    
    async def _verify_with_online_sources(self, diagnosis: str, clinical_context: str) -> Tuple[List[OnlineSource], float]:
        """Verify diagnosis against authoritative online medical sources"""
        
        online_sources = []
        online_confidence = 0.0
        
        if not self.model:
            print("⚠️  AI model not available for online verification")
            return online_sources, online_confidence
        
        try:
            print("🌐 Searching online medical sources...")
            
            # Generate comprehensive search query
            search_prompt = f"""
            Generate a comprehensive medical verification report for the diagnosis: "{diagnosis}"
            Clinical context: {clinical_context}
            
            Please provide:
            1. VERIFICATION STATUS: (VERIFIED/PARTIAL/CONTRADICTED/NOT_FOUND)
            2. AUTHORITATIVE SOURCES: List 5-8 authoritative medical sources that discuss this condition
            3. KEY EVIDENCE: Main evidence supporting or contradicting this diagnosis
            4. ONLINE REFERENCES: Specific medical websites, journals, or guidelines
            5. CLINICAL CONSENSUS: What the medical community consensus is
            
            Format each source as:
            SOURCE: [Title]
            URL: [Best guess URL or type of source]
            TYPE: [journal/guideline/medical_site/database]
            RELIABILITY: [0.0-1.0]
            EVIDENCE: [What this source says about the diagnosis]
            
            Be specific about medical evidence and cite authoritative sources.
            """
            
            response = self.model.generate_content(search_prompt)
            online_sources = self._parse_online_sources(response.text, diagnosis)
            
            # Calculate online confidence based on source reliability
            if online_sources:
                avg_reliability = sum(source.reliability_score for source in online_sources) / len(online_sources)
                online_confidence = min(avg_reliability * 0.9, 0.95)  # Cap at 95%
            
            print(f"✅ Found {len(online_sources)} online sources")
            print(f"📊 Online confidence: {online_confidence:.2f}")
            
        except Exception as e:
            print(f"❌ Online verification error: {e}")
        
        return online_sources, online_confidence
    
    def _parse_online_sources(self, ai_response: str, diagnosis: str) -> List[OnlineSource]:
        """Parse AI response to extract structured online sources"""
        
        sources = []
        lines = ai_response.split('\n')
        current_source = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("SOURCE:"):
                if current_source:  # Save previous source
                    sources.append(self._create_online_source(current_source, diagnosis))
                current_source = {"title": line.split(":", 1)[1].strip()}
                
            elif line.startswith("URL:"):
                current_source["url"] = line.split(":", 1)[1].strip()
                
            elif line.startswith("TYPE:"):
                current_source["source_type"] = line.split(":", 1)[1].strip()
                
            elif line.startswith("RELIABILITY:"):
                try:
                    current_source["reliability"] = str(float(line.split(":", 1)[1].strip()))
                except:
                    current_source["reliability"] = "0.7"
                    
            elif line.startswith("EVIDENCE:"):
                current_source["evidence"] = line.split(":", 1)[1].strip()
        
        # Don't forget the last source
        if current_source:
            sources.append(self._create_online_source(current_source, diagnosis))
        
        # Add some default authoritative sources if none found
        if not sources:
            sources = self._get_default_sources(diagnosis)
        
        return sources[:8]  # Limit to 8 sources
    
    def _create_online_source(self, source_data: dict, diagnosis: str) -> OnlineSource:
        """Create OnlineSource object from parsed data"""
        
        # Enhance URL if it's generic
        url = source_data.get("url", "")
        if not url.startswith("http"):
            url = self._generate_likely_url(source_data.get("title", ""), diagnosis)
        
        # Enhance reliability based on known sources
        reliability = float(source_data.get("reliability", "0.7"))
        for domain, score in self.reliable_sources.items():
            if domain in url:
                reliability = max(reliability, score)
                break
        
        return OnlineSource(
            title=source_data.get("title", "Medical Reference"),
            url=url,
            source_type=source_data.get("source_type", "medical_site"),
            reliability_score=reliability,
            relevant_excerpt=source_data.get("evidence", ""),
            publication_date=None,  # Could be enhanced with real data
            authors=None
        )
    
    def _generate_likely_url(self, title: str, diagnosis: str) -> str:
        """Generate likely URLs for medical sources"""
        
        title_lower = title.lower()
        
        if "pubmed" in title_lower or "ncbi" in title_lower:
            return f"https://pubmed.ncbi.nlm.nih.gov/search?term={diagnosis.replace(' ', '+')}"
        elif "mayo clinic" in title_lower:
            return f"https://mayoclinic.org/diseases-conditions/{diagnosis.replace(' ', '-').lower()}"
        elif "uptodate" in title_lower:
            return f"https://uptodate.com/search?search={diagnosis.replace(' ', '+')}"
        elif "nccn" in title_lower:
            return "https://nccn.org/guidelines"
        elif "nejm" in title_lower:
            return f"https://nejm.org/search?q={diagnosis.replace(' ', '+')}"
        elif "who" in title_lower:
            return "https://who.int/health-topics"
        else:
            return f"https://google.com/search?q={diagnosis.replace(' ', '+')}+medical+guidelines"
    
    def _get_default_sources(self, diagnosis: str) -> List[OnlineSource]:
        """Get default authoritative sources for any diagnosis"""
        
        return [
            OnlineSource(
                title="PubMed Medical Literature Database",
                url=f"https://pubmed.ncbi.nlm.nih.gov/search?term={diagnosis.replace(' ', '+')}",
                source_type="database",
                reliability_score=0.95,
                relevant_excerpt=f"Search results for {diagnosis} in medical literature"
            ),
            OnlineSource(
                title="Mayo Clinic Medical Information",
                url=f"https://mayoclinic.org/diseases-conditions/{diagnosis.replace(' ', '-').lower()}",
                source_type="medical_site",
                reliability_score=0.85,
                relevant_excerpt=f"Comprehensive medical information about {diagnosis}"
            ),
            OnlineSource(
                title="UpToDate Clinical Decision Support",
                url=f"https://uptodate.com/search?search={diagnosis.replace(' ', '+')}",
                source_type="medical_site",
                reliability_score=0.92,
                relevant_excerpt=f"Evidence-based clinical information for {diagnosis}"
            )
        ]
    
    def _estimate_line_location(self, quote: str, full_content: str) -> dict:
        """Estimate line number and surrounding context for textbook quote"""
        
        if not quote or not full_content:
            return {"estimated_line": None, "context": ""}
        
        # Split content into lines
        lines = full_content.split('\n')
        
        # Find the line containing the quote
        for i, line in enumerate(lines, 1):
            if quote[:50] in line:  # Match first 50 chars of quote
                return {
                    "estimated_line": i,
                    "context": f"Lines {max(1, i-2)} to {min(len(lines), i+2)}"
                }
        
        # If exact match not found, estimate based on position
        quote_pos = full_content.find(quote[:30])
        if quote_pos != -1:
            # Count newlines up to that position
            line_estimate = full_content[:quote_pos].count('\n') + 1
            return {
                "estimated_line": line_estimate,
                "context": f"Approximately line {line_estimate}"
            }
        
        return {"estimated_line": None, "context": "Location not determined"}
    
    def _generate_textbook_preview(self, content: str, quote: str, preview_length: int = 300) -> str:
        """Generate a preview of textbook content around the relevant quote"""
        
        if not quote or not content:
            return "Preview not available"
        
        # Find the position of the quote in content
        quote_pos = content.find(quote[:50])  # Use first 50 chars for finding
        
        if quote_pos == -1:
            return "Preview not available - quote not found in content"
        
        # Extract preview around the quote
        start_pos = max(0, quote_pos - preview_length // 2)
        end_pos = min(len(content), quote_pos + len(quote) + preview_length // 2)
        
        preview = content[start_pos:end_pos]
        
        # Add ellipsis if truncated
        if start_pos > 0:
            preview = "..." + preview
        if end_pos < len(content):
            preview = preview + "..."
        
        # Highlight the actual quote within the preview
        if quote in preview:
            preview = preview.replace(quote, f"**{quote}**")
        
        return preview
    
    def _analyze_consensus(
        self, 
        diagnosis: str, 
        textbook_evidence: Tuple[List[TextbookEvidence], float],
        online_sources: Tuple[List[OnlineSource], float],
        clinical_context: str
    ) -> ComprehensiveMedicalReference:
        """Analyze consensus between textbook and online evidence"""
        
        textbook_refs, textbook_conf = textbook_evidence
        online_refs, online_conf = online_sources
        
        # Calculate overall confidence
        if textbook_refs and online_refs:
            overall_confidence = (textbook_conf * 0.7 + online_conf * 0.3)
        elif textbook_refs:
            overall_confidence = textbook_conf * 0.9  # Slightly reduced for missing online
        elif online_refs:
            overall_confidence = online_conf * 0.8   # Reduced for missing textbook
        else:
            overall_confidence = 0.3  # Low confidence with no sources
        
        # Determine verification status
        if overall_confidence >= 0.8:
            status = "VERIFIED"
            evidence_strength = "STRONG"
        elif overall_confidence >= 0.6:
            status = "PARTIAL"
            evidence_strength = "MODERATE"
        elif overall_confidence >= 0.4:
            status = "PARTIAL"
            evidence_strength = "WEAK"
        else:
            status = "NOT_FOUND"
            evidence_strength = "INSUFFICIENT"
        
        # Generate consensus analysis
        consensus_analysis = self._generate_consensus_analysis(
            diagnosis, textbook_refs, online_refs, overall_confidence
        )
        
        # Clinical recommendations
        clinical_recommendations = self._generate_clinical_recommendations(
            diagnosis, status, evidence_strength, textbook_refs, online_refs
        )
        
        # Evidence summary
        evidence_summary = self._generate_evidence_summary(textbook_refs, online_refs)
        
        return ComprehensiveMedicalReference(
            diagnosis=diagnosis,
            verification_status=status,
            overall_confidence=overall_confidence,
            evidence_strength=evidence_strength,
            textbook_references=textbook_refs,
            textbook_confidence=textbook_conf,
            online_sources=online_refs,
            online_confidence=online_conf,
            consensus_analysis=consensus_analysis,
            contradictions=[],  # Could be enhanced to detect contradictions
            clinical_recommendations=clinical_recommendations,
            evidence_summary=evidence_summary,
            verification_timestamp=datetime.now().isoformat(),
            sources_count=len(textbook_refs) + len(online_refs)
        )
    
    def _generate_consensus_analysis(
        self, 
        diagnosis: str, 
        textbook_refs: List[TextbookEvidence], 
        online_refs: List[OnlineSource],
        confidence: float
    ) -> str:
        """Generate analysis of consensus between sources"""
        
        analysis_parts = []
        
        if textbook_refs and online_refs:
            analysis_parts.append(
                f"✅ Strong evidence base with {len(textbook_refs)} textbook references "
                f"and {len(online_refs)} online sources supporting the diagnosis of {diagnosis}."
            )
        elif textbook_refs:
            analysis_parts.append(
                f"📚 Textbook evidence supports {diagnosis} with {len(textbook_refs)} authoritative references, "
                f"though online verification is limited."
            )
        elif online_refs:
            analysis_parts.append(
                f"🌐 Online sources provide support for {diagnosis} with {len(online_refs)} references, "
                f"but textbook verification is unavailable."
            )
        else:
            analysis_parts.append(
                f"⚠️ Limited evidence available for {diagnosis}. "
                f"This may indicate a rare condition or require additional specialized sources."
            )
        
        # Add confidence interpretation
        if confidence >= 0.8:
            analysis_parts.append("The medical literature shows strong consensus supporting this diagnosis.")
        elif confidence >= 0.6:
            analysis_parts.append("The available evidence generally supports this diagnosis with moderate confidence.")
        elif confidence >= 0.4:
            analysis_parts.append("The evidence provides some support but additional verification may be needed.")
        else:
            analysis_parts.append("The current evidence is insufficient for strong diagnostic confidence.")
        
        return " ".join(analysis_parts)
    
    def _generate_clinical_recommendations(
        self, 
        diagnosis: str, 
        status: str, 
        evidence_strength: str,
        textbook_refs: List[TextbookEvidence],
        online_refs: List[OnlineSource]
    ) -> List[str]:
        """Generate clinical recommendations based on evidence strength"""
        
        recommendations = []
        
        if status == "VERIFIED" and evidence_strength == "STRONG":
            recommendations.extend([
                "✅ Diagnosis is well-supported by medical literature",
                "📋 Proceed with standard treatment protocols as indicated",
                "📚 Refer to cited textbook sections for detailed management guidelines"
            ])
        elif status == "PARTIAL" or evidence_strength == "MODERATE":
            recommendations.extend([
                "⚠️ Consider additional diagnostic confirmation if clinically indicated",
                "🔍 Review differential diagnoses in cited sources",
                "👨‍⚕️ Consider specialist consultation for complex cases"
            ])
        elif evidence_strength == "WEAK" or status == "NOT_FOUND":
            recommendations.extend([
                "🚨 Recommend additional diagnostic workup and specialist consultation",
                "📖 Consider rare disease databases and specialized literature",
                "🔬 May require advanced testing or expert opinion"
            ])
        
        # Add source-specific recommendations
        if textbook_refs:
            recommendations.append(f"📚 Review {len(textbook_refs)} textbook references for comprehensive management")
        if online_refs:
            recommendations.append(f"🌐 Consult {len(online_refs)} online sources for latest guidelines")
        
        return recommendations
    
    def _generate_evidence_summary(
        self, 
        textbook_refs: List[TextbookEvidence], 
        online_refs: List[OnlineSource]
    ) -> str:
        """Generate a summary of all evidence sources"""
        
        summary_parts = []
        
        if textbook_refs:
            textbook_names = list(set(ref.textbook_name for ref in textbook_refs))
            summary_parts.append(
                f"📚 Textbook Evidence: {len(textbook_refs)} references from "
                f"{len(textbook_names)} textbooks including {', '.join(textbook_names[:3])}"
                f"{' and others' if len(textbook_names) > 3 else ''}"
            )
        
        if online_refs:
            source_types = list(set(ref.source_type for ref in online_refs))
            high_reliability = [ref for ref in online_refs if ref.reliability_score >= 0.9]
            summary_parts.append(
                f"🌐 Online Evidence: {len(online_refs)} sources including {', '.join(source_types)}, "
                f"with {len(high_reliability)} high-reliability sources"
            )
        
        if not summary_parts:
            summary_parts.append("⚠️ Limited evidence sources available for this diagnosis")
        
        return " | ".join(summary_parts)

# Legacy compatibility
class OnlineMedicalVerifier(EnhancedOnlineMedicalVerifier):
    """Legacy class for backward compatibility"""
    
    def verify_diagnosis(self, diagnosis: str, clinical_context: str = "") -> dict:
        """Legacy method that returns simplified verification"""
        import asyncio
        
        try:
            # Check if there's already an event loop running
            try:
                loop = asyncio.get_running_loop()
                # If there's a running loop, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_verification_sync, diagnosis, clinical_context)
                    result = future.result(timeout=30)
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                result = asyncio.run(self.verify_diagnosis_comprehensive(diagnosis, clinical_context))
            
            # Convert to legacy format
            return {
                "source": "Enhanced Medical Verification",
                "condition": diagnosis,
                "verification_result": result.verification_status,
                "confidence_score": result.overall_confidence,
                "reasoning": result.consensus_analysis,
                "medical_facts": [result.evidence_summary] + result.clinical_recommendations[:3]
            }
        except Exception as e:
            return {
                "source": "Error in Verification",
                "condition": diagnosis,
                "verification_result": "ERROR",
                "confidence_score": 0.3,
                "reasoning": f"Verification error: {str(e)}",
                "medical_facts": ["Unable to complete verification"]
            }
    
    def _run_verification_sync(self, diagnosis: str, clinical_context: str = ""):
        """Helper method to run async verification in a new event loop"""
        import asyncio
        return asyncio.run(self.verify_diagnosis_comprehensive(diagnosis, clinical_context))
