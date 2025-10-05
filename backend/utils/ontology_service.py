"""
Enhanced Ontology Service for CortexMD
Integrates UMLS, SNOMED CT, and ICD-10 with Redis caching
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
import hashlib

logger = logging.getLogger(__name__)

class OntologyService:
    """Enhanced ontology service with UMLS, SNOMED, and ICD integration"""
    
    def __init__(self):
        self.umls_api_key = None
        self.base_urls = {
            'umls': 'https://uts-ws.nlm.nih.gov/rest',
            'snomed': 'https://browser.ihtsdotools.org/snowstorm/snomed-ct',
            'icd10': 'https://id.who.int/icd/release/10/2019'
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CortexMD/1.0 Medical Ontology Service'
        })
        
        # Load UMLS API key from environment or fallback configuration
        import os
        self.umls_api_key = os.getenv('UMLS_API_KEY')
        
        # Use fallback API key if environment variable not set
        if not self.umls_api_key:
            try:
                from config.fol_config import ONTOLOGY_CONFIG
                self.umls_api_key = ONTOLOGY_CONFIG.get('umls_api_key')
                if self.umls_api_key:
                    logger.info("Using fallback UMLS API key from configuration")
            except Exception as e:
                logger.warning(f"Could not load fallback UMLS API key: {e}")
        
        # Initialize Redis for caching
        try:
            from .enhanced_redis_service import enhanced_redis_service
            self.redis_service = enhanced_redis_service
        except Exception as e:
            logger.warning(f"Redis service unavailable for ontology caching: {e}")
            self.redis_service = None
    
    def normalize_medical_term(self, term: str) -> str:
        """Normalize medical term for consistent lookup"""
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r'\s+', ' ', term.strip().lower())
        
        # Remove common medical prefixes/suffixes that might interfere
        patterns_to_remove = [
            r'\b(acute|chronic|severe|mild|moderate)\s+',
            r'\s+(syndrome|disease|disorder|condition)$',
            r'\b(primary|secondary|tertiary)\s+',
        ]
        
        for pattern in patterns_to_remove:
            normalized = re.sub(pattern, '', normalized)
        
        return normalized.strip()
    
    def get_umls_concept(self, term: str) -> Optional[Dict[str, Any]]:
        """Get UMLS concept information for a medical term with intelligent search strategy"""
        if not self.umls_api_key:
            logger.warning("UMLS API key not configured")
            return None
        
        try:
            # Check cache first
            if self.redis_service:
                cached = self.redis_service.get_ontology_mapping(f"umls_{term}")
                if cached:
                    logger.info(f"Retrieved UMLS concept from cache: {term}")
                    return cached
            
            # Try multiple search strategies
            search_terms = self._generate_search_terms(term)
            
            for search_term in search_terms:
                logger.info(f"Trying UMLS search with term: '{search_term}'")
                umls_result = self._search_umls_with_term(search_term, term)
                if umls_result:
                    # Cache the result using original term
                    if self.redis_service:
                        self.redis_service.cache_ontology_mapping(f"umls_{term}", umls_result)
                    
                    logger.info(f"✅ Found UMLS concept: {umls_result['cui']} for '{term}' using search term '{search_term}'")
                    return umls_result
            
            logger.warning(f"No UMLS concept found for '{term}' after trying {len(search_terms)} search strategies")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving UMLS concept for '{term}': {e}")
            return None
    
    def _generate_search_terms(self, term: str) -> List[str]:
        """Generate multiple search term variations for better UMLS matching"""
        search_terms = []
        import re
        
        # 1. Original term
        search_terms.append(term)
        
        # 2. Normalized term
        normalized = self.normalize_medical_term(term)
        if normalized != term:
            search_terms.append(normalized)
        
        # 3. Remove parenthetical expressions (common issue)
        no_parens = re.sub(r'\s*\([^)]*\)', '', term).strip()
        if no_parens and no_parens not in search_terms:
            search_terms.append(no_parens)
        
        # 4. Remove qualifiers like "likely", "possible", "suspected"
        qualifiers = ['likely', 'possible', 'suspected', 'probable', 'acute', 'chronic', 'severe', 'mild', 'moderate']
        clean_term = term
        for qualifier in qualifiers:
            clean_term = re.sub(rf'\b{qualifier}\b', '', clean_term, flags=re.IGNORECASE).strip()
        clean_term = re.sub(r'\s+', ' ', clean_term)  # Remove extra spaces
        if clean_term and clean_term not in search_terms:
            search_terms.append(clean_term)
        
        # 5. Split complex terms and try main component
        if ' and ' in term:
            main_component = term.split(' and ')[0].strip()
            if main_component not in search_terms:
                search_terms.append(main_component)
        
        # 6. Handle "or" conjunctions - try first component
        if ' or ' in term:
            first_option = term.split(' or ')[0].strip()
            # Remove any trailing parentheses or qualifiers
            first_option = re.sub(r'\s*\([^)]*\).*$', '', first_option).strip()
            if first_option and first_option not in search_terms:
                search_terms.append(first_option)
        
        # 7. Remove grade specifications
        no_grade = re.sub(r'\s*(high-grade|low-grade|grade\s*\d+)', '', term, flags=re.IGNORECASE).strip()
        if no_grade and no_grade not in search_terms:
            search_terms.append(no_grade)
        
        # 8. Extract core medical condition (aggressive simplification)
        # Look for core patterns like "Tissue Sarcoma", "Carcinoma", etc.
        core_patterns = [
            r'\b\w*sarcoma\b',
            r'\b\w*carcinoma\b', 
            r'\b\w*adenocarcinoma\b',
            r'\b\w*lymphoma\b',
            r'\b\w*leukemia\b',
            r'\b\w*myeloma\b',
            r'\b\w*melanoma\b'
        ]
        
        for pattern in core_patterns:
            matches = re.findall(pattern, term, re.IGNORECASE)
            for match in matches:
                if match.lower() not in [t.lower() for t in search_terms]:
                    search_terms.append(match)
                    
                # Also try with "Soft Tissue" prefix if it's a sarcoma
                if 'sarcoma' in match.lower() and 'soft tissue' in term.lower():
                    soft_tissue_variant = f"Soft Tissue {match}"
                    if soft_tissue_variant not in search_terms:
                        search_terms.append(soft_tissue_variant)
        
        # 9. Try just the main anatomical location + condition type
        anatomical_locations = ['soft tissue', 'bone', 'brain', 'lung', 'breast', 'liver', 'kidney', 'prostate']
        condition_types = ['sarcoma', 'carcinoma', 'adenocarcinoma', 'tumor', 'cancer', 'neoplasm']
        
        term_lower = term.lower()
        for location in anatomical_locations:
            for condition in condition_types:
                if location in term_lower and condition in term_lower:
                    simple_term = f"{location} {condition}".title()
                    if simple_term not in search_terms:
                        search_terms.append(simple_term)
        
        # 10. Final fallback - try just the last significant word
        words = term.split()
        if len(words) > 1:
            significant_words = [w for w in words if len(w) > 4 and w.lower() not in ['likely', 'possible', 'suspected']]
            if significant_words:
                last_significant = significant_words[-1]
                if last_significant not in search_terms:
                    search_terms.append(last_significant)
        
        return search_terms
    
    def _search_umls_with_term(self, search_term: str, original_term: str) -> Optional[Dict[str, Any]]:
        """Perform actual UMLS search with a specific term"""
        try:
            search_url = f"{self.base_urls['umls']}/search/current"
            params = {
                'string': search_term,
                'apiKey': self.umls_api_key,
                'returnIdType': 'concept',
                'pageSize': 10
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('result', {}).get('results', [])
            
            if results:
                # Get the best match (first result)
                best_match = results[0]
                cui = best_match.get('ui')
                
                # Get detailed concept information
                concept_details = self._get_umls_concept_details(cui)
                
                # Calculate confidence based on how close the search term was to original
                base_confidence = self._calculate_match_confidence(original_term, best_match.get('name', ''))
                
                # Adjust confidence based on search term used
                if search_term == original_term:
                    confidence = base_confidence
                elif search_term in original_term:
                    confidence = base_confidence * 0.9  # Slight penalty for using simplified term
                else:
                    confidence = base_confidence * 0.8  # More penalty for heavily modified term
                
                umls_data = {
                    'cui': cui,
                    'name': best_match.get('name'),
                    'source': 'UMLS',
                    'confidence': min(confidence, 1.0),
                    'details': concept_details,
                    'search_term': original_term,
                    'normalized_term': search_term,
                    'retrieved_at': datetime.now().isoformat()
                }
                
                return umls_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching UMLS with term '{search_term}': {e}")
            return None
    
    def _get_umls_concept_details(self, cui: str) -> Dict[str, Any]:
        """Get detailed UMLS concept information"""
        try:
            details_url = f"{self.base_urls['umls']}/content/current/CUI/{cui}"
            params = {'apiKey': self.umls_api_key}
            
            response = self.session.get(details_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            result = data.get('result', {})
            
            return {
                'definition': result.get('definition'),
                'semantic_types': result.get('semanticTypes', []),
                'atoms': result.get('atoms', [])[:5],  # Limit to first 5 atoms
                'relations': result.get('relations', [])[:5]  # Limit to first 5 relations
            }
            
        except Exception as e:
            logger.error(f"Error getting UMLS concept details for {cui}: {e}")
            return {}
    
    def get_snomed_concept(self, term: str) -> Optional[Dict[str, Any]]:
        """Get SNOMED CT concept for a medical term"""
        try:
            # Check cache first
            if self.redis_service:
                cached = self.redis_service.get_snomed_codes(term)
                if cached:
                    logger.info(f"Retrieved SNOMED concept from cache: {term}")
                    return cached
            
            # Normalize term
            normalized_term = self.normalize_medical_term(term)
            
            # Search SNOMED CT International Edition
            search_url = f"{self.base_urls['snomed']}/MAIN/concepts"
            params = {
                'term': normalized_term,
                'activeFilter': True,
                'limit': 10,
                'expand': 'fsn,pt'
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            items = data.get('items', [])
            
            if items:
                # Get the best match
                best_match = items[0]
                concept_id = best_match.get('conceptId')
                
                snomed_data = {
                    'concept_id': concept_id,
                    'fsn': best_match.get('fsn', {}).get('term'),
                    'pt': best_match.get('pt', {}).get('term'),
                    'active': best_match.get('active'),
                    'module_id': best_match.get('moduleId'),
                    'source': 'SNOMED CT',
                    'confidence': self._calculate_match_confidence(term, best_match.get('pt', {}).get('term', '')),
                    'search_term': term,
                    'normalized_term': normalized_term,
                    'retrieved_at': datetime.now().isoformat()
                }
                
                # Cache the result
                if self.redis_service:
                    self.redis_service.cache_snomed_codes(term, snomed_data)
                
                logger.info(f"Retrieved SNOMED concept: {concept_id} for term '{term}'")
                return snomed_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving SNOMED concept for '{term}': {e}")
            return None
    
    def get_icd10_code(self, term: str) -> Optional[Dict[str, Any]]:
        """Get ICD-10 code for a medical term"""
        try:
            # Check cache first
            cache_key = f"icd10_{hashlib.md5(term.lower().encode()).hexdigest()}"
            if self.redis_service:
                cached = self.redis_service.get_ontology_mapping(cache_key)
                if cached:
                    logger.info(f"Retrieved ICD-10 code from cache: {term}")
                    return cached
            
            # For now, use a simplified ICD-10 mapping
            # In production, you would integrate with WHO ICD API
            icd10_mappings = self._get_common_icd10_mappings()
            
            normalized_term = self.normalize_medical_term(term)
            
            # Find best match
            best_match = None
            best_score = 0.0
            
            for icd_code, icd_data in icd10_mappings.items():
                for keyword in icd_data['keywords']:
                    score = self._calculate_match_confidence(normalized_term, keyword)
                    if score > best_score:
                        best_score = score
                        best_match = {
                            'code': icd_code,
                            'description': icd_data['description'],
                            'category': icd_data.get('category', ''),
                            'confidence': score
                        }
            
            if best_match and best_score > 0.5:  # Minimum confidence threshold
                icd10_data = {
                    **best_match,
                    'source': 'ICD-10',
                    'search_term': term,
                    'normalized_term': normalized_term,
                    'retrieved_at': datetime.now().isoformat()
                }
                
                # Cache the result
                if self.redis_service:
                    self.redis_service.cache_ontology_mapping(cache_key, icd10_data)
                
                logger.info(f"Retrieved ICD-10 code: {best_match['code']} for term '{term}'")
                return icd10_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving ICD-10 code for '{term}': {e}")
            return None
    
    def _get_common_icd10_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Get common ICD-10 mappings (simplified for demo)"""
        return {
            'J44.1': {
                'description': 'Chronic obstructive pulmonary disease with acute exacerbation',
                'keywords': ['copd', 'chronic obstructive pulmonary disease', 'emphysema', 'chronic bronchitis'],
                'category': 'Respiratory'
            },
            'I25.10': {
                'description': 'Atherosclerotic heart disease of native coronary artery without angina pectoris',
                'keywords': ['coronary artery disease', 'atherosclerotic heart disease', 'cad'],
                'category': 'Cardiovascular'
            },
            'E11.9': {
                'description': 'Type 2 diabetes mellitus without complications',
                'keywords': ['diabetes', 'type 2 diabetes', 'diabetes mellitus'],
                'category': 'Endocrine'
            },
            'M79.3': {
                'description': 'Panniculitis, unspecified',
                'keywords': ['panniculitis', 'inflammation', 'subcutaneous tissue'],
                'category': 'Musculoskeletal'
            },
            'R50.9': {
                'description': 'Fever, unspecified',
                'keywords': ['fever', 'pyrexia', 'elevated temperature'],
                'category': 'Symptoms'
            },
            'R06.02': {
                'description': 'Shortness of breath',
                'keywords': ['shortness of breath', 'dyspnea', 'breathing difficulty'],
                'category': 'Respiratory symptoms'
            },
            'R51': {
                'description': 'Headache',
                'keywords': ['headache', 'cephalgia', 'head pain'],
                'category': 'Neurological symptoms'
            }
        }
    
    def _calculate_match_confidence(self, original_term: str, matched_term: str) -> float:
        """Calculate confidence score for term matching"""
        if not original_term or not matched_term:
            return 0.0
        
        # Normalize both terms
        orig_norm = self.normalize_medical_term(original_term)
        match_norm = self.normalize_medical_term(matched_term)
        
        # Exact match
        if orig_norm == match_norm:
            return 1.0
        
        # Substring match
        if orig_norm in match_norm or match_norm in orig_norm:
            return 0.8
        
        # Word overlap
        orig_words = set(orig_norm.split())
        match_words = set(match_norm.split())
        
        if orig_words and match_words:
            overlap = len(orig_words.intersection(match_words))
            total_words = len(orig_words.union(match_words))
            return overlap / total_words
        
        return 0.0
    
    def get_comprehensive_ontology_mapping(self, term: str) -> Dict[str, Any]:
        """Get comprehensive ontology mapping from all sources"""
        try:
            # Check for comprehensive cache first
            cache_key = f"comprehensive_{hashlib.md5(term.lower().encode()).hexdigest()}"
            if self.redis_service:
                cached = self.redis_service.get_ontology_mapping(cache_key)
                if cached:
                    logger.info(f"Retrieved comprehensive ontology from cache: {term}")
                    return cached
            
            # Get mappings from all sources
            umls_data = self.get_umls_concept(term)
            snomed_data = self.get_snomed_concept(term)
            icd10_data = self.get_icd10_code(term)
            
            # Combine results
            comprehensive_mapping = {
                'original_term': term,
                'normalized_term': self.normalize_medical_term(term),
                'umls': umls_data,
                'snomed': snomed_data,
                'icd10': icd10_data,
                'mapping_completeness': self._calculate_mapping_completeness(umls_data, snomed_data, icd10_data),
                'best_match': self._determine_best_match(umls_data, snomed_data, icd10_data),
                'retrieved_at': datetime.now().isoformat()
            }
            
            # Cache comprehensive mapping
            if self.redis_service:
                self.redis_service.cache_ontology_mapping(cache_key, comprehensive_mapping)
            
            logger.info(f"Generated comprehensive ontology mapping for: {term}")
            return comprehensive_mapping
            
        except Exception as e:
            logger.error(f"Error generating comprehensive ontology mapping for '{term}': {e}")
            return {
                'original_term': term,
                'error': str(e),
                'retrieved_at': datetime.now().isoformat()
            }
    
    def _calculate_mapping_completeness(self, umls_data: Optional[Dict], 
                                      snomed_data: Optional[Dict], 
                                      icd10_data: Optional[Dict]) -> Dict[str, Any]:
        """Calculate completeness of ontology mapping"""
        sources_found = []
        total_confidence = 0.0
        
        if umls_data:
            sources_found.append('UMLS')
            total_confidence += umls_data.get('confidence', 0.0)
        
        if snomed_data:
            sources_found.append('SNOMED CT')
            total_confidence += snomed_data.get('confidence', 0.0)
        
        if icd10_data:
            sources_found.append('ICD-10')
            total_confidence += icd10_data.get('confidence', 0.0)
        
        return {
            'sources_found': sources_found,
            'source_count': len(sources_found),
            'average_confidence': total_confidence / max(len(sources_found), 1),
            'completeness_score': len(sources_found) / 3.0  # Out of 3 possible sources
        }
    
    def _determine_best_match(self, umls_data: Optional[Dict], 
                            snomed_data: Optional[Dict], 
                            icd10_data: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """Determine the best ontology match based on confidence scores"""
        candidates = []
        
        if umls_data:
            candidates.append(('UMLS', umls_data))
        if snomed_data:
            candidates.append(('SNOMED CT', snomed_data))
        if icd10_data:
            candidates.append(('ICD-10', icd10_data))
        
        if not candidates:
            return None
        
        # Sort by confidence score
        candidates.sort(key=lambda x: x[1].get('confidence', 0.0), reverse=True)
        
        best_source, best_data = candidates[0]
        return {
            'source': best_source,
            'confidence': best_data.get('confidence', 0.0),
            'primary_code': best_data.get('cui') or best_data.get('concept_id') or best_data.get('code'),
            'primary_name': best_data.get('name') or best_data.get('pt') or best_data.get('description'),
            'data': best_data
        }

# Global ontology service instance
ontology_service = OntologyService()
