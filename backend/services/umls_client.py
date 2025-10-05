"""
UMLS (Unified Medical Language System) API Client
Provides access to medical terminology and concept mapping services
"""
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

@dataclass
class UMLSSearchResult:
    """Represents a UMLS search result"""
    cui: str
    name: str
    semantic_types: List[str]
    definitions: List[str]
    synonyms: List[str]
    source_origins: List[str]
    score: float = 0.0

@dataclass
class UMLSConcept:
    """Represents a complete UMLS concept"""
    cui: str
    preferred_name: str
    semantic_types: List[str]
    definitions: List[str]
    synonyms: List[str]
    atoms: List[Dict]
    relations: List[Dict]
    source_atoms: Dict[str, List]

class UMLSClient:
    """Client for interacting with UMLS REST API"""

    def __init__(self, api_key: str, version: str = "current"):
        """
        Initialize UMLS client

        Args:
            api_key: UMLS API key
            version: UMLS version (e.g., "current", "2023AA")
        """
        self.api_key = api_key
        self.version = version
        self.base_url = "https://uts-ws.nlm.nih.gov/rest"
        self.session: Optional[aiohttp.ClientSession] = None
        self.tgt_token = None
        self.tgt_expires = 0
        self._session_lock = asyncio.Lock()

        # Rate limiting
        self.requests_per_second = 5
        self.last_request_time = 0
        self.request_interval = 1.0 / self.requests_per_second

        logger.info(f"Initialized UMLS client for version {version}")

    async def __aenter__(self):
        """Async context manager entry"""
        async with self._session_lock:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        async with self._session_lock:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None

    # UMLS uses simple API key authentication - no TGT/service tickets needed
    # The official documentation shows direct apiKey parameter usage

    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last)

        self.last_request_time = time.time()

    async def search_concepts(self, query: str, search_type: str = "words",
                            return_id_type: str = "concept", page_size: int = 10) -> List[UMLSSearchResult]:
        """
        Search for concepts in UMLS using direct API key authentication

        Args:
            query: Search term
            search_type: Type of search ("exact", "words", "leftTruncation", "rightTruncation", "normalizedString")
            return_id_type: Type of ID to return ("concept", "code", "sourceConcept", "sourceDescriptor")
            page_size: Number of results to return

        Returns:
            List of search results
        """
        # Use the simplified UMLS search endpoint with direct API key
        url = f"{self.base_url}/search/current"
        
        params = {
            'string': query,
            'apiKey': self.api_key,
            'pageNumber': 1,
            'pageSize': page_size
        }
        
        # Add optional parameters if specified
        if search_type != "exact":
            params['searchType'] = search_type
        if return_id_type != "concept":
            params['returnIdType'] = return_id_type
            
        # Focus on key medical vocabularies
        params['sabs'] = "SNOMEDCT_US,ICD10CM,MEDLINEPLUS"
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            logger.info(f"Searching UMLS for: {query}")
            async with self.session.get(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    # Parse the response according to the official UMLS API format
                    result_data = data.get('result', {})
                    items = result_data.get('results', [])
                    
                    logger.info(f"UMLS found {len(items)} results for query: {query}")
                    
                    for result in items:
                        search_result = UMLSSearchResult(
                            cui=result.get('ui', ''),
                            name=result.get('name', ''),
                            semantic_types=[],  # Will be populated by getting concept details
                            definitions=[],
                            synonyms=[],
                            source_origins=[result.get('rootSource', '')],
                            score=1.0  # UMLS doesn't provide scores in this endpoint
                        )
                        results.append(search_result)
                    
                    return results
                    
                elif response.status == 401:
                    logger.error("UMLS API authentication failed - invalid API key")
                    raise Exception("UMLS API authentication failed: Invalid API key")
                elif response.status == 429:
                    logger.warning("UMLS API rate limit exceeded")
                    raise Exception("UMLS API rate limit exceeded")
                else:
                    error_text = await response.text()
                    logger.error(f"UMLS API error {response.status}: {error_text}")
                    raise Exception(f"UMLS API error {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            logger.error("UMLS API request timeout")
            raise Exception("UMLS API request timeout")
        except Exception as e:
            logger.error(f"UMLS search failed for query '{query}': {str(e)}")
            raise Exception(f"UMLS search failed: {str(e)}")

    async def get_concept_details(self, cui: str) -> Optional[UMLSConcept]:
        """
        Get detailed information for a UMLS concept

        Args:
            cui: Concept Unique Identifier

        Returns:
            Detailed concept information or None if not found
        """
        try:
            await self._rate_limit()
            
            # Get basic concept information
            url = f"{self.base_url}/content/current/CUI/{cui}"
            params = {'apiKey': self.api_key}
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with self.session.get(url, params=params, timeout=timeout) as response:
                if response.status != 200:
                    logger.error(f"Failed to get concept details for CUI {cui}: HTTP {response.status}")
                    return None
                
                data = await response.json()
                result = data.get("result")
                
                if not result:
                    return None

                # Get semantic types
                semantic_types = []
                for st in result.get("semanticTypes", []):
                    semantic_types.append(st.get("name", ""))

                # Get definitions
                definitions = await self._get_concept_definitions(cui)

                # Get synonyms/atoms
                synonyms = await self._get_concept_atoms(cui)

                # Get relations
                relations = await self._get_concept_relations(cui)

                concept = UMLSConcept(
                    cui=cui,
                    preferred_name=result.get("name", ""),
                    semantic_types=semantic_types,
                    definitions=definitions,
                    synonyms=synonyms,
                    atoms=result.get("atoms", []),
                    relations=relations,
                    source_atoms={}
                )

                logger.info(f"Retrieved UMLS concept details for CUI: {cui}")
                return concept

        except Exception as e:
            logger.error(f"Failed to get UMLS concept details for CUI {cui}: {str(e)}")
            return None

    async def _get_concept_definitions(self, cui: str) -> List[str]:
        """Get definitions for a concept"""
        try:
            await self._rate_limit()
            
            url = f"{self.base_url}/content/current/CUI/{cui}/definitions"
            params = {'apiKey': self.api_key}
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with self.session.get(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    definitions = []
                    for definition in data.get("result", []):
                        definitions.append(definition.get("value", ""))
                    return definitions
                else:
                    return []

        except Exception:
            return []

    async def _get_concept_atoms(self, cui: str) -> List[str]:
        """Get atoms (synonyms) for a concept"""
        try:
            await self._rate_limit()
            
            url = f"{self.base_url}/content/current/CUI/{cui}/atoms"
            params = {
                'apiKey': self.api_key,
                'sabs': 'SNOMEDCT_US,ICD10CM,RXNORM'
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with self.session.get(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    synonyms = []
                    for atom in data.get("result", []):
                        name = atom.get("name", "")
                        if name and name not in synonyms:
                            synonyms.append(name)
                    return synonyms
                else:
                    return []

        except Exception:
            return []

    async def _get_concept_relations(self, cui: str) -> List[Dict]:
        """Get relations for a concept"""
        try:
            await self._rate_limit()
            
            url = f"{self.base_url}/content/current/CUI/{cui}/relations"
            params = {'apiKey': self.api_key}
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with self.session.get(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    relations = []
                    for relation in data.get("result", []):
                        relations.append({
                            "relation": relation.get("relationLabel", ""),
                            "related_cui": relation.get("relatedId", ""),
                            "related_name": relation.get("relatedIdName", ""),
                            "source": relation.get("rootSource", "")
                        })
                    return relations
                else:
                    return []

        except Exception:
            return []

    async def get_semantic_types(self) -> Dict[str, str]:
        """Get all available semantic types"""
        try:
            await self._rate_limit()
            
            url = f"{self.base_url}/semantic-network/current/TUI"
            params = {'apiKey': self.api_key}
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with self.session.get(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    semantic_types = {}
                    for st in data.get("result", []):
                        semantic_types[st.get("abbreviation", "")] = st.get("name", "")
                    return semantic_types
                else:
                    return {}

        except Exception as e:
            logger.error(f"Failed to get semantic types: {str(e)}")
            return {}

   
    async def get_concept_hierarchy(self, cui: str, direction: str = "children") -> List[Dict]:
        """
        Get concept hierarchy (parents or children)

        Args:
            cui: Concept Unique Identifier
            direction: "parents" or "children"
        """
        try:
            await self._rate_limit()
            
            if direction == "parents":
                url = f"{self.base_url}/content/current/CUI/{cui}/parents"
            else:
                url = f"{self.base_url}/content/current/CUI/{cui}/children"

            params = {'apiKey': self.api_key}
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with self.session.get(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    hierarchy = []
                    for item in data.get("result", []):
                        hierarchy.append({
                            "cui": item.get("ui", ""),
                            "name": item.get("name", ""),
                            "relation": direction
                        })
                    return hierarchy
                else:
                    return []

        except Exception as e:
            logger.error(f"Failed to get {direction} for CUI {cui}: {str(e)}")
            return []

