"""
Simplified UMLS Code Lookup Service
Provides basic code lookup functionality only
"""
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import time
from pathlib import Path
import csv

from .umls_client import UMLSClient, UMLSSearchResult, UMLSConcept

logger = logging.getLogger(__name__)

@dataclass
class CodeLookupResult:
    """Result of code lookup operation"""
    code: str
    source_vocabulary: str
    cui: str
    name: str
    uri: str
    semantic_types: List[str]
    definitions: List[str]
    synonyms: List[str]
    relations: List[Dict]
    success: bool
    error_message: Optional[str] = None

@dataclass
class BatchLookupResult:
    """Result of batch code lookup operation"""
    total_codes: int
    successful_lookups: int
    failed_lookups: int
    results: List[CodeLookupResult]
    execution_time: float
    # Graph functionality removed
    graph_nodes_added: int = 0
    graph_relationships_added: int = 0

class UMLSCodeLookupService:
    """Simplified service for UMLS code lookups - lookup only"""

    def __init__(self, api_key: str, neo4j_service=None):
        """
        Initialize the code lookup service

        Args:
            api_key: UMLS API key
            neo4j_service: Ignored - kept for compatibility
        """
        self.api_key = api_key
        self.umls_client = UMLSClient(api_key)
        # Neo4j functionality removed
        self.session = None
        
        # Rate limiting for batch operations
        self.batch_delay = 0.2  # 200ms between requests
        
        logger.info("Initialized UMLS Code Lookup Service (lookup only mode)")

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        await self.umls_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        await self.umls_client.__aexit__(exc_type, exc_val, exc_tb)

    async def lookup_code(self, code: str, source_vocabulary: str) -> CodeLookupResult:
        """
        Lookup a single code in UMLS

        Args:
            code: The code to lookup
            source_vocabulary: Source vocabulary (e.g., SNOMEDCT_US, ICD10CM, RXNORM)

        Returns:
            CodeLookupResult with concept information
        """
        try:
            logger.info(f"Looking up code: {code} in vocabulary: {source_vocabulary}")
            
            # Search for the code using UMLS search API
            search_results = await self._search_code_in_vocabulary(code, source_vocabulary)
            
            if not search_results:
                return CodeLookupResult(
                    code=code,
                    source_vocabulary=source_vocabulary,
                    cui="",
                    name="",
                    uri="",
                    semantic_types=[],
                    definitions=[],
                    synonyms=[],
                    relations=[],
                    success=False,
                    error_message=f"No results found for code {code}"
                )

            # Get the first (best) result
            best_result = search_results[0]
            
            # Get detailed concept information
            concept_details = await self.umls_client.get_concept_details(best_result.cui)
            
            if concept_details:
                result = CodeLookupResult(
                    code=code,
                    source_vocabulary=source_vocabulary,
                    cui=best_result.cui,
                    name=concept_details.preferred_name,
                    uri=f"https://uts.nlm.nih.gov/uts/umls/concept/{best_result.cui}",
                    semantic_types=concept_details.semantic_types,
                    definitions=concept_details.definitions,
                    synonyms=concept_details.synonyms,
                    relations=concept_details.relations,
                    success=True
                )
                
                # Graph functionality removed
                
                return result
            else:
                return CodeLookupResult(
                    code=code,
                    source_vocabulary=source_vocabulary,
                    cui=best_result.cui,
                    name=best_result.name,
                    uri=f"https://uts.nlm.nih.gov/uts/umls/concept/{best_result.cui}",
                    semantic_types=best_result.semantic_types,
                    definitions=[],
                    synonyms=[],
                    relations=[],
                    success=True
                )

        except Exception as e:
            logger.error(f"Error looking up code {code}: {str(e)}")
            return CodeLookupResult(
                code=code,
                source_vocabulary=source_vocabulary,
                cui="",
                name="",
                uri="",
                semantic_types=[],
                definitions=[],
                synonyms=[],
                relations=[],
                success=False,
                error_message=str(e)
            )

    async def _search_code_in_vocabulary(self, code: str, source_vocabulary: str) -> List[UMLSSearchResult]:
        """Search for a specific code in a vocabulary"""
        
        # Use the UMLS search endpoint with sourceUI input type
        url = f"{self.umls_client.base_url}/search/current"
        
        params = {
            'string': code,
            'apiKey': self.api_key,
            'inputType': 'sourceUI',  # Search by source code
            'rootSource': source_vocabulary,
            'pageNumber': 1,
            'pageSize': 10
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            async with self.session.get(url, params=params, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    result_data = data.get('result', {})
                    items = result_data.get('results', [])
                    
                    for result in items:
                        search_result = UMLSSearchResult(
                            cui=result.get('ui', ''),
                            name=result.get('name', ''),
                            semantic_types=[],
                            definitions=[],
                            synonyms=[],
                            source_origins=[result.get('rootSource', '')],
                            score=1.0
                        )
                        results.append(search_result)
                    
                    return results
                else:
                    logger.error(f"UMLS search failed with status {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching for code {code}: {str(e)}")
            return []

    async def lookup_codes_from_file(self, file_path: str, source_vocabulary: str, 
                                   output_file: str = None) -> BatchLookupResult:
        """
        Lookup codes from a text file (one code per line)

        Args:
            file_path: Path to input file with codes
            source_vocabulary: Source vocabulary for all codes
            output_file: Optional output file path

        Returns:
            BatchLookupResult with all lookup results
        """
        start_time = time.time()
        
        try:
            # Read codes from file
            codes = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.isspace():
                        codes.append(line)
            
            logger.info(f"Processing {len(codes)} codes from {file_path}")
            
            # Process codes
            results = []
            # Graph tracking removed
            
            for i, code in enumerate(codes):
                logger.info(f"Processing code {i+1}/{len(codes)}: {code}")
                
                result = await self.lookup_code(code, source_vocabulary)
                results.append(result)
                
                # Graph functionality removed
                
                # Rate limiting
                if i < len(codes) - 1:
                    await asyncio.sleep(self.batch_delay)
            
            # Create output file if specified
            if output_file:
                await self._write_results_to_file(results, output_file)
            
            execution_time = time.time() - start_time
            successful_lookups = sum(1 for r in results if r.success)
            
            batch_result = BatchLookupResult(
                total_codes=len(codes),
                successful_lookups=successful_lookups,
                failed_lookups=len(codes) - successful_lookups,
                results=results,
                execution_time=execution_time,
                # Graph functionality removed
                graph_nodes_added=0,
                graph_relationships_added=0
            )
            
            logger.info(f"Batch lookup completed: {successful_lookups}/{len(codes)} successful")
            return batch_result
            
        except Exception as e:
            logger.error(f"Error in batch lookup: {str(e)}")
            raise

    async def _write_results_to_file(self, results: List[CodeLookupResult], output_file: str):
        """Write lookup results to output file"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"SEARCH CODE: {result.code}\n\n")
                
                if result.success:
                    f.write(f"CUI: {result.cui}\n")
                    f.write(f"Name: {result.name}\n")
                    f.write(f"URI: {result.uri}\n")
                    f.write(f"Source Vocabulary: {result.source_vocabulary}\n")
                    f.write(f"Code: {result.code}\n")
                    
                    if result.semantic_types:
                        f.write(f"Semantic Types: {', '.join(result.semantic_types)}\n")
                    
                    if result.definitions:
                        f.write(f"Definitions:\n")
                        for definition in result.definitions:
                            f.write(f"  - {definition}\n")
                    
                    if result.synonyms:
                        f.write(f"Synonyms: {', '.join(result.synonyms[:5])}\n")  # Limit to first 5
                    
                    f.write("\n")
                else:
                    f.write(f"No results found.\n")
                    if result.error_message:
                        f.write(f"Error: {result.error_message}\n")
                    f.write("\n")
                
                f.write("***\n\n")

    async def _add_concept_to_graph(self, concept: UMLSConcept, original_code: str, vocabulary: str):
        """Add concept to graph - disabled"""
        logger.info(f"Graph functionality disabled for concept {concept.cui}")
        return

    def _get_primary_label(self, semantic_types: List[str]) -> Optional[str]:
        """Determine primary label - disabled"""
        return None

    async def get_concept_details_for_popup(self, cui: str) -> Dict[str, Any]:
        """
        Get basic concept information - simplified without graph data

        Args:
            cui: Concept Unique Identifier

        Returns:
            Dictionary with basic concept details
        """
        try:
            concept = await self.umls_client.get_concept_details(cui)
            
            if not concept:
                return {"error": "Concept not found"}
            
            return {
                "cui": concept.cui,
                "preferred_name": concept.preferred_name,
                "semantic_types": concept.semantic_types,
                "definitions": concept.definitions,
                "synonyms": concept.synonyms[:10],  # Limit for UI
                "relations": concept.relations[:10],  # Limit for UI
                "related_concepts": [],  # Empty - no graph lookup
                "umls_uri": f"https://uts.nlm.nih.gov/uts/umls/concept/{cui}",
                "graph_visualization_data": {}  # Empty - no graph data
            }
            
        except Exception as e:
            logger.error(f"Error getting concept details: {str(e)}")
            return {"error": str(e)}

    async def _get_related_concepts_from_graph(self, cui: str) -> List[Dict]:
        """Get related concepts from Neo4j graph - disabled"""
        logger.info(f"Graph functionality disabled for CUI: {cui}")
        return []

    async def _get_graph_visualization_data(self, cui: str) -> Dict[str, Any]:
        """Get data for graph visualization in popup - disabled"""
        logger.info(f"Graph visualization disabled for CUI: {cui}")
        return {}

    async def search_concepts_with_popup_data(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for concepts - simplified without popup data

        Args:
            query: Search term
            max_results: Maximum number of results

        Returns:
            List of basic concept dictionaries
        """
        try:
            search_results = await self.umls_client.search_concepts(
                query=query, 
                page_size=max_results
            )
            
            simplified_results = []
            for result in search_results:
                # Return basic concept info without popup data
                simplified_results.append({
                    "cui": result.cui,
                    "name": result.name,
                    "semantic_types": result.semantic_types,
                    "search_score": result.score,
                    "definitions": [],  # Empty for simplicity
                    "related_concepts": []  # Empty - no graph lookup
                })
            
            return simplified_results
            
        except Exception as e:
            logger.error(f"Error in search concepts: {str(e)}")
            return []
