#!/usr/bin/env python3
"""
🌐 Enhanced Online Medical Verification System
Multi-engine search with robust fallbacks and proper citations
"""

import requests
import json
import asyncio
import aiohttp
import time
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse, quote_plus, unquote
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MedicalSource:
    """Enhanced medical source with citation information"""
    title: str
    url: str
    content_snippet: str
    domain: str
    relevance_score: float
    credibility_score: float
    source_type: str
    citation_format: str
    publication_date: Optional[str] = None
    authors: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

@dataclass
class MedicalVerificationResult:
    """Enhanced verification result with citations"""
    verification_status: str
    confidence_score: float
    sources: List[MedicalSource]
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    clinical_notes: str
    verification_summary: str
    timestamp: str
    search_strategies_used: List[str]
    citations: List[str]
    bibliography: List[str]

class EnhancedOnlineVerifier:
    """Enhanced online medical verification with multiple search engines"""
    
    def __init__(self):
        self.session = requests.Session()
        
        # Rotate through different user agents to avoid blocking
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
        
        # Update session headers
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1'
        })
        
        # Trusted medical domains with enhanced credibility scores
        self.trusted_domains = {
            # Tier 1 - Highest credibility (0.90-1.00)
            'nih.gov': 0.98,
            'cdc.gov': 0.96,
            'who.int': 0.95,
            'ncbi.nlm.nih.gov': 0.97,
            'pubmed.ncbi.nlm.nih.gov': 0.96,
            'medlineplus.gov': 0.92,
            'nci.nih.gov': 0.94,
            
            # Tier 2 - High credibility (0.80-0.89)
            'mayoclinic.org': 0.89,
            'clevelandclinic.org': 0.87,
            'hopkinsmedicine.org': 0.88,
            'uptodate.com': 0.86,
            'bmj.com': 0.85,
            'thelancet.com': 0.87,
            'nejm.org': 0.88,
            'jama.jamanetwork.com': 0.86,
            'nature.com': 0.85,
            'science.org': 0.84,
            
            # Tier 3 - Good credibility (0.70-0.79)
            'webmd.com': 0.78,
            'healthline.com': 0.76,
            'medicalnewstoday.com': 0.75,
            'medscape.com': 0.77,
            'drugs.com': 0.74,
            'rxlist.com': 0.73,
            'patient.info': 0.72,
            'emedicine.medscape.com': 0.78,
            
            # Tier 4 - Moderate credibility (0.60-0.69)
            'medicine.net': 0.68,
            'everydayhealth.com': 0.65,
            'healthgrades.com': 0.67,
            'verywell.com': 0.66
        }
        
        # Medical journals and databases
        self.journal_domains = {
            'pubmed.ncbi.nlm.nih.gov': 0.96,
            'bmj.com': 0.94,
            'thelancet.com': 0.95,
            'nejm.org': 0.96,
            'jama.jamanetwork.com': 0.94,
            'nature.com': 0.93,
            'science.org': 0.92,
            'cell.com': 0.91,
            'sciencedirect.com': 0.88,
            'springer.com': 0.87
        }
    
    def verify_diagnosis_online(self, diagnosis: str, symptoms: List[str] = None, 
                              patient_age: int = None, patient_gender: str = None) -> MedicalVerificationResult:
        """Enhanced medical verification using multiple search strategies"""
        
        print(f"🩺 ENHANCED ONLINE MEDICAL VERIFICATION: {diagnosis}")
        print("=" * 70)
        
        sources = []
        search_strategies = []
        verification_attempts = []
        
        # Strategy 1: Multi-engine web search
        try:
            print("🔍 Strategy 1: Multi-engine web search...")
            web_sources = self._multi_engine_search(diagnosis, symptoms)
            sources.extend(web_sources)
            search_strategies.append("multi_engine_search")
            verification_attempts.append(f"Multi-engine search: {len(web_sources)} sources")
            print(f"  ✅ Multi-engine search found {len(web_sources)} sources")
        except Exception as e:
            print(f"  ❌ Multi-engine search failed: {e}")
            verification_attempts.append(f"Multi-engine search: FAILED ({str(e)[:50]}...)")
        
        # Strategy 2: Direct medical database search
        try:
            print("🔍 Strategy 2: Direct medical database search...")
            db_sources = self._search_medical_databases(diagnosis, symptoms)
            sources.extend(db_sources)
            search_strategies.append("medical_database_search")
            verification_attempts.append(f"Medical database: {len(db_sources)} sources")
            print(f"  ✅ Medical database search found {len(db_sources)} sources")
        except Exception as e:
            print(f"  ❌ Medical database search failed: {e}")
            verification_attempts.append(f"Medical database: FAILED ({str(e)[:50]}...)")
        
        # Strategy 3: Academic journal search
        try:
            print("🔍 Strategy 3: Academic journal search...")
            journal_sources = self._search_academic_journals(diagnosis, symptoms)
            sources.extend(journal_sources)
            search_strategies.append("academic_journal_search")
            verification_attempts.append(f"Academic journals: {len(journal_sources)} sources")
            print(f"  ✅ Academic journal search found {len(journal_sources)} sources")
        except Exception as e:
            print(f"  ❌ Academic journal search failed: {e}")
            verification_attempts.append(f"Academic journals: FAILED ({str(e)[:50]}...)")
        
        # Strategy 4: Wikipedia medical search (as fallback)
        if len(sources) < 3:
            try:
                print("🔍 Strategy 4: Wikipedia medical search...")
                wiki_sources = self._search_wikipedia_medical(diagnosis, symptoms)
                sources.extend(wiki_sources)
                search_strategies.append("wikipedia_search")
                verification_attempts.append(f"Wikipedia: {len(wiki_sources)} sources")
                print(f"  ✅ Wikipedia search found {len(wiki_sources)} sources")
            except Exception as e:
                print(f"  ❌ Wikipedia search failed: {e}")
                verification_attempts.append(f"Wikipedia: FAILED ({str(e)[:50]}...)")
        
        # Remove duplicates and sort by credibility
        sources = self._deduplicate_sources(sources)
        sources.sort(key=lambda x: (x.credibility_score, x.relevance_score), reverse=True)
        
        # Generate evidence analysis
        supporting_evidence, contradicting_evidence = self._analyze_evidence(sources, diagnosis, symptoms)
        
        # Calculate confidence and status
        confidence_score, verification_status = self._calculate_verification_metrics(sources, diagnosis)
        
        # Generate citations and bibliography
        citations = [source.citation_format for source in sources[:10]]
        bibliography = self._generate_bibliography(sources[:10])
        
        # Generate clinical notes
        clinical_notes = self._generate_clinical_notes(
            diagnosis, sources, patient_age, patient_gender, verification_attempts
        )
        
        # Generate summary
        summary = self._generate_verification_summary(
            diagnosis, verification_status, confidence_score, len(sources)
        )
        
        print(f"✅ Verification completed: {verification_status} (Confidence: {confidence_score:.3f})")
        print(f"📚 Total sources found: {len(sources)}")
        print(f"🔧 Strategies used: {', '.join(search_strategies)}")
        
        return MedicalVerificationResult(
            verification_status=verification_status,
            confidence_score=confidence_score,
            sources=sources[:15],  # Top 15 sources
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            clinical_notes=clinical_notes,
            verification_summary=summary,
            timestamp=datetime.now().isoformat(),
            search_strategies_used=search_strategies,
            citations=citations,
            bibliography=bibliography
        )
    
    def _multi_engine_search(self, diagnosis: str, symptoms: List[str] = None) -> List[MedicalSource]:
        """Search using multiple search engines with fallbacks"""
        
        sources = []
        
        # Search engines to try (in order of preference)
        search_engines = [
            self._search_duckduckgo_html,
            self._search_startpage,
            self._search_bing,
            self._search_searx
        ]
        
        for search_func in search_engines:
            try:
                engine_sources = search_func(diagnosis, symptoms)
                sources.extend(engine_sources)
                
                # If we have enough good sources, we can stop
                if len([s for s in sources if s.credibility_score > 0.7]) >= 5:
                    break
                    
            except Exception as e:
                logger.warning(f"Search engine {search_func.__name__} failed: {e}")
                continue
        
        return sources
    
    def _search_duckduckgo_html(self, diagnosis: str, symptoms: List[str] = None) -> List[MedicalSource]:
        """Enhanced DuckDuckGo search using HTML interface"""
        
        sources = []
        
        try:
            # Create enhanced search query
            query_parts = [diagnosis, "medical condition"]
            if symptoms:
                query_parts.extend(symptoms[:2])  # Add top 2 symptoms
            query_parts.extend(["symptoms", "treatment", "diagnosis"])
            
            query = " ".join(query_parts)
            
            # Use DuckDuckGo HTML interface (more reliable than lite)
            search_url = f"https://html.duckduckgo.com/html?q={quote_plus(query)}"
            
            # Rotate user agent for this request
            headers = self.session.headers.copy()
            headers['User-Agent'] = random.choice(self.user_agents)
            
            print(f"📄 DuckDuckGo HTML search: {search_url}")
            
            response = self.session.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find search results
            result_links = soup.find_all('a', class_='result__a')
            
            for link in result_links[:10]:  # Process top 10 results
                try:
                    title = link.get_text().strip()
                    href = link.get('href', '')
                    
                    if not href or 'duckduckgo.com' in href:
                        continue
                    
                    # Clean up the URL
                    actual_url = self._clean_search_url(href)
                    if not actual_url:
                        continue
                    
                    domain = urlparse(actual_url).netloc.lower()
                    
                    # Check if it's a trusted medical domain
                    credibility = self._get_domain_credibility(domain)
                    if credibility < 0.5:  # Skip low credibility sources
                        continue
                    
                    # Extract content
                    content = self._extract_page_content(actual_url)
                    if not content or len(content) < 100:
                        continue
                    
                    relevance = self._calculate_relevance(content, diagnosis, symptoms)
                    
                    source = MedicalSource(
                        title=title[:150],
                        url=actual_url,
                        content_snippet=content[:300] + "...",
                        domain=domain,
                        relevance_score=relevance,
                        credibility_score=credibility,
                        source_type="web_search",
                        citation_format=self._generate_citation(title, domain, actual_url),
                        keywords=self._extract_keywords(content, diagnosis)
                    )
                    
                    sources.append(source)
                    print(f"  ✅ {domain} (Cred: {credibility:.2f}, Rel: {relevance:.2f})")
                    
                    if len(sources) >= 8:  # Limit DuckDuckGo results
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing DuckDuckGo result: {e}")
                    continue
            
            print(f"📊 DuckDuckGo HTML found {len(sources)} medical sources")
            
        except Exception as e:
            logger.error(f"DuckDuckGo HTML search error: {e}")
            raise
        
        return sources
    
    def _search_startpage(self, diagnosis: str, symptoms: List[str] = None) -> List[MedicalSource]:
        """Search using Startpage (Google proxy)"""
        
        sources = []
        
        try:
            query = f"{diagnosis} medical condition"
            if symptoms:
                query += f" {' '.join(symptoms[:2])}"
            
            search_url = f"https://www.startpage.com/sp/search?query={quote_plus(query)}&cat=web&pl=opensearch"
            
            headers = self.session.headers.copy()
            headers['User-Agent'] = random.choice(self.user_agents)
            
            response = self.session.get(search_url, headers=headers, timeout=15)
            if response.status_code != 200:
                return sources
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find result links
            for result in soup.find_all('a', class_='w-gl__result-title')[:8]:
                try:
                    title = result.get_text().strip()
                    href = result.get('href', '')
                    
                    if not href:
                        continue
                    
                    actual_url = urljoin("https://www.startpage.com", href) if href.startswith('/') else href
                    domain = urlparse(actual_url).netloc.lower()
                    
                    credibility = self._get_domain_credibility(domain)
                    if credibility < 0.6:
                        continue
                    
                    content = self._extract_page_content(actual_url)
                    if not content or len(content) < 100:
                        continue
                    
                    relevance = self._calculate_relevance(content, diagnosis, symptoms)
                    
                    source = MedicalSource(
                        title=title[:150],
                        url=actual_url,
                        content_snippet=content[:300] + "...",
                        domain=domain,
                        relevance_score=relevance,
                        credibility_score=credibility,
                        source_type="web_search",
                        citation_format=self._generate_citation(title, domain, actual_url),
                        keywords=self._extract_keywords(content, diagnosis)
                    )
                    
                    sources.append(source)
                    
                except Exception as e:
                    logger.warning(f"Error processing Startpage result: {e}")
                    continue
            
            print(f"📊 Startpage found {len(sources)} sources")
            
        except Exception as e:
            logger.error(f"Startpage search error: {e}")
            raise
        
        return sources
    
    def _search_bing(self, diagnosis: str, symptoms: List[str] = None) -> List[MedicalSource]:
        """Search using Bing"""
        
        sources = []
        
        try:
            query = f"{diagnosis} medical condition site:mayoclinic.org OR site:webmd.com OR site:healthline.com OR site:nih.gov"
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
            
            headers = self.session.headers.copy()
            headers['User-Agent'] = random.choice(self.user_agents)
            
            response = self.session.get(search_url, headers=headers, timeout=15)
            if response.status_code != 200:
                return sources
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find organic results
            for result in soup.find_all('h2')[:6]:
                try:
                    link = result.find('a')
                    if not link:
                        continue
                    
                    title = link.get_text().strip()
                    href = link.get('href', '')
                    
                    if not href or 'bing.com' in href:
                        continue
                    
                    domain = urlparse(href).netloc.lower()
                    credibility = self._get_domain_credibility(domain)
                    
                    if credibility < 0.7:  # Higher threshold for Bing
                        continue
                    
                    content = self._extract_page_content(href)
                    if not content or len(content) < 100:
                        continue
                    
                    relevance = self._calculate_relevance(content, diagnosis, symptoms)
                    
                    source = MedicalSource(
                        title=title[:150],
                        url=href,
                        content_snippet=content[:300] + "...",
                        domain=domain,
                        relevance_score=relevance,
                        credibility_score=credibility,
                        source_type="web_search",
                        citation_format=self._generate_citation(title, domain, href),
                        keywords=self._extract_keywords(content, diagnosis)
                    )
                    
                    sources.append(source)
                    
                except Exception as e:
                    logger.warning(f"Error processing Bing result: {e}")
                    continue
            
            print(f"📊 Bing found {len(sources)} sources")
            
        except Exception as e:
            logger.error(f"Bing search error: {e}")
            raise
        
        return sources
    
    def _search_searx(self, diagnosis: str, symptoms: List[str] = None) -> List[MedicalSource]:
        """Search using SearX instances"""
        
        sources = []
        
        # Public SearX instances
        searx_instances = [
            "https://searx.be",
            "https://search.sapti.me",
            "https://searx.ninja"
        ]
        
        for instance in searx_instances:
            try:
                query = f"{diagnosis} medical"
                search_url = f"{instance}/search?q={quote_plus(query)}&format=json"
                
                headers = self.session.headers.copy()
                headers['User-Agent'] = random.choice(self.user_agents)
                
                response = self.session.get(search_url, headers=headers, timeout=10)
                if response.status_code != 200:
                    continue
                
                data = response.json()
                results = data.get('results', [])
                
                for result in results[:5]:
                    try:
                        title = result.get('title', '')
                        url = result.get('url', '')
                        content = result.get('content', '')
                        
                        if not url or not title:
                            continue
                        
                        domain = urlparse(url).netloc.lower()
                        credibility = self._get_domain_credibility(domain)
                        
                        if credibility < 0.6:
                            continue
                        
                        relevance = self._calculate_relevance(content, diagnosis, symptoms)
                        
                        source = MedicalSource(
                            title=title[:150],
                            url=url,
                            content_snippet=content[:300] + "...",
                            domain=domain,
                            relevance_score=relevance,
                            credibility_score=credibility,
                            source_type="web_search",
                            citation_format=self._generate_citation(title, domain, url),
                            keywords=self._extract_keywords(content, diagnosis)
                        )
                        
                        sources.append(source)
                        
                    except Exception as e:
                        logger.warning(f"Error processing SearX result: {e}")
                        continue
                
                if sources:  # If we got results from this instance, don't try others
                    break
                    
            except Exception as e:
                logger.warning(f"SearX instance {instance} failed: {e}")
                continue
        
        print(f"📊 SearX found {len(sources)} sources")
        return sources
    
    def _search_medical_databases(self, diagnosis: str, symptoms: List[str] = None) -> List[MedicalSource]:
        """Search specific medical databases and websites"""
        
        sources = []
        
        # Direct searches on trusted medical sites
        medical_sites = [
            {
                'name': 'Mayo Clinic',
                'base_url': 'https://www.mayoclinic.org',
                'search_path': '/search/search-results?q={query}',
                'credibility': 0.89
            },
            {
                'name': 'MedlinePlus',
                'base_url': 'https://medlineplus.gov',
                'search_path': '/search?query={query}',
                'credibility': 0.92
            },
            {
                'name': 'WebMD',
                'base_url': 'https://www.webmd.com',
                'search_path': '/search/search_results/default.aspx?query={query}',
                'credibility': 0.78
            }
        ]
        
        for site in medical_sites:
            try:
                query = f"{diagnosis} symptoms treatment"
                search_url = site['base_url'] + site['search_path'].format(query=quote_plus(query))
                
                headers = self.session.headers.copy()
                headers['User-Agent'] = random.choice(self.user_agents)
                
                response = self.session.get(search_url, headers=headers, timeout=15)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find relevant links (site-specific selectors)
                links = []
                if 'mayoclinic' in site['base_url']:
                    links = soup.find_all('a', class_='search-result-link')
                elif 'medlineplus' in site['base_url']:
                    links = soup.find_all('a', href=True)
                elif 'webmd' in site['base_url']:
                    links = soup.find_all('a', class_='search-results-title')
                
                for link in links[:3]:  # Top 3 from each site
                    try:
                        title = link.get_text().strip()
                        href = link.get('href', '')
                        
                        if not href or not title:
                            continue
                        
                        if href.startswith('/'):
                            href = site['base_url'] + href
                        
                        # Filter for medical content
                        if not any(term in title.lower() for term in [diagnosis.lower(), 'condition', 'disease', 'symptom']):
                            continue
                        
                        content = self._extract_page_content(href)
                        if not content or len(content) < 150:
                            continue
                        
                        relevance = self._calculate_relevance(content, diagnosis, symptoms)
                        
                        source = MedicalSource(
                            title=title[:150],
                            url=href,
                            content_snippet=content[:350] + "...",
                            domain=urlparse(href).netloc.lower(),
                            relevance_score=relevance,
                            credibility_score=site['credibility'],
                            source_type="medical_database",
                            citation_format=self._generate_citation(title, site['name'], href),
                            keywords=self._extract_keywords(content, diagnosis)
                        )
                        
                        sources.append(source)
                        
                    except Exception as e:
                        logger.warning(f"Error processing {site['name']} result: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error searching {site['name']}: {e}")
                continue
        
        print(f"📊 Medical databases found {len(sources)} sources")
        return sources
    
    def _search_academic_journals(self, diagnosis: str, symptoms: List[str] = None) -> List[MedicalSource]:
        """Search academic medical journals"""
        
        sources = []
        
        try:
            # PubMed search
            query = f"{diagnosis} AND (diagnosis OR treatment OR symptoms)"
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={quote_plus(query)}&size=10"
            
            headers = self.session.headers.copy()
            headers['User-Agent'] = random.choice(self.user_agents)
            
            response = self.session.get(pubmed_url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find article links
                articles = soup.find_all('article', class_='full-docsum')
                
                for article in articles[:5]:
                    try:
                        title_elem = article.find('a', class_='docsum-title')
                        if not title_elem:
                            continue
                        
                        title = title_elem.get_text().strip()
                        pmid_elem = article.find('span', class_='docsum-pmid')
                        pmid = pmid_elem.get_text().strip() if pmid_elem else None
                        
                        # Try to get abstract
                        abstract_elem = article.find('div', class_='full-view-snippet')
                        abstract = abstract_elem.get_text().strip() if abstract_elem else ""
                        
                        # Get authors
                        authors_elem = article.find('span', class_='docsum-authors')
                        authors = authors_elem.get_text().strip() if authors_elem else ""
                        
                        # Construct PubMed URL
                        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else pubmed_url
                        
                        relevance = self._calculate_relevance(title + " " + abstract, diagnosis, symptoms)
                        
                        source = MedicalSource(
                            title=title[:200],
                            url=url,
                            content_snippet=abstract[:400] + "..." if abstract else title,
                            domain="pubmed.ncbi.nlm.nih.gov",
                            relevance_score=relevance,
                            credibility_score=0.96,
                            source_type="academic_journal",
                            citation_format=self._generate_pubmed_citation(title, authors, pmid),
                            authors=authors,
                            pmid=pmid,
                            abstract=abstract,
                            keywords=self._extract_keywords(title + " " + abstract, diagnosis)
                        )
                        
                        sources.append(source)
                        
                    except Exception as e:
                        logger.warning(f"Error processing PubMed article: {e}")
                        continue
            
            print(f"📊 Academic journals found {len(sources)} sources")
            
        except Exception as e:
            logger.error(f"Academic journal search error: {e}")
        
        return sources
    
    def _search_wikipedia_medical(self, diagnosis: str, symptoms: List[str] = None) -> List[MedicalSource]:
        """Search Wikipedia for medical information"""
        
        sources = []
        
        try:
            # Wikipedia API search
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(diagnosis)}"
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                title = data.get('title', diagnosis)
                extract = data.get('extract', '')
                url = data.get('content_urls', {}).get('desktop', {}).get('page', '')
                
                if extract and len(extract) > 100:
                    relevance = self._calculate_relevance(extract, diagnosis, symptoms)
                    
                    source = MedicalSource(
                        title=f"Wikipedia: {title}",
                        url=url,
                        content_snippet=extract[:400] + "...",
                        domain="en.wikipedia.org",
                        relevance_score=relevance,
                        credibility_score=0.70,  # Moderate credibility for Wikipedia
                        source_type="encyclopedia",
                        citation_format=f"Wikipedia contributors. {title}. Wikipedia. Accessed {datetime.now().strftime('%Y-%m-%d')}. {url}",
                        keywords=self._extract_keywords(extract, diagnosis)
                    )
                    
                    sources.append(source)
            
            print(f"📊 Wikipedia found {len(sources)} sources")
            
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
        
        return sources
    
    def _clean_search_url(self, url: str) -> Optional[str]:
        """Clean and validate search result URLs"""
        
        try:
            # Handle DuckDuckGo redirect URLs
            if '/l/?uddg=' in url:
                actual_url = url.split('/l/?uddg=')[1].split('&')[0]
                return unquote(actual_url)
            
            # Handle other redirect patterns
            if url.startswith('http'):
                return url
            
            return None
            
        except Exception:
            return None
    
    def _extract_page_content(self, url: str, max_length: int = 2000) -> Optional[str]:
        """Extract meaningful content from a web page"""
        
        try:
            headers = self.session.headers.copy()
            headers['User-Agent'] = random.choice(self.user_agents)
            
            response = self.session.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Try to find main content areas
            content_selectors = [
                'main', '[role="main"]', '.main-content', '.content',
                'article', '.article', '#content', '.entry-content',
                '.post-content', '.page-content'
            ]
            
            content_text = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content_text = content_elem.get_text(separator=' ', strip=True)
                    break
            
            # Fallback to body text
            if not content_text or len(content_text) < 200:
                content_text = soup.get_text(separator=' ', strip=True)
            
            # Clean up the text
            content_text = re.sub(r'\s+', ' ', content_text)
            content_text = content_text.strip()
            
            return content_text[:max_length] if content_text else None
            
        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {e}")
            return None
    
    def _get_domain_credibility(self, domain: str) -> float:
        """Get credibility score for a domain"""
        
        domain = domain.lower()
        
        # Check exact matches first
        if domain in self.trusted_domains:
            return self.trusted_domains[domain]
        
        # Check partial matches
        for trusted_domain, score in self.trusted_domains.items():
            if trusted_domain in domain or domain in trusted_domain:
                return score * 0.95  # Slightly lower for partial matches
        
        # Check for medical keywords in domain
        medical_keywords = ['med', 'health', 'clinic', 'hospital', 'doctor', 'patient']
        if any(keyword in domain for keyword in medical_keywords):
            return 0.60  # Moderate credibility for medical-sounding domains
        
        # Educational and government domains
        if domain.endswith('.edu'):
            return 0.85
        elif domain.endswith('.gov'):
            return 0.90
        elif domain.endswith('.org'):
            return 0.70
        
        return 0.50  # Default credibility
    
    def _calculate_relevance(self, content: str, diagnosis: str, symptoms: List[str] = None) -> float:
        """Calculate relevance score based on content analysis"""
        
        if not content:
            return 0.0
        
        content_lower = content.lower()
        diagnosis_lower = diagnosis.lower()
        
        score = 0.0
        
        # Exact diagnosis match
        if diagnosis_lower in content_lower:
            score += 0.4
        
        # Partial diagnosis matches
        diagnosis_words = diagnosis_lower.split()
        for word in diagnosis_words:
            if len(word) > 3 and word in content_lower:
                score += 0.1
        
        # Symptom matches
        if symptoms:
            for symptom in symptoms:
                if symptom.lower() in content_lower:
                    score += 0.15
        
        # Medical terminology
        medical_terms = [
            'symptom', 'treatment', 'diagnosis', 'condition', 'disease',
            'therapy', 'patient', 'clinical', 'medical', 'cause', 'risk'
        ]
        for term in medical_terms:
            if term in content_lower:
                score += 0.05
        
        return min(score, 1.0)
    
    def _extract_keywords(self, content: str, diagnosis: str) -> List[str]:
        """Extract relevant keywords from content"""
        
        if not content:
            return []
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        
        # Medical keywords
        medical_keywords = [
            'symptom', 'treatment', 'therapy', 'diagnosis', 'condition',
            'disease', 'disorder', 'syndrome', 'patient', 'clinical'
        ]
        
        # Include diagnosis-related words
        diagnosis_words = diagnosis.lower().split()
        
        keywords = list(set(words) & set(medical_keywords + diagnosis_words))
        return keywords[:10]
    
    def _generate_citation(self, title: str, domain: str, url: str) -> str:
        """Generate proper citation format"""
        
        # Clean title
        clean_title = re.sub(r'[^\w\s-]', '', title).strip()
        if not clean_title:
            clean_title = "Medical Information"
        
        # Get current date
        access_date = datetime.now().strftime("%B %d, %Y")
        
        # Format citation based on source type
        if 'nih.gov' in domain or 'ncbi.nlm.nih.gov' in domain:
            return f"{clean_title}. National Institutes of Health. Accessed {access_date}. {url}"
        elif 'mayoclinic.org' in domain:
            return f"{clean_title}. Mayo Clinic. Accessed {access_date}. {url}"
        elif 'webmd.com' in domain:
            return f"{clean_title}. WebMD. Accessed {access_date}. {url}"
        elif 'healthline.com' in domain:
            return f"{clean_title}. Healthline. Accessed {access_date}. {url}"
        else:
            return f"{clean_title}. {domain}. Accessed {access_date}. {url}"
    
    def _generate_pubmed_citation(self, title: str, authors: str, pmid: str) -> str:
        """Generate PubMed citation"""
        
        citation = ""
        if authors:
            citation += f"{authors}. "
        
        citation += f"{title}. "
        
        if pmid:
            citation += f"PubMed PMID: {pmid}. "
        
        citation += f"Accessed {datetime.now().strftime('%B %d, %Y')}."
        
        return citation
    
    def _deduplicate_sources(self, sources: List[MedicalSource]) -> List[MedicalSource]:
        """Remove duplicate sources based on URL and title similarity"""
        
        seen_urls = set()
        seen_titles = set()
        unique_sources = []
        
        for source in sources:
            # URL deduplication
            if source.url in seen_urls:
                continue
            
            # Title similarity check (simple)
            title_key = source.title.lower()[:50]  # First 50 chars
            if title_key in seen_titles:
                continue
            
            seen_urls.add(source.url)
            seen_titles.add(title_key)
            unique_sources.append(source)
        
        return unique_sources
    
    def _analyze_evidence(self, sources: List[MedicalSource], diagnosis: str, 
                         symptoms: List[str] = None) -> Tuple[List[str], List[str]]:
        """Analyze sources for supporting and contradicting evidence"""
        
        supporting_evidence = []
        contradicting_evidence = []
        
        for source in sources[:8]:  # Analyze top 8 sources
            content = source.content_snippet.lower()
            diagnosis_lower = diagnosis.lower()
            
            # Look for supporting evidence
            supporting_indicators = [
                f"confirms {diagnosis_lower}",
                f"{diagnosis_lower} is characterized by",
                f"symptoms of {diagnosis_lower}",
                f"treatment for {diagnosis_lower}",
                f"{diagnosis_lower} typically",
                "diagnosis", "treatment", "symptoms", "condition"
            ]
            
            if any(indicator in content for indicator in supporting_indicators):
                evidence = f"Source: {source.domain} - {source.content_snippet[:150]}..."
                supporting_evidence.append(evidence)
            
            # Look for contradicting evidence
            contradicting_indicators = [
                "unlikely", "not consistent", "different condition",
                "misdiagnosis", "alternative diagnosis", "rare condition"
            ]
            
            if any(indicator in content for indicator in contradicting_indicators):
                evidence = f"Note from {source.domain}: {source.content_snippet[:150]}..."
                contradicting_evidence.append(evidence)
        
        return supporting_evidence[:5], contradicting_evidence[:3]
    
    def _calculate_verification_metrics(self, sources: List[MedicalSource], diagnosis: str) -> Tuple[float, str]:
        """Calculate overall confidence score and verification status"""
        
        if not sources:
            return 0.3, "NOT_VERIFIED"
        
        # Calculate weighted average credibility
        total_weight = 0
        weighted_credibility = 0
        
        for i, source in enumerate(sources[:10]):  # Top 10 sources
            # Weight decreases with position
            weight = 1.0 / (i + 1)
            weighted_credibility += source.credibility_score * weight
            total_weight += weight
        
        avg_credibility = weighted_credibility / total_weight if total_weight > 0 else 0
        
        # Calculate relevance score
        avg_relevance = sum(s.relevance_score for s in sources[:5]) / min(len(sources), 5)
        
        # Bonus for multiple high-quality sources
        high_quality_sources = len([s for s in sources if s.credibility_score > 0.8])
        quality_bonus = min(high_quality_sources * 0.05, 0.15)
        
        # Bonus for diverse source types
        source_types = set(s.source_type for s in sources)
        diversity_bonus = min(len(source_types) * 0.03, 0.1)
        
        # Calculate final confidence
        confidence = (avg_credibility * 0.5 + avg_relevance * 0.3) + quality_bonus + diversity_bonus
        confidence = min(confidence, 0.95)  # Cap at 95%
        
        # Determine verification status
        if confidence >= 0.75 and len(sources) >= 3:
            status = "VERIFIED"
        elif confidence >= 0.55 and len(sources) >= 2:
            status = "PARTIALLY_VERIFIED"
        elif confidence >= 0.4:
            status = "LIMITED_VERIFICATION"
        else:
            status = "INSUFFICIENT_EVIDENCE"
        
        return confidence, status
    
    def _generate_bibliography(self, sources: List[MedicalSource]) -> List[str]:
        """Generate formatted bibliography"""
        
        bibliography = []
        
        for i, source in enumerate(sources, 1):
            entry = f"{i}. {source.citation_format}"
            bibliography.append(entry)
        
        return bibliography
    
    def _generate_clinical_notes(self, diagnosis: str, sources: List[MedicalSource],
                                patient_age: int = None, patient_gender: str = None,
                                verification_attempts: List[str] = None) -> str:
        """Generate comprehensive clinical notes"""
        
        notes = f"Enhanced online medical verification for '{diagnosis}' completed. "
        
        if sources:
            notes += f"Analysis based on {len(sources)} medical sources. "
            
            # Source quality analysis
            high_cred_sources = len([s for s in sources if s.credibility_score > 0.8])
            if high_cred_sources > 0:
                notes += f"{high_cred_sources} high-credibility sources identified. "
            
            # Primary reference
            if sources:
                primary = sources[0]
                notes += f"Primary reference: {primary.domain} (Credibility: {primary.credibility_score:.2f}). "
        
        # Patient demographics
        if patient_age:
            age_group = "pediatric" if patient_age < 18 else "geriatric" if patient_age > 65 else "adult"
            notes += f"Patient demographics: {age_group} patient (age {patient_age}). "
        
        if patient_gender:
            notes += f"Gender: {patient_gender}. "
        
        # Verification strategy notes
        if verification_attempts:
            notes += f"Search strategies employed: {len(verification_attempts)} methods. "
            successful_attempts = [attempt for attempt in verification_attempts if "FAILED" not in attempt]
            notes += f"{len(successful_attempts)} successful verification attempts. "
        
        notes += "Recommend clinical correlation and professional medical evaluation."
        
        return notes
    
    def _generate_verification_summary(self, diagnosis: str, status: str, 
                                     confidence: float, source_count: int) -> str:
        """Generate verification summary"""
        
        if status == "VERIFIED" and confidence > 0.8:
            return f"Diagnosis '{diagnosis}' is well-supported by medical literature with high confidence (Score: {confidence:.3f}, Sources: {source_count})"
        elif status == "VERIFIED" or status == "PARTIALLY_VERIFIED":
            return f"Diagnosis '{diagnosis}' has reasonable medical support with moderate confidence (Score: {confidence:.3f}, Sources: {source_count})"
        elif status == "LIMITED_VERIFICATION":
            return f"Limited verification available for '{diagnosis}' - recommend additional clinical evaluation (Score: {confidence:.3f}, Sources: {source_count})"
        else:
            return f"Insufficient evidence for '{diagnosis}' in current medical literature - clinical consultation strongly recommended (Score: {confidence:.3f}, Sources: {source_count})"


# Test function
async def test_enhanced_verifier():
    """Test the enhanced online verifier"""
    
    verifier = EnhancedOnlineVerifier()
    
    # Test cases
    test_cases = [
        {
            'diagnosis': 'diabetes mellitus',
            'symptoms': ['increased thirst', 'frequent urination', 'fatigue'],
            'age': 55,
            'gender': 'male'
        },
        {
            'diagnosis': 'hypertension',
            'symptoms': ['headache', 'dizziness'],
            'age': 45,
            'gender': 'female'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"TESTING: {test_case['diagnosis'].upper()}")
        print('=' * 80)
        
        result = verifier.verify_diagnosis_online(
            diagnosis=test_case['diagnosis'],
            symptoms=test_case['symptoms'],
            patient_age=test_case['age'],
            patient_gender=test_case['gender']
        )
        
        print(f"\n📋 VERIFICATION RESULTS")
        print(f"Status: {result.verification_status}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Sources Found: {len(result.sources)}")
        print(f"Strategies Used: {', '.join(result.search_strategies_used)}")
        
        print(f"\n📚 TOP SOURCES:")
        for i, source in enumerate(result.sources[:3], 1):
            print(f"\n[{i}] {source.title}")
            print(f"    🌐 {source.domain} (Credibility: {source.credibility_score:.2f})")
            print(f"    📖 {source.citation_format}")
            print(f"    📄 {source.content_snippet[:200]}...")
        
        print(f"\n📋 BIBLIOGRAPHY:")
        for entry in result.bibliography[:5]:
            print(f"  {entry}")
        
        print(f"\n📝 CLINICAL NOTES:")
        print(f"  {result.clinical_notes}")
        
        print(f"\n📄 SUMMARY:")
        print(f"  {result.verification_summary}")


if __name__ == "__main__":
    asyncio.run(test_enhanced_verifier())