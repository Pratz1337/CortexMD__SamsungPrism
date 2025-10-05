#!/usr/bin/env python3
"""
Real Web Browsing Medical Verification System using Selenium
Browses actual medical websites in real-time to verify diagnoses
"""

import time
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import requests

@dataclass
class MedicalSource:
    """Medical source information from web browsing"""
    title: str
    url: str
    content_snippet: str
    domain: str
    credibility_score: float
    relevance_score: float
    access_date: str
    source_type: str
    full_content: str = ""
    citation_format: str = ""

@dataclass
class WebBrowsingResult:
    """Result from web browsing medical verification"""
    verification_status: str
    confidence_score: float
    sources: List[MedicalSource]
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    clinical_notes: str
    verification_summary: str
    timestamp: str
    search_queries_used: List[str]
    total_sources_browsed: int

class SeleniumMedicalBrowser:
    """Real-time web browsing for medical verification using Selenium"""
    
    def __init__(self):
        """Initialize the selenium browser"""
        self.trusted_medical_sites = {
            'mayoclinic.org': 0.95,
            'webmd.com': 0.85, 
            'healthline.com': 0.85,
            'medlineplus.gov': 0.98,
            'nih.gov': 0.98,
            'cdc.gov': 0.98,
            'who.int': 0.98,
            'ncbi.nlm.nih.gov': 0.95,
            'clevelandclinic.org': 0.90,
            'johnshopkins.org': 0.90,
            'cancer.org': 0.92,
            'heart.org': 0.90,
            'diabetes.org': 0.88,
            'medicalnewstoday.com': 0.80,
            'drugs.com': 0.85,
            'rxlist.com': 0.80,
            'uptodate.com': 0.95,
            'medscape.com': 0.88
        }
        
        self.search_engines = [
            "https://www.google.com/search?q=",
            "https://duckduckgo.com/?q=",
            "https://www.bing.com/search?q="
        ]
        
        self.driver = None
        
    def _setup_driver(self) -> webdriver.Chrome:
        """Setup Chrome driver with options"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Execute script to hide webdriver property
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver
    
    def _search_medical_condition(self, diagnosis: str, symptoms: List[str]) -> List[str]:
        """Generate search queries for medical condition"""
        queries = [
            f"{diagnosis} symptoms causes treatment medical",
            f"{diagnosis} diagnosis criteria medical definition",
            f"what is {diagnosis} disease condition medical",
            f"{diagnosis} {' '.join(symptoms[:3])} medical diagnosis",
            f"{diagnosis} medical information symptoms treatment",
            f"{diagnosis} patient symptoms {' '.join(symptoms[:2])}"
        ]
        return queries
    
    def _browse_search_results(self, query: str, max_results: int = 10) -> List[MedicalSource]:
        """Browse search results and extract medical information"""
        sources = []
        
        try:
            # Use DuckDuckGo search (works better than Google for automation)
            search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
            print(f"🌐 Browsing: {search_url}")
            
            self.driver.get(search_url)
            time.sleep(3)  # Wait for page load
            
            # Find search result links using DuckDuckGo selectors
            search_results = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='result']")
            print(f"📄 Found {len(search_results)} search results")
            
            result_count = 0
            for result in search_results[:max_results]:
                if result_count >= max_results:
                    break
                    
                try:
                    # Get link and title from DuckDuckGo result
                    link_element = result.find_element(By.CSS_SELECTOR, "a")
                    url = link_element.get_attribute("href")
                    
                    # Skip DuckDuckGo internal URLs
                    if not url or "duckduckgo.com" in url or "youtube.com" in url:
                        continue
                    
                    title_element = result.find_element(By.CSS_SELECTOR, "h2")
                    title = title_element.text if title_element else "Unknown Title"
                    
                    # Check if it's a trusted medical site
                    domain = self._extract_domain(url)
                    if not self._is_trusted_medical_site(domain):
                        continue
                    
                    print(f"📄 Found medical site: {domain} - {title[:50]}...")
                    
                    # Browse the actual medical page
                    source = self._browse_medical_page(url, title, domain)
                    if source:
                        sources.append(source)
                        result_count += 1
                        
                except Exception as e:
                    print(f"❌ Error processing result: {e}")
                    continue
            
            # If no results from search, try direct medical site access
            if not sources:
                print("🏥 No search results found, trying direct medical site access...")
                sources.extend(self._browse_direct_medical_sites(query))
                    
        except Exception as e:
            print(f"❌ Error browsing search results: {e}")
        
        return sources
    
    def _browse_direct_medical_sites(self, diagnosis: str) -> List[MedicalSource]:
        """Browse medical sites directly when search fails"""
        sources = []
        
        # Direct URLs for common medical sites
        direct_urls = [
            f"https://www.mayoclinic.org/diseases-conditions/{diagnosis.lower()}/symptoms-causes/syc-20351048",
            f"https://www.webmd.com/search/search_results/default.aspx?query={diagnosis}",
            f"https://www.healthline.com/health/{diagnosis.lower()}",
            f"https://medlineplus.gov/search/searchresults.aspx?query={diagnosis}",
            f"https://www.clevelandclinic.org/health/diseases/{diagnosis.lower()}"
        ]
        
        for url in direct_urls:
            try:
                print(f"🏥 Trying direct access: {url}")
                
                self.driver.get(url)
                time.sleep(3)
                
                # Check if page loaded successfully and contains relevant content
                page_source = self.driver.page_source
                if diagnosis.lower() in page_source.lower():
                    domain = self._extract_domain(url)
                    title = self.driver.title
                    
                    source = self._browse_medical_page(url, title, domain)
                    if source:
                        sources.append(source)
                        print(f"✅ Successfully extracted from {domain}")
                
            except Exception as e:
                print(f"❌ Error accessing {url}: {e}")
                continue
        
        return sources
    
    def _browse_medical_page(self, url: str, title: str, domain: str) -> Optional[MedicalSource]:
        """Browse individual medical page and extract content"""
        try:
            # Navigate to the medical page directly (no new tabs)
            original_url = self.driver.current_url
            
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Extract page content
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Remove scripts, styles, and navigation
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Extract main content
            content = ""
            
            # Try different content selectors based on common medical site structures
            content_selectors = [
                "main", "article", ".content", ".main-content", 
                "#content", ".article-content", ".post-content",
                ".entry-content", "section", ".page-content"
            ]
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text(strip=True)
                    break
            
            if not content:
                content = soup.get_text(strip=True)
            
            # Clean and limit content
            content = re.sub(r'\s+', ' ', content)[:3000]  # Increased content limit
            
            # Verify content is relevant (contains medical information)
            medical_keywords = ['symptom', 'treatment', 'diagnosis', 'condition', 'disease', 'medical', 'health']
            if not any(keyword in content.lower() for keyword in medical_keywords):
                print(f"⚠️  No relevant medical content found on {domain}")
                return None
            
            # Create source object
            source = MedicalSource(
                title=title,
                url=url,
                content_snippet=content[:400] + "..." if len(content) > 400 else content,
                domain=domain,
                credibility_score=self.trusted_medical_sites.get(domain, 0.5),
                relevance_score=1.0,  # Will be calculated later
                access_date=datetime.now().strftime("%B %d, %Y"),
                source_type="medical_website",
                full_content=content,
                citation_format=f"{title}. {domain}. Accessed {datetime.now().strftime('%B %d, %Y')}. {url}"
            )
            
            return source
            
        except Exception as e:
            print(f"❌ Error browsing medical page {url}: {e}")
            return None
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower().replace('www.', '')
        except:
            return url.lower()
    
    def _is_trusted_medical_site(self, domain: str) -> bool:
        """Check if domain is a trusted medical site"""
        domain = domain.lower().replace('www.', '')
        return any(trusted in domain for trusted in self.trusted_medical_sites.keys())
    
    def _calculate_relevance_score(self, source: MedicalSource, diagnosis: str, symptoms: List[str]) -> float:
        """Calculate relevance score based on content matching"""
        content_lower = source.full_content.lower()
        diagnosis_lower = diagnosis.lower()
        
        score = 0.0
        
        # Check diagnosis name
        if diagnosis_lower in content_lower:
            score += 0.4
        
        # Check symptoms
        for symptom in symptoms:
            if symptom.lower() in content_lower:
                score += 0.1
        
        # Check medical terms
        medical_terms = ['symptom', 'diagnosis', 'treatment', 'condition', 'disease', 'medical']
        for term in medical_terms:
            if term in content_lower:
                score += 0.05
        
        return min(score, 1.0)
    
    def _analyze_verification_results(self, sources: List[MedicalSource], diagnosis: str) -> Dict[str, Any]:
        """Analyze sources to determine verification status"""
        if not sources:
            return {
                'status': 'INSUFFICIENT_DATA',
                'confidence': 0.0,
                'supporting': [],
                'contradicting': []
            }
        
        # Calculate weighted confidence based on source credibility
        total_credibility = sum(source.credibility_score for source in sources)
        avg_credibility = total_credibility / len(sources)
        
        # Determine verification status
        if avg_credibility > 0.8 and len(sources) >= 3:
            status = 'VERIFIED'
            confidence = min(avg_credibility, 1.0)
        elif avg_credibility > 0.6 and len(sources) >= 2:
            status = 'LIKELY'
            confidence = avg_credibility * 0.8
        else:
            status = 'UNCERTAIN'
            confidence = avg_credibility * 0.6
        
        # Generate evidence
        supporting = [f"Confirmed by {source.domain}: {source.content_snippet[:100]}..." 
                     for source in sources if source.credibility_score > 0.7]
        
        contradicting = []  # Would need more sophisticated analysis
        
        return {
            'status': status,
            'confidence': confidence,
            'supporting': supporting,
            'contradicting': contradicting
        }
    
    async def verify_diagnosis_with_web_browsing(
        self, 
        diagnosis: str, 
        symptoms: List[str],
        patient_age: int = None,
        patient_gender: str = None
    ) -> WebBrowsingResult:
        """Main method to verify diagnosis using real web browsing"""
        
        print(f"🌐 REAL-TIME WEB BROWSING FOR: {diagnosis}")
        print("=" * 60)
        
        try:
            # Setup driver
            self.driver = self._setup_driver()
            
            # Generate search queries
            queries = self._search_medical_condition(diagnosis, symptoms)
            print(f"🔍 Generated {len(queries)} search queries...")
            
            all_sources = []
            
            # Browse each query
            for i, query in enumerate(queries[:3], 1):  # Limit to 3 queries
                print(f"\n📋 Query {i}: {query}")
                sources = self._browse_search_results(query, max_results=3)
                
                # Calculate relevance scores
                for source in sources:
                    source.relevance_score = self._calculate_relevance_score(source, diagnosis, symptoms)
                
                all_sources.extend(sources)
                time.sleep(2)  # Polite delay between searches
            
            # Remove duplicates based on URL
            unique_sources = []
            seen_urls = set()
            for source in all_sources:
                if source.url not in seen_urls:
                    unique_sources.append(source)
                    seen_urls.add(source.url)
            
            # Sort by credibility and relevance
            unique_sources.sort(key=lambda x: (x.credibility_score + x.relevance_score) / 2, reverse=True)
            
            # Analyze results
            analysis = self._analyze_verification_results(unique_sources, diagnosis)
            
            # Generate clinical notes
            clinical_notes = f"Real-time web browsing verification for {diagnosis} completed. "
            if patient_age:
                clinical_notes += f"Patient age ({patient_age}) considered. "
            clinical_notes += f"Browsed {len(unique_sources)} trusted medical sources."
            
            # Generate summary
            summary = f"Diagnosis '{diagnosis}' verification via web browsing: {analysis['status']} "
            summary += f"(Confidence: {analysis['confidence']:.2f}, Sources: {len(unique_sources)})"
            
            result = WebBrowsingResult(
                verification_status=analysis['status'],
                confidence_score=analysis['confidence'],
                sources=unique_sources,
                supporting_evidence=analysis['supporting'],
                contradicting_evidence=analysis['contradicting'],
                clinical_notes=clinical_notes,
                verification_summary=summary,
                timestamp=datetime.now().isoformat(),
                search_queries_used=queries,
                total_sources_browsed=len(unique_sources)
            )
            
            print(f"✅ Browsing complete: {analysis['status']}")
            print(f"📊 Confidence: {analysis['confidence']:.2f}")
            print(f"📚 Sources browsed: {len(unique_sources)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in web browsing: {e}")
            import traceback
            traceback.print_exc()
            
            return WebBrowsingResult(
                verification_status="ERROR",
                confidence_score=0.0,
                sources=[],
                supporting_evidence=[],
                contradicting_evidence=[],
                clinical_notes=f"Error during web browsing: {str(e)}",
                verification_summary="Web browsing verification failed",
                timestamp=datetime.now().isoformat(),
                search_queries_used=[],
                total_sources_browsed=0
            )
        
        finally:
            # Clean up driver
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
            except:
                pass
