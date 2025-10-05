#!/usr/bin/env python3
"""
API Performance Testing and Optimization Tool
Tests and optimizes API response times with intelligent caching strategies
"""

import asyncio
import aiohttp
import json
import logging
import time
import statistics
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import concurrent.futures
import sys
import os
from urllib.parse import urljoin
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add backend to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_performance_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Configuration for performance testing"""
    base_url: str = "http://localhost:5000/"
    concurrent_users: int = 10
    test_duration_seconds: int = 60
    ramp_up_seconds: int = 10
    cooldown_seconds: int = 5
    request_timeout: float = 30.0
    enable_caching: bool = True
    cache_strategy: str = "intelligent"  # 'simple', 'intelligent', 'aggressive'

@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    path: str
    method: str = "POST"
    payload_template: Dict[str, Any] = None
    expected_status_code: int = 200
    name: str = ""
    priority: int = 1  # 1=low, 2=medium, 3=high
    cacheable: bool = True

@dataclass
class PerformanceResult:
    """Result of a single API call"""
    endpoint: str
    response_time: float
    status_code: int
    success: bool
    error_message: Optional[str] = None
    response_size: int = 0
    timestamp: float = 0.0

@dataclass
class LoadTestResult:
    """Aggregated results for load testing"""
    endpoint: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    error_rate: float
    throughput_mbps: float

@dataclass
class CachingRecommendation:
    """Caching strategy recommendation"""
    endpoint: str
    current_avg_time: float
    projected_avg_time: float
    cache_hit_ratio_estimate: float
    recommended_strategy: str
    implementation_complexity: str
    expected_performance_gain: float

class APIPerformanceTester:
    """Comprehensive API performance testing and optimization tool"""

    def __init__(self, config: PerformanceConfig):
        """Initialize the performance tester"""
        self.config = config
        self.session = None
        self.endpoints = self._load_endpoints()
        self.cache_manager = None

        if config.enable_caching:
            self.cache_manager = IntelligentCacheManager(config.cache_strategy)

    def _load_endpoints(self) -> List[APIEndpoint]:
        """Load API endpoints to test"""
        return [
            APIEndpoint(
                path="/api/predicates/extract",
                method="POST",
                payload_template={
                    "text": "Patient presents with chest pain and shortness of breath. "
                           "Blood pressure is 160/90 mmHg. Heart rate is 95 bpm. "
                           "EKG shows ST elevation in leads V1-V3. "
                           "Troponin level is 0.8 ng/ml. Diagnosed with myocardial infarction."
                },
                name="Predicate Extraction",
                priority=3,
                cacheable=True
            ),
            APIEndpoint(
                path="/api/predicates/validate",
                method="POST",
                payload_template={
                    "predicates": ["has_symptom(patient, chest_pain)", "has_condition(patient, myocardial_infarction)"],
                    "patient_data": {
                        "symptoms": ["chest pain", "dyspnea"],
                        "vitals": {"bp": "160/90", "hr": "95"},
                        "lab_results": {"troponin": "0.8"}
                    }
                },
                name="Predicate Validation",
                priority=2,
                cacheable=True
            ),
            APIEndpoint(
                path="/api/health",
                method="GET",
                payload_template={},
                name="Health Check",
                priority=1,
                cacheable=True
            ),
            APIEndpoint(
                path="/diagnose",
                method="POST",
                payload_template={
                    "clinical_text": "65-year-old male with acute chest pain radiating to left arm. "
                                   "Pain started 2 hours ago, described as crushing sensation. "
                                   "Associated with shortness of breath and diaphoresis. "
                                   "No prior cardiac history. Currently on aspirin 81mg daily.",
                    "patient_id": "PERF_TEST_PATIENT_{timestamp}",
                    "anonymize": True
                },
                name="Full Diagnosis",
                priority=3,
                cacheable=False
            ),
            APIEndpoint(
                path="/chat",
                method="POST",
                payload_template={
                    "session_id": "perf_test_session",
                    "diagnosis_session_id": "perf_test_diagnosis",
                    "message": "What are the treatment options for myocardial infarction?"
                },
                name="Chat Interaction",
                priority=2,
                cacheable=False
            )
        ]

    async def run_comprehensive_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance testing"""
        logger.info("Starting comprehensive API performance testing...")

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
        ) as session:
            self.session = session

            # Phase 1: Warm-up phase
            logger.info("Phase 1: Warm-up testing...")
            await self._warm_up_system()

            # Phase 2: Baseline single-user testing
            logger.info("Phase 2: Baseline single-user testing...")
            baseline_results = await self._run_baseline_tests()

            # Phase 3: Load testing with concurrent users
            logger.info("Phase 3: Load testing...")
            load_results = await self._run_load_tests()

            # Phase 4: Stress testing
            logger.info("Phase 4: Stress testing...")
            stress_results = await self._run_stress_tests()

            # Phase 5: Caching analysis and optimization
            logger.info("Phase 5: Caching analysis...")
            caching_recommendations = await self._analyze_caching_opportunities(load_results)

            # Generate comprehensive report
            report = self._generate_performance_report(
                baseline_results, load_results, stress_results, caching_recommendations
            )

            return report

    async def _warm_up_system(self):
        """Warm up the system with light load"""
        logger.info("Warming up system...")

        for endpoint in self.endpoints:
            try:
                # Make a few requests to warm up caches
                for i in range(3):
                    await self._make_request(endpoint)
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Warm-up failed for {endpoint.path}: {str(e)}")

        await asyncio.sleep(2)  # Allow system to stabilize

    async def _run_baseline_tests(self) -> Dict[str, LoadTestResult]:
        """Run baseline single-user tests"""
        logger.info("Running baseline tests...")

        baseline_results = {}

        for endpoint in self.endpoints:
            logger.info(f"Testing baseline for {endpoint.name}...")

            results = []
            for i in range(10):  # 10 requests per endpoint
                result = await self._make_request(endpoint)
                results.append(result)
                await asyncio.sleep(0.2)  # Small delay between requests

            # Calculate metrics
            successful_requests = [r for r in results if r.success]
            response_times = [r.response_time for r in results if r.success]

            if response_times:
                baseline_results[endpoint.name] = LoadTestResult(
                    endpoint=endpoint.name,
                    total_requests=len(results),
                    successful_requests=len(successful_requests),
                    failed_requests=len(results) - len(successful_requests),
                    average_response_time=statistics.mean(response_times),
                    median_response_time=statistics.median(response_times),
                    p95_response_time=sorted(response_times)[int(len(response_times) * 0.95)],
                    p99_response_time=sorted(response_times)[int(len(response_times) * 0.99)],
                    min_response_time=min(response_times),
                    max_response_time=max(response_times),
                    requests_per_second=1.0 / statistics.mean(response_times) if response_times else 0,
                    error_rate=(len(results) - len(successful_requests)) / len(results),
                    throughput_mbps=sum(r.response_size for r in successful_requests) / len(successful_requests) / 1024 / 1024 if successful_requests else 0
                )

        return baseline_results

    async def _run_load_tests(self) -> Dict[str, LoadTestResult]:
        """Run load tests with concurrent users"""
        logger.info(f"Running load tests with {self.config.concurrent_users} concurrent users...")

        load_results = {}

        for endpoint in self.endpoints:
            logger.info(f"Load testing {endpoint.name}...")

            # Create concurrent requests
            semaphore = asyncio.Semaphore(self.config.concurrent_users)

            async def limited_request():
                async with semaphore:
                    return await self._make_request(endpoint)

            # Run requests concurrently
            tasks = [limited_request() for _ in range(self.config.concurrent_users * 5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            successful_results = [r for r in results if isinstance(r, PerformanceResult) and r.success]
            response_times = [r.response_time for r in successful_results]

            if response_times:
                load_results[endpoint.name] = LoadTestResult(
                    endpoint=endpoint.name,
                    total_requests=len(tasks),
                    successful_requests=len(successful_results),
                    failed_requests=len(tasks) - len(successful_results),
                    average_response_time=statistics.mean(response_times),
                    median_response_time=statistics.median(response_times),
                    p95_response_time=sorted(response_times)[int(len(response_times) * 0.95)],
                    p99_response_time=sorted(response_times)[int(len(response_times) * 0.99)],
                    min_response_time=min(response_times),
                    max_response_time=max(response_times),
                    requests_per_second=len(successful_results) / sum(response_times),
                    error_rate=(len(tasks) - len(successful_results)) / len(tasks),
                    throughput_mbps=sum(r.response_size for r in successful_results) / len(successful_results) / 1024 / 1024 if successful_results else 0
                )

        return load_results

    async def _run_stress_tests(self) -> Dict[str, LoadTestResult]:
        """Run stress tests to find breaking points"""
        logger.info("Running stress tests...")

        stress_results = {}

        # Test with increasing load
        stress_levels = [self.config.concurrent_users * 2, self.config.concurrent_users * 3]

        for stress_level in stress_levels:
            logger.info(f"Stress testing with {stress_level} concurrent users...")

            stress_endpoint_results = {}

            for endpoint in self.endpoints:
                if endpoint.priority >= 2:  # Only test high-priority endpoints in stress tests
                    semaphore = asyncio.Semaphore(stress_level)

                    async def limited_request():
                        async with semaphore:
                            return await self._make_request(endpoint)

                    tasks = [limited_request() for _ in range(stress_level * 3)]
                    start_time = time.time()
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    end_time = time.time()

                    successful_results = [r for r in results if isinstance(r, PerformanceResult) and r.success]
                    response_times = [r.response_time for r in successful_results]

                    if response_times:
                        stress_endpoint_results[endpoint.name] = LoadTestResult(
                            endpoint=endpoint.name,
                            total_requests=len(tasks),
                            successful_requests=len(successful_results),
                            failed_requests=len(tasks) - len(successful_results),
                            average_response_time=statistics.mean(response_times),
                            median_response_time=statistics.median(response_times),
                            p95_response_time=sorted(response_times)[int(len(response_times) * 0.95)],
                            p99_response_time=sorted(response_times)[int(len(response_times) * 0.99)],
                            min_response_time=min(response_times),
                            max_response_time=max(response_times),
                            requests_per_second=len(tasks) / (end_time - start_time),
                            error_rate=(len(tasks) - len(successful_results)) / len(tasks),
                            throughput_mbps=sum(r.response_size for r in successful_results) / len(successful_results) / 1024 / 1024 if successful_results else 0
                        )

            stress_results[f"stress_level_{stress_level}"] = stress_endpoint_results

        return stress_results

    async def _make_request(self, endpoint: APIEndpoint) -> PerformanceResult:
        """Make a single API request and measure performance"""
        start_time = time.time()

        try:
            url = urljoin(self.config.base_url, endpoint.path)

            # Prepare payload
            payload = endpoint.payload_template.copy() if endpoint.payload_template else {}

            # Add dynamic content for certain endpoints
            if "timestamp" in str(payload):
                timestamp = str(int(time.time()))
                payload = json.loads(json.dumps(payload).replace("{timestamp}", timestamp))

            # Make request
            async with self.session.request(
                endpoint.method,
                url,
                json=payload if payload else None,
                headers={'Content-Type': 'application/json'}
            ) as response:
                response_time = time.time() - start_time

                # Read response content
                content = await response.text()
                response_size = len(content)

                success = response.status == endpoint.expected_status_code

                return PerformanceResult(
                    endpoint=endpoint.name,
                    response_time=response_time,
                    status_code=response.status,
                    success=success,
                    error_message=None if success else content[:200],
                    response_size=response_size,
                    timestamp=time.time()
                )

        except Exception as e:
            response_time = time.time() - start_time
            return PerformanceResult(
                endpoint=endpoint.name,
                response_time=response_time,
                status_code=0,
                success=False,
                error_message=str(e),
                timestamp=time.time()
            )

    async def _analyze_caching_opportunities(self, load_results: Dict[str, LoadTestResult]) -> Dict[str, CachingRecommendation]:
        """Analyze caching opportunities and provide recommendations"""
        logger.info("Analyzing caching opportunities...")

        recommendations = {}

        for endpoint_name, result in load_results.items():
            # Find corresponding endpoint config
            endpoint = next((e for e in self.endpoints if e.name == endpoint_name), None)
            if not endpoint or not endpoint.cacheable:
                continue

            current_avg_time = result.average_response_time

            # Estimate cache hit ratio based on endpoint characteristics
            if endpoint.priority == 3:  # High priority endpoints
                cache_hit_ratio = 0.7  # 70% cache hit ratio for frequently accessed data
            elif endpoint.priority == 2:  # Medium priority
                cache_hit_ratio = 0.5  # 50% cache hit ratio
            else:
                cache_hit_ratio = 0.3  # 30% cache hit ratio

            # Estimate performance gain
            # Assume cache response time is 10ms for hit, current time for miss
            cache_hit_time = 0.01  # 10ms
            cache_miss_time = current_avg_time
            projected_avg_time = (cache_hit_time * cache_hit_ratio) + (cache_miss_time * (1 - cache_hit_ratio))
            performance_gain = (current_avg_time - projected_avg_time) / current_avg_time

            # Determine recommended strategy
            if performance_gain > 0.5:  # >50% improvement
                recommended_strategy = "aggressive_caching"
                complexity = "medium"
            elif performance_gain > 0.3:  # >30% improvement
                recommended_strategy = "intelligent_caching"
                complexity = "low"
            else:
                recommended_strategy = "selective_caching"
                complexity = "low"

            recommendations[endpoint_name] = CachingRecommendation(
                endpoint=endpoint_name,
                current_avg_time=current_avg_time,
                projected_avg_time=projected_avg_time,
                cache_hit_ratio_estimate=cache_hit_ratio,
                recommended_strategy=recommended_strategy,
                implementation_complexity=complexity,
                expected_performance_gain=performance_gain
            )

        return recommendations

    def _generate_performance_report(self, baseline_results: Dict[str, LoadTestResult],
                                   load_results: Dict[str, LoadTestResult],
                                   stress_results: Dict[str, Dict[str, LoadTestResult]],
                                   caching_recommendations: Dict[str, CachingRecommendation]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""

        # Calculate system-wide metrics
        all_response_times = []
        total_requests = 0
        total_successful = 0

        for result in baseline_results.values():
            all_response_times.extend([result.average_response_time] * result.successful_requests)
            total_requests += result.total_requests
            total_successful += result.successful_requests

        for result in load_results.values():
            all_response_times.extend([result.average_response_time] * result.successful_requests)
            total_requests += result.total_requests
            total_successful += result.successful_requests

        overall_avg_response_time = statistics.mean(all_response_times) if all_response_times else 0
        overall_success_rate = total_successful / total_requests if total_requests > 0 else 0

        # Identify bottlenecks
        bottlenecks = []
        for endpoint_name, result in load_results.items():
            if result.average_response_time > 2.0:  # >2 seconds is slow
                bottlenecks.append({
                    "endpoint": endpoint_name,
                    "avg_response_time": result.average_response_time,
                    "error_rate": result.error_rate
                })

        # Performance recommendations
        recommendations = []

        if overall_avg_response_time > 1.0:
            recommendations.append("Consider implementing response caching for slow endpoints")
        if overall_success_rate < 0.95:
            recommendations.append("Investigate and fix high error rates on failing endpoints")
        if bottlenecks:
            recommendations.append("Optimize identified bottleneck endpoints")

        # Add caching recommendations
        for rec in caching_recommendations.values():
            if rec.expected_performance_gain > 0.3:
                recommendations.append(
                    f"Implement {rec.recommended_strategy} for {rec.endpoint} "
                    f"(expected {rec.expected_performance_gain:.1%} improvement)"
                )

        report = {
            "test_summary": {
                "test_timestamp": time.time(),
                "configuration": asdict(self.config),
                "overall_metrics": {
                    "average_response_time": overall_avg_response_time,
                    "success_rate": overall_success_rate,
                    "total_requests": total_requests,
                    "total_successful": total_successful
                }
            },
            "baseline_results": {k: asdict(v) for k, v in baseline_results.items()},
            "load_results": {k: asdict(v) for k, v in load_results.items()},
            "stress_results": {
                level: {k: asdict(v) for k, v in results.items()}
                for level, results in stress_results.items()
            },
            "caching_recommendations": {k: asdict(v) for k, v in caching_recommendations.items()},
            "bottlenecks": bottlenecks,
            "recommendations": recommendations
        }

        # Save detailed report
        with open('api_performance_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        self._print_performance_summary(report)

        return report

    def _print_performance_summary(self, report: Dict[str, Any]):
        """Print performance test summary"""
        print("\n" + "="*80)
        print("API PERFORMANCE TEST RESULTS")
        print("="*80)

        metrics = report["test_summary"]["overall_metrics"]
        print(".2f")
        print(".1%")
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Successful Requests: {metrics['total_successful']}")

        print("\nENDPOINT PERFORMANCE:")
        for endpoint, result in report["load_results"].items():
            print(f"  {endpoint:25} | Avg: {result['average_response_time']:.2f}s | "
                  f"P95: {result['p95_response_time']:.2f}s | Success: {result['successful_requests']/result['total_requests']:.1%}")

        print("\nBOTTLENECKS:")
        for bottleneck in report["bottlenecks"]:
            print(f"  ⚠️  {bottleneck['endpoint']}: {bottleneck['avg_response_time']:.2f}s avg response time")

        print("\nRECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

        print("\nCACHING OPPORTUNITIES:")
        for endpoint, rec in report["caching_recommendations"].items():
            print(f"  {endpoint:25} | Expected Gain: {rec['expected_performance_gain']:.1%} | "
                  f"Strategy: {rec['recommended_strategy']}")

        print("="*80)

class IntelligentCacheManager:
    """Intelligent caching manager with multiple strategies"""

    def __init__(self, strategy: str = "intelligent"):
        """Initialize cache manager"""
        self.strategy = strategy
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size_limit = 1000

    def get_cache_key(self, endpoint: str, payload: Dict[str, Any]) -> str:
        """Generate cache key based on strategy"""
        if self.strategy == "simple":
            # Simple key based on endpoint only
            return f"{endpoint}"
        elif self.strategy == "intelligent":
            # Intelligent key based on semantic content
            key_parts = [endpoint]

            # Extract semantic elements from payload
            if "text" in payload:
                # Create hash of key medical terms
                text = payload["text"].lower()
                medical_terms = []
                for term in ["chest pain", "myocardial", "infarction", "diabetes", "hypertension"]:
                    if term in text:
                        medical_terms.append(term)
                if medical_terms:
                    key_parts.append("|".join(sorted(medical_terms)))

            return ":".join(key_parts)
        else:  # aggressive
            # Aggressive key based on full payload
            import hashlib
            payload_str = json.dumps(payload, sort_keys=True)
            payload_hash = hashlib.md5(payload_str.encode()).hexdigest()[:8]
            return f"{endpoint}:{payload_hash}"

    def get(self, endpoint: str, payload: Dict[str, Any]) -> Optional[Any]:
        """Get cached result"""
        key = self.get_cache_key(endpoint, payload)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        else:
            self.cache_misses += 1
            return None

    def set(self, endpoint: str, payload: Dict[str, Any], result: Any):
        """Set cached result"""
        key = self.get_cache_key(endpoint, payload)
        self.cache[key] = result

        # Implement LRU eviction if cache is full
        if len(self.cache) > self.cache_size_limit:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

    def get_hit_ratio(self) -> float:
        """Get cache hit ratio"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0

async def main():
    """Main performance testing function"""
    print("API Performance Testing and Optimization Tool")
    print("=" * 60)

    # Load configuration from environment or use defaults
    config = PerformanceConfig(
        base_url=os.getenv("API_BASE_URL", "http://localhost:5000/"),
        concurrent_users=int(os.getenv("PERF_CONCURRENT_USERS", "10")),
        test_duration_seconds=int(os.getenv("PERF_DURATION", "60")),
        enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
        cache_strategy=os.getenv("CACHE_STRATEGY", "intelligent")
    )

    print(f"Configuration:")
    print(f"  Base URL: {config.base_url}")
    print(f"  Concurrent Users: {config.concurrent_users}")
    print(f"  Test Duration: {config.test_duration_seconds}s")
    print(f"  Caching: {config.enable_caching} ({config.cache_strategy})")

    # Initialize tester
    tester = APIPerformanceTester(config)

    try:
        # Run comprehensive tests
        print("\n🚀 Starting comprehensive performance testing...")
        report = await tester.run_comprehensive_performance_test()

        print("\n✅ Performance testing complete!")
        print("📊 Detailed results saved to api_performance_report.json")

        # Print key insights
        metrics = report["test_summary"]["overall_metrics"]
        if metrics["average_response_time"] < 1.0 and metrics["success_rate"] > 0.95:
            print("🎉 Excellent performance! System is well-optimized.")
        elif metrics["average_response_time"] < 2.0 and metrics["success_rate"] > 0.90:
            print("👍 Good performance with minor optimization opportunities.")
        else:
            print("⚠️  Performance optimization needed. Review recommendations.")

    except Exception as e:
        logger.error(f"Performance testing failed: {str(e)}")
        print(f"❌ Performance testing failed: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
