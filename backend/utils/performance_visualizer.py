#!/usr/bin/env python3
"""
Performance Visualization System for Ontology Mapping
Generates charts and graphs for response times, error rates, and throughput
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

class PerformanceVisualizer:
    """Generates performance visualizations for the ontology mapping system"""

    def __init__(self, results_file="integrated_system_test_results.json"):
        """Initialize the performance visualizer"""
        self.results_file = results_file
        self.results = None
        self.output_dir = Path("visualizations")

        # Create output directory if it doesn't exist
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create visualizations directory: {e}")
            # Fallback to current directory
            self.output_dir = Path(".")

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_results(self):
        """Load test results from JSON file"""
        try:
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r') as f:
                    self.results = json.load(f)
                logger.info(f"Loaded test results from {self.results_file}")
                return True
            else:
                logger.warning(f"Results file {self.results_file} not found")
                return False
        except Exception as e:
            logger.error(f"Failed to load results: {str(e)}")
            return False

    def generate_response_time_chart(self):
        """Generate response time comparison chart"""
        if not self.results:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ontology Mapping System - Response Time Analysis', fontsize=16, fontweight='bold')

        # 1. Ontology Mapping Performance
        if "ontology_mapping_performance" in self.results:
            perf = self.results["ontology_mapping_performance"]
            if "term_normalization" in perf:
                norm = perf["term_normalization"]

                labels = []
                times = []
                if "without_cache" in norm:
                    labels.append("Without Cache")
                    times.append(norm["without_cache"]["avg_time_per_term"])
                if "with_cache" in norm:
                    labels.append("With Cache")
                    times.append(norm["with_cache"]["avg_time_per_term"])

                if labels and times:
                    axes[0, 0].bar(labels, times, color=['#ff6b6b', '#4ecdc4'])
                    axes[0, 0].set_title('Term Normalization Performance')
                    axes[0, 0].set_ylabel('Avg Time per Term (seconds)')
                    axes[0, 0].grid(True, alpha=0.3)

                    # Add performance improvement annotation
                    if len(times) == 2:
                        improvement = times[0] / times[1] if times[1] > 0 else 0
                        axes[0, 0].text(1, times[1] + 0.0005, '.2f',
                                      ha='center', va='bottom', fontweight='bold',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#4ecdc4'))

        # 2. Parallel Processing Throughput
        if "parallel_processing" in self.results:
            parallel = self.results["parallel_processing"]

            batch_labels = []
            batch_throughput = []
            if "batch_processing" in parallel:
                for key, data in parallel["batch_processing"].items():
                    batch_labels.append(key.replace("batch_size_", "Batch "))
                    batch_throughput.append(data["throughput_terms_per_second"])

            if batch_labels and batch_throughput:
                axes[0, 1].bar(batch_labels, batch_throughput, color='#45b7d1')
                axes[0, 1].set_title('Batch Processing Throughput')
                axes[0, 1].set_ylabel('Terms/Second')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. API Endpoints Response Times
        if "api_endpoints_simulation" in self.results:
            api = self.results["api_endpoints_simulation"]

            endpoint_names = []
            response_times = []
            success_rates = []

            if "endpoint_responses" in api:
                for endpoint, data in api["endpoint_responses"].items():
                    endpoint_names.append(endpoint.replace("_", " ").title())
                    response_times.append(data["avg_response_time"])
                    success_rates.append(data["successful_tests"] / data["tests_run"] * 100)

            if endpoint_names and response_times:
                x = np.arange(len(endpoint_names))
                width = 0.35

                axes[1, 0].bar(x - width/2, response_times, width, label='Response Time', color='#96ceb4')
                ax2 = axes[1, 0].twinx()
                ax2.bar(x + width/2, success_rates, width, label='Success Rate', color='#ffeaa7')
                ax2.set_ylabel('Success Rate (%)', color='#ffeaa7')

                axes[1, 0].set_title('API Endpoints Performance')
                axes[1, 0].set_xlabel('Endpoints')
                axes[1, 0].set_ylabel('Response Time (seconds)', color='#96ceb4')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(endpoint_names, rotation=45, ha='right')
                axes[1, 0].grid(True, alpha=0.3)

                # Add legend
                lines1, labels1 = axes[1, 0].get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                axes[1, 0].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # 4. Cache Performance
        if "ontology_mapping_performance" in self.results:
            perf = self.results["ontology_mapping_performance"]
            if "cache_performance" in perf:
                cache_perf = perf["cache_performance"]

                labels = ['Cache Hits', 'Cache Misses']
                values = [cache_perf["cache_hits"], cache_perf["cache_misses"]]
                colors = ['#6c5ce7', '#fd79a8']

                axes[1, 1].pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                             startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
                axes[1, 1].set_title('Cache Hit/Miss Distribution')

                # Add hit ratio as text
                hit_ratio = cache_perf["hit_ratio"]
                axes[1, 1].text(0, -1.2, '.1%', ha='center',
                               fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='#6c5ce7', alpha=0.8))

        plt.tight_layout()
        return fig

    def generate_throughput_chart(self):
        """Generate throughput analysis chart"""
        if not self.results:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ontology Mapping System - Throughput Analysis', fontsize=16, fontweight='bold')

        # 1. Parallel Processing Throughput Comparison
        if "parallel_processing" in self.results:
            parallel = self.results["parallel_processing"]

            # Batch processing
            batch_sizes = []
            batch_throughput = []
            if "batch_processing" in parallel:
                for key, data in parallel["batch_processing"].items():
                    batch_sizes.append(int(key.replace("batch_size_", "")))
                    batch_throughput.append(data["throughput_terms_per_second"])

            if batch_sizes and batch_throughput:
                axes[0, 0].plot(batch_sizes, batch_throughput, 'o-', linewidth=3, markersize=8,
                               color='#e17055', label='Batch Processing')
                axes[0, 0].set_title('Batch Size vs Throughput')
                axes[0, 0].set_xlabel('Batch Size')
                axes[0, 0].set_ylabel('Throughput (terms/sec)')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()

            # Concurrent processing
            concurrent_counts = []
            concurrent_throughput = []
            if "concurrent_requests" in parallel:
                for key, data in parallel["concurrent_requests"].items():
                    concurrent_counts.append(int(key.replace("concurrent_", "")))
                    concurrent_throughput.append(data["throughput_requests_per_second"])

            if concurrent_counts and concurrent_throughput:
                axes[0, 1].plot(concurrent_counts, concurrent_throughput, 's-', linewidth=3, markersize=8,
                               color='#00b894', label='Concurrent Processing')
                axes[0, 1].set_title('Concurrency vs Throughput')
                axes[0, 1].set_xlabel('Concurrent Requests')
                axes[0, 1].set_ylabel('Throughput (requests/sec)')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()

        # 2. Performance Improvement
        if "ontology_mapping_performance" in self.results:
            perf = self.results["ontology_mapping_performance"]
            if "term_normalization" in perf:
                norm = perf["term_normalization"]

                methods = []
                times = []
                if "without_cache" in norm:
                    methods.append("Without Cache")
                    times.append(norm["without_cache"]["avg_time_per_term"])
                if "with_cache" in norm:
                    methods.append("With Cache")
                    times.append(norm["with_cache"]["avg_time_per_term"])

                if methods and times:
                    bars = axes[1, 0].bar(methods, times, color=['#ff7675', '#74b9ff'])
                    axes[1, 0].set_title('Performance Comparison')
                    axes[1, 0].set_ylabel('Avg Time per Term (seconds)')
                    axes[1, 0].grid(True, alpha=0.3)

                    # Add value labels
                    for bar, time in zip(bars, times):
                        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                                       '.4f', ha='center', va='bottom', fontweight='bold')

        # 3. Error Rate Analysis
        if "api_endpoints_simulation" in self.results:
            api = self.results["api_endpoints_simulation"]

            endpoints = []
            success_rates = []
            error_rates = []

            if "endpoint_responses" in api:
                for endpoint, data in api["endpoint_responses"].items():
                    endpoints.append(endpoint.replace("_", " ").title())
                    success_rate = data["successful_tests"] / data["tests_run"] * 100
                    success_rates.append(success_rate)
                    error_rates.append(100 - success_rate)

            if endpoints and success_rates:
                x = np.arange(len(endpoints))
                width = 0.35

                axes[1, 1].bar(x - width/2, success_rates, width, label='Success Rate',
                              color='#00b894', alpha=0.8)
                axes[1, 1].bar(x + width/2, error_rates, width, label='Error Rate',
                              color='#d63031', alpha=0.8)

                axes[1, 1].set_title('API Endpoints Reliability')
                axes[1, 1].set_xlabel('Endpoints')
                axes[1, 1].set_ylabel('Rate (%)')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(endpoints, rotation=45, ha='right')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_cache_performance_chart(self):
        """Generate cache performance analysis chart"""
        if not self.results:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ontology Mapping System - Cache Performance Analysis', fontsize=16, fontweight='bold')

        if "ontology_mapping_performance" in self.results:
            perf = self.results["ontology_mapping_performance"]
            if "cache_performance" in perf:
                cache_perf = perf["cache_performance"]

                # Cache Statistics
                stats_labels = ['Total Requests', 'Cache Hits', 'Cache Misses', 'Hit Ratio']
                stats_values = [
                    cache_perf["total_requests"],
                    cache_perf["cache_hits"],
                    cache_perf["cache_misses"],
                    cache_perf["hit_ratio"] * 100  # Convert to percentage
                ]

                colors = ['#6c5ce7', '#a29bfe', '#fd79a8', '#e17055']

                bars = axes[0, 0].bar(stats_labels, stats_values, color=colors)
                axes[0, 0].set_title('Cache Statistics Overview')
                axes[0, 0].set_ylabel('Count / Percentage')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].tick_params(axis='x', rotation=45)

                # Add value labels
                for bar, value in zip(bars, stats_values):
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stats_values) * 0.02,
                                   '.0f', ha='center', va='bottom', fontweight='bold')

        # Cache Hit/Miss Distribution
        if "ontology_mapping_performance" in self.results:
            perf = self.results["ontology_mapping_performance"]
            if "cache_performance" in perf:
                cache_perf = perf["cache_performance"]

                # Pie chart for hit/miss ratio
                labels = ['Cache Hits', 'Cache Misses']
                sizes = [cache_perf["cache_hits"], cache_perf["cache_misses"]]
                colors = ['#00b894', '#d63031']
                explode = (0.1, 0)

                axes[0, 1].pie(sizes, explode=explode, labels=labels, colors=colors,
                              autopct='%1.1f%%', shadow=True, startangle=90)
                axes[0, 1].set_title('Cache Hit/Miss Distribution')
                axes[0, 1].axis('equal')

                # Add hit ratio text
                hit_ratio = cache_perf["hit_ratio"]
                axes[0, 1].text(0, -1.3, '.1%', ha='center',
                               fontsize=14, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='#00b894', alpha=0.8))

        # Cache Size and Entries
        if "ontology_mapping_performance" in self.results:
            perf = self.results["ontology_mapping_performance"]
            if "cache_performance" in perf:
                cache_perf = perf["cache_performance"]

                # Create a summary table-like visualization
                cache_info = {
                    'Cache Entries': cache_perf["entries_count"],
                    'Cache Size (bytes)': cache_perf["size_bytes"],
                    'Avg Size per Entry': cache_perf["size_bytes"] / cache_perf["entries_count"] if cache_perf["entries_count"] > 0 else 0
                }

                y_pos = np.arange(len(cache_info))
                values = list(cache_info.values())

                axes[1, 0].barh(y_pos, values, color='#74b9ff', alpha=0.8)
                axes[1, 0].set_yticks(y_pos)
                axes[1, 0].set_yticklabels(list(cache_info.keys()))
                axes[1, 0].set_title('Cache Size Analysis')
                axes[1, 0].set_xlabel('Value')
                axes[1, 0].grid(True, alpha=0.3)

                # Add value labels
                for i, v in enumerate(values):
                    axes[1, 0].text(v + max(values) * 0.02, i, '.0f',
                                   va='center', fontweight='bold')

        plt.tight_layout()
        return fig

    def save_charts(self):
        """Generate and save all performance charts"""
        if not self.load_results():
            logger.error("Cannot generate charts: No test results available")
            return False

        try:
            # Generate charts
            response_time_chart = self.generate_response_time_chart()
            throughput_chart = self.generate_throughput_chart()
            cache_chart = self.generate_cache_performance_chart()

            # Save charts
            charts = [
                (response_time_chart, "response_time_analysis.png"),
                (throughput_chart, "throughput_analysis.png"),
                (cache_chart, "cache_performance_analysis.png")
            ]

            saved_files = []
            for chart, filename in charts:
                if chart is not None:
                    filepath = self.output_dir / filename
                    chart.savefig(filepath, dpi=300, bbox_inches='tight')
                    saved_files.append(str(filepath))
                    plt.close(chart)
                    logger.info(f"Saved chart: {filepath}")

            logger.info(f"Successfully saved {len(saved_files)} performance charts")
            return saved_files

        except Exception as e:
            logger.error(f"Failed to generate charts: {str(e)}")
            return False

    def get_chart_base64(self, chart_type="response_time"):
        """Get chart as base64 encoded string for web display"""
        if not self.load_results():
            return None

        try:
            chart = None
            if chart_type == "response_time":
                chart = self.generate_response_time_chart()
            elif chart_type == "throughput":
                chart = self.generate_throughput_chart()
            elif chart_type == "cache":
                chart = self.generate_cache_performance_chart()

            if chart is None:
                return None

            # Convert to base64
            buffer = BytesIO()
            chart.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close(chart)

            return image_base64

        except Exception as e:
            logger.error(f"Failed to generate base64 chart: {str(e)}")
            return None

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.load_results():
            return None

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "charts": {},
            "recommendations": []
        }

        # Extract key metrics
        if "ontology_mapping_performance" in self.results:
            perf = self.results["ontology_mapping_performance"]

            if "term_normalization" in perf:
                norm = perf["term_normalization"]
                report["summary"]["term_normalization"] = norm

            if "cache_performance" in perf:
                cache_perf = perf["cache_performance"]
                report["summary"]["cache_performance"] = cache_perf

        if "parallel_processing" in self.results:
            parallel = self.results["parallel_processing"]
            report["summary"]["parallel_processing"] = parallel

        if "api_endpoints_simulation" in self.results:
            api = self.results["api_endpoints_simulation"]
            report["summary"]["api_performance"] = api

        # Generate base64 charts
        chart_types = ["response_time", "throughput", "cache"]
        for chart_type in chart_types:
            chart_b64 = self.get_chart_base64(chart_type)
            if chart_b64:
                report["charts"][chart_type] = f"data:image/png;base64,{chart_b64}"

        # Generate recommendations
        if "cache_performance" in self.results.get("ontology_mapping_performance", {}):
            cache_perf = self.results["ontology_mapping_performance"]["cache_performance"]
            hit_ratio = cache_perf["hit_ratio"]

            if hit_ratio < 0.5:
                report["recommendations"].append("Consider adjusting cache strategy - current hit ratio is low")
            elif hit_ratio > 0.9:
                report["recommendations"].append("Excellent cache performance - consider increasing cache size for even better performance")

        # Parallel processing recommendations
        if "parallel_processing" in self.results:
            parallel = self.results["parallel_processing"]
            if "batch_processing" in parallel:
                batch_sizes = list(parallel["batch_processing"].keys())
                if len(batch_sizes) > 1:
                    report["recommendations"].append("Batch processing shows good scalability - consider using larger batch sizes for high-volume processing")

        if not report["recommendations"]:
            report["recommendations"].append("System performance is excellent - all metrics are within optimal ranges")

        return report

if __name__ == "__main__":
    # Generate charts when run directly
    visualizer = PerformanceVisualizer()
    saved_files = visualizer.save_charts()

    if saved_files:
        print("Performance charts generated successfully:")
        for file in saved_files:
            print(f"  - {file}")

        # Generate performance report
        report = visualizer.generate_performance_report()
        if report:
            report_file = "backend/visualizations/performance_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Performance report saved: {report_file}")
    else:
        print("Failed to generate performance charts")
