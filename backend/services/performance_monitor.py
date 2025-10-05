"""
Performance Monitor for CortexMD Knowledge Graph
Provides real-time monitoring, alerting, and performance analytics
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psutil
import aiofiles
from pathlib import Path
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import socket

from services.neo4j_service import Neo4jService
from services.enhanced_knowledge_graph import EnhancedKnowledgeGraphService

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    query_count: int = 0
    avg_query_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    neo4j_heap_used: float = 0.0
    neo4j_heap_max: float = 0.0
    cache_hit_ratio: float = 0.0
    active_connections: int = 0
    error_count: int = 0
    slow_queries: List[Dict] = field(default_factory=list)

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric: str
    condition: str  # '>', '<', '>=', '<=', '=='
    threshold: float
    duration_seconds: int = 60
    severity: str = "warning"  # warning, error, critical
    enabled: bool = True
    description: str = ""
    cooldown_minutes: int = 5

@dataclass
class Alert:
    """Active alert instance"""
    rule_name: str
    severity: str
    message: str
    timestamp: datetime
    value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class PerformanceMonitor:
    """Real-time performance monitoring and alerting system"""

    def __init__(self, neo4j_service: Neo4jService = None,
                 knowledge_graph_service: EnhancedKnowledgeGraphService = None):
        """
        Initialize performance monitor

        Args:
            neo4j_service: Neo4j service instance
            knowledge_graph_service: Enhanced knowledge graph service instance
        """
        self.neo4j_service = neo4j_service or Neo4jService()
        self.kg_service = knowledge_graph_service

        # Configuration
        self.monitoring_interval = 30  # seconds
        self.metrics_history_size = 1000
        self.alert_history_size = 500

        # Storage
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_cooldowns: Dict[str, datetime] = {}

        # Alert rules
        self.alert_rules = self._initialize_alert_rules()

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.last_collection_time = datetime.now()

        # Notification channels
        self.email_config = {
            "enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "",
            "password": "",
            "recipients": []
        }

        self.slack_config = {
            "enabled": False,
            "webhook_url": "",
            "channel": "#alerts"
        }

        # Data directories
        self.monitoring_dir = Path("backend/monitoring")
        self.metrics_file = self.monitoring_dir / "performance_metrics.json"
        self.alerts_file = self.monitoring_dir / "alerts.json"

        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialized Performance Monitor")

    def _initialize_alert_rules(self) -> Dict[str, AlertRule]:
        """Initialize default alert rules"""
        return {
            "high_memory_usage": AlertRule(
                name="High Memory Usage",
                metric="memory_usage",
                condition=">",
                threshold=85.0,
                severity="warning",
                description="System memory usage is above 85%"
            ),
            "neo4j_heap_high": AlertRule(
                name="Neo4j Heap Usage High",
                metric="neo4j_heap_used_percent",
                condition=">",
                threshold=80.0,
                severity="error",
                description="Neo4j heap usage is above 80%"
            ),
            "slow_query_rate": AlertRule(
                name="High Slow Query Rate",
                metric="slow_query_rate",
                condition=">",
                threshold=10.0,
                severity="warning",
                description="More than 10% of queries are slow (>1s)"
            ),
            "cache_hit_ratio_low": AlertRule(
                name="Low Cache Hit Ratio",
                metric="cache_hit_ratio",
                condition="<",
                threshold=70.0,
                severity="info",
                description="Cache hit ratio dropped below 70%"
            ),
            "high_error_rate": AlertRule(
                name="High Error Rate",
                metric="error_rate",
                condition=">",
                threshold=5.0,
                severity="error",
                description="Error rate is above 5%"
            ),
            "neo4j_connections_high": AlertRule(
                name="High Neo4j Connections",
                metric="neo4j_connections",
                condition=">",
                threshold=50,
                severity="warning",
                description="Neo4j has more than 50 active connections"
            )
        }

    async def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return

        self.is_monitoring = True
        logger.info("Starting performance monitoring...")

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        # Load existing data
        await self._load_persistent_data()

        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Create new event loop for async operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Collect metrics
                metrics = loop.run_until_complete(self._collect_metrics())

                # Store metrics
                self._store_metrics(metrics)

                # Check alert rules
                loop.run_until_complete(self._check_alerts())

                # Cleanup old data
                self._cleanup_old_data()

                # Save persistent data
                loop.run_until_complete(self._save_persistent_data())

                loop.close()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")

            # Wait for next collection
            time.sleep(self.monitoring_interval)

    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        timestamp = datetime.now()

        # System metrics
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)

        # Neo4j metrics
        neo4j_metrics = await self._collect_neo4j_metrics()

        # Application metrics
        app_metrics = await self._collect_application_metrics()

        # Slow queries (queries taking >1 second)
        slow_queries = await self._collect_slow_queries()

        metrics = PerformanceMetrics(
            timestamp=timestamp,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            neo4j_heap_used=neo4j_metrics.get("heap_used", 0.0),
            neo4j_heap_max=neo4j_metrics.get("heap_max", 0.0),
            cache_hit_ratio=app_metrics.get("cache_hit_ratio", 0.0),
            active_connections=neo4j_metrics.get("active_connections", 0),
            error_count=app_metrics.get("error_count", 0),
            slow_queries=slow_queries
        )

        # Calculate derived metrics
        if neo4j_metrics.get("heap_max", 0) > 0:
            metrics.neo4j_heap_used = (neo4j_metrics["heap_used"] / neo4j_metrics["heap_max"]) * 100

        return metrics

    async def _collect_neo4j_metrics(self) -> Dict[str, Any]:
        """Collect Neo4j-specific metrics"""
        metrics = {}

        try:
            async with self.neo4j_service.driver.session() as session:
                # Heap usage
                heap_query = """
                CALL dbms.listConfig('dbms.memory.heap.max_size')
                YIELD name, value
                RETURN value as heap_max
                """
                heap_result = await session.run(heap_query)
                heap_record = await heap_result.single()
                if heap_record:
                    heap_max_str = heap_record["heap_max"]
                    # Parse heap size (e.g., "4G" -> 4 * 1024 * 1024 * 1024)
                    if heap_max_str.endswith('G'):
                        metrics["heap_max"] = float(heap_max_str[:-1]) * 1024 * 1024 * 1024
                    elif heap_max_str.endswith('M'):
                        metrics["heap_max"] = float(heap_max_str[:-1]) * 1024 * 1024

                # Active connections
                conn_query = """
                CALL dbms.listConnections()
                YIELD connectionId, connectTime, connector
                WHERE connector = 'bolt'
                RETURN count(*) as active_connections
                """
                conn_result = await session.run(conn_query)
                conn_record = await conn_result.single()
                metrics["active_connections"] = conn_record["active_connections"] if conn_record else 0

                # Query statistics
                query_stats_query = """
                CALL dbms.queryJmx('org.neo4j.dbms.query:*')
                YIELD name, attributes
                WHERE name CONTAINS 'Query'
                RETURN name, attributes
                """
                stats_result = await session.run(query_stats_query)
                async for record in stats_result:
                    if "executingQueries" in record["name"]:
                        metrics["executing_queries"] = record["attributes"].get("Count", 0)

        except Exception as e:
            logger.debug(f"Failed to collect Neo4j metrics: {str(e)}")
            metrics = {"error": str(e)}

        return metrics

    async def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics"""
        metrics = {}

        try:
            # Cache hit ratio (if intelligent cache is available)
            if hasattr(self.neo4j_service, 'cache'):
                cache_stats = await self.neo4j_service.cache.get_stats()
                total_requests = cache_stats.get("hits", 0) + cache_stats.get("misses", 0)
                if total_requests > 0:
                    metrics["cache_hit_ratio"] = (cache_stats.get("hits", 0) / total_requests) * 100

            # Error count from recent logs (simplified)
            metrics["error_count"] = 0  # Would be populated from actual error tracking

        except Exception as e:
            logger.debug(f"Failed to collect application metrics: {str(e)}")

        return metrics

    async def _collect_slow_queries(self) -> List[Dict]:
        """Collect slow query information"""
        slow_queries = []

        try:
            async with self.neo4j_service.driver.session() as session:
                # Get queries running longer than 1 second
                slow_query_cypher = """
                CALL dbms.listQueries()
                YIELD queryId, query, parameters, startTime, elapsedTimeMillis
                WHERE elapsedTimeMillis > 1000
                RETURN queryId, query, elapsedTimeMillis, startTime
                ORDER BY elapsedTimeMillis DESC
                LIMIT 10
                """

                result = await session.run(slow_query_cypher)
                async for record in result:
                    slow_queries.append({
                        "query_id": record["queryId"],
                        "query": record["query"][:200] + "..." if len(record["query"]) > 200 else record["query"],
                        "elapsed_ms": record["elapsedTimeMillis"],
                        "start_time": record["startTime"]
                    })

        except Exception as e:
            logger.debug(f"Failed to collect slow queries: {str(e)}")

        return slow_queries

    def _store_metrics(self, metrics: PerformanceMetrics):
        """Store metrics in history"""
        self.metrics_history.append(metrics)

        # Maintain history size limit
        if len(self.metrics_history) > self.metrics_history_size:
            self.metrics_history = self.metrics_history[-self.metrics_history_size:]

    async def _check_alerts(self):
        """Check all alert rules against current metrics"""
        if not self.metrics_history:
            return

        current_metrics = self.metrics_history[-1]

        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            # Check cooldown
            if rule_name in self.alert_cooldowns:
                if datetime.now() < self.alert_cooldowns[rule_name]:
                    continue
                else:
                    del self.alert_cooldowns[rule_name]

            # Evaluate rule condition
            if self._evaluate_alert_condition(rule, current_metrics):
                await self._trigger_alert(rule, current_metrics)

    def _evaluate_alert_condition(self, rule: AlertRule, metrics: PerformanceMetrics) -> bool:
        """Evaluate if an alert condition is met"""
        metric_value = self._get_metric_value(rule.metric, metrics)

        if metric_value is None:
            return False

        if rule.condition == ">":
            return metric_value > rule.threshold
        elif rule.condition == "<":
            return metric_value < rule.threshold
        elif rule.condition == ">=":
            return metric_value >= rule.threshold
        elif rule.condition == "<=":
            return metric_value <= rule.threshold
        elif rule.condition == "==":
            return abs(metric_value - rule.threshold) < 0.001

        return False

    def _get_metric_value(self, metric_name: str, metrics: PerformanceMetrics) -> Optional[float]:
        """Get metric value from metrics object"""
        metric_map = {
            "memory_usage": metrics.memory_usage,
            "cpu_usage": metrics.cpu_usage,
            "neo4j_heap_used_percent": metrics.neo4j_heap_used,
            "cache_hit_ratio": metrics.cache_hit_ratio,
            "neo4j_connections": metrics.active_connections,
            "error_rate": (metrics.error_count / max(metrics.query_count, 1)) * 100,
            "slow_query_rate": (len(metrics.slow_queries) / max(metrics.query_count, 1)) * 100
        }

        return metric_map.get(metric_name)

    async def _trigger_alert(self, rule: AlertRule, metrics: PerformanceMetrics):
        """Trigger an alert"""
        alert_key = f"{rule.name}_{rule.metric}"

        # Check if alert is already active
        if alert_key in self.active_alerts:
            return

        # Create new alert
        metric_value = self._get_metric_value(rule.metric, metrics)

        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            message=f"{rule.description} (Current: {metric_value:.2f}, Threshold: {rule.threshold})",
            timestamp=datetime.now(),
            value=metric_value,
            threshold=rule.threshold
        )

        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)

        # Maintain alert history size
        if len(self.alert_history) > self.alert_history_size:
            self.alert_history = self.alert_history[-self.alert_history_size:]

        # Set cooldown
        self.alert_cooldowns[rule.name] = datetime.now() + timedelta(minutes=rule.cooldown_minutes)

        # Send notifications
        await self._send_alert_notifications(alert)

        logger.warning(f"Alert triggered: {rule.name} - {alert.message}")

    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        # Email notifications
        if self.email_config["enabled"]:
            await self._send_email_alert(alert)

        # Slack notifications
        if self.slack_config["enabled"]:
            await self._send_slack_alert(alert)

        # Log alert
        logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.message}")

    async def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config["username"]
            msg['To'] = ", ".join(self.email_config["recipients"])
            msg['Subject'] = f"CortexMD Alert: {alert.rule_name}"

            body = f"""
            CortexMD Performance Alert

            Severity: {alert.severity.upper()}
            Rule: {alert.rule_name}
            Message: {alert.message}
            Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

            This is an automated message from CortexMD monitoring system.
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
            server.starttls()
            server.login(self.email_config["username"], self.email_config["password"])
            text = msg.as_string()
            server.sendmail(self.email_config["username"], self.email_config["recipients"], text)
            server.quit()

        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")

    async def _send_slack_alert(self, alert: Alert):
        """Send alert via Slack"""
        try:
            payload = {
                "channel": self.slack_config["channel"],
                "text": f"🚨 *CortexMD Alert*\n*Severity:* {alert.severity.upper()}\n*Rule:* {alert.rule_name}\n*Message:* {alert.message}\n*Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            }

            response = requests.post(self.slack_config["webhook_url"], json=payload)
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")

    def resolve_alert(self, alert_key: str):
        """Manually resolve an alert"""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_at = datetime.now()

            logger.info(f"Alert resolved: {alert.rule_name}")

    async def _load_persistent_data(self):
        """Load persistent monitoring data"""
        try:
            if self.metrics_file.exists():
                async with aiofiles.open(self.metrics_file, 'r') as f:
                    data = json.loads(await f.read())
                    # Would load metrics history here

            if self.alerts_file.exists():
                async with aiofiles.open(self.alerts_file, 'r') as f:
                    data = json.loads(await f.read())
                    # Would load alert history here

        except Exception as e:
            logger.debug(f"Failed to load persistent data: {str(e)}")

    async def _save_persistent_data(self):
        """Save monitoring data persistently"""
        try:
            # Save recent metrics
            recent_metrics = self.metrics_history[-100:] if len(self.metrics_history) > 100 else self.metrics_history
            metrics_data = {
                "metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "memory_usage": m.memory_usage,
                        "cpu_usage": m.cpu_usage,
                        "neo4j_heap_used": m.neo4j_heap_used,
                        "cache_hit_ratio": m.cache_hit_ratio,
                        "active_connections": m.active_connections,
                        "error_count": m.error_count
                    } for m in recent_metrics
                ]
            }

            async with aiofiles.open(self.metrics_file, 'w') as f:
                await f.write(json.dumps(metrics_data, indent=2))

            # Save alerts
            alerts_data = {
                "active_alerts": [
                    {
                        "rule_name": alert.rule_name,
                        "severity": alert.severity,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "value": alert.value,
                        "threshold": alert.threshold,
                        "resolved": alert.resolved
                    } for alert in self.active_alerts.values()
                ],
                "alert_history": [
                    {
                        "rule_name": alert.rule_name,
                        "severity": alert.severity,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "resolved": alert.resolved
                    } for alert in self.alert_history[-100:]
                ]
            }

            async with aiofiles.open(self.alerts_file, 'w') as f:
                await f.write(json.dumps(alerts_data, indent=2))

        except Exception as e:
            logger.debug(f"Failed to save persistent data: {str(e)}")

    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        cutoff_time = datetime.now() - timedelta(hours=24)

        # Clean up old alerts
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]

    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        if not self.metrics_history:
            return {"error": "No metrics available"}

        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]

        if not recent_metrics:
            return {"error": f"No metrics available for the last {hours} hours"}

        # Calculate averages
        summary = {
            "period_hours": hours,
            "metrics_count": len(recent_metrics),
            "average_memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "average_cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "average_cache_hit_ratio": sum(m.cache_hit_ratio for m in recent_metrics) / len(recent_metrics),
            "max_memory_usage": max(m.memory_usage for m in recent_metrics),
            "max_cpu_usage": max(m.cpu_usage for m in recent_metrics),
            "total_errors": sum(m.error_count for m in recent_metrics),
            "total_slow_queries": sum(len(m.slow_queries) for m in recent_metrics),
            "active_alerts_count": len(self.active_alerts)
        }

        return summary

    def configure_notifications(self, email_config: Dict = None, slack_config: Dict = None):
        """Configure notification channels"""
        if email_config:
            self.email_config.update(email_config)

        if slack_config:
            self.slack_config.update(slack_config)

    def add_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule"""
        self.alert_rules[rule.name] = rule

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.metrics_history:
            return {"status": "unknown", "message": "No metrics available"}

        latest_metrics = self.metrics_history[-1]

        # Determine health based on various metrics
        issues = []

        if latest_metrics.memory_usage > 90:
            issues.append("High memory usage")
        if latest_metrics.cpu_usage > 90:
            issues.append("High CPU usage")
        if latest_metrics.neo4j_heap_used > 85:
            issues.append("High Neo4j heap usage")
        if latest_metrics.cache_hit_ratio < 60:
            issues.append("Low cache hit ratio")
        if len(self.active_alerts) > 0:
            issues.append(f"{len(self.active_alerts)} active alerts")

        if issues:
            status = "warning" if len(issues) < 3 else "error"
            message = f"System has {len(issues)} issues: {', '.join(issues)}"
        else:
            status = "healthy"
            message = "All systems operating normally"

        return {
            "status": status,
            "message": message,
            "issues": issues,
            "active_alerts": len(self.active_alerts),
            "last_check": latest_metrics.timestamp.isoformat()
        }
