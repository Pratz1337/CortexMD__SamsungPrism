"""
Knowledge Graph Visualizer for CortexMD
Provides interactive visualizations for graph reasoning results and knowledge exploration
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import base64
from io import BytesIO
import aiofiles

from services.neo4j_service import Neo4jService
from services.enhanced_knowledge_graph import EnhancedKnowledgeGraphService

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    width: int = 1200
    height: int = 800
    show_labels: bool = True
    max_nodes: int = 100
    layout_algorithm: str = "force"  # force, circular, hierarchical
    color_scheme: str = "viridis"
    interactive: bool = True

class KnowledgeGraphVisualizer:
    """Advanced visualizer for knowledge graph exploration and reasoning results"""

    def __init__(self, neo4j_service: Neo4jService = None,
                 knowledge_graph_service: EnhancedKnowledgeGraphService = None):
        """
        Initialize knowledge graph visualizer

        Args:
            neo4j_service: Neo4j service instance
            knowledge_graph_service: Enhanced knowledge graph service instance
        """
        self.neo4j_service = neo4j_service or Neo4jService()
        self.kg_service = knowledge_graph_service

        # Visualization configuration
        self.config = VisualizationConfig()

        # Output directories
        self.visualizations_dir = Path("backend/visualizations")
        self.static_dir = self.visualizations_dir / "static"
        self.dynamic_dir = self.visualizations_dir / "dynamic"

        for dir_path in [self.visualizations_dir, self.static_dir, self.dynamic_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Color schemes for different node types
        self.node_colors = {
            "Disease": "#FF6B6B",
            "Symptom": "#4ECDC4",
            "Drug": "#45B7D1",
            "Clinical_Drug": "#45B7D1",
            "Pharmacologic_Substance": "#96CEB4",
            "Clinical_Attribute": "#FFEAA7",
            "Finding": "#DDA0DD",
            "Procedure": "#98D8C8",
            "Anatomy": "#F7DC6F",
            "Risk_Factor": "#BB8FCE"
        }

        logger.info("Initialized Knowledge Graph Visualizer")

    async def __aenter__(self):
        """Async context manager entry"""
        if hasattr(self.neo4j_service, '__aenter__'):
            await self.neo4j_service.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if hasattr(self.neo4j_service, '__aexit__'):
            await self.neo4j_service.__aexit__(exc_type, exc_val, exc_tb)

    async def visualize_symptom_clusters(self, symptom_clusters: List) -> Dict[str, Any]:
        """
        Visualize symptom clusters as an interactive network

        Args:
            symptom_clusters: List of symptom cluster objects

        Returns:
            Dictionary containing visualization data and HTML
        """
        if not symptom_clusters:
            return {"error": "No symptom clusters to visualize"}

        # Create network graph
        G = nx.Graph()

        # Add nodes and edges for each cluster
        for i, cluster in enumerate(symptom_clusters):
            cluster_id = f"cluster_{i}"

            # Add cluster center node
            G.add_node(cluster_id,
                      label=f"Cluster {i+1}",
                      type="cluster",
                      size=len(cluster.symptoms) * 10,
                      color="#FF6B6B")

            # Add symptom nodes and connect to cluster
            for symptom in cluster.symptoms:
                symptom_id = f"symptom_{symptom.replace(' ', '_')}"
                G.add_node(symptom_id,
                          label=symptom,
                          type="symptom",
                          size=8,
                          color="#4ECDC4")
                G.add_edge(cluster_id, symptom_id, weight=1)

            # Add disease associations
            for disease in cluster.common_diseases[:3]:  # Limit to top 3
                disease_id = f"disease_{disease['name'].replace(' ', '_')}"
                G.add_node(disease_id,
                          label=disease['name'],
                          type="disease",
                          size=disease['frequency'] * 5,
                          color="#45B7D1")
                G.add_edge(cluster_id, disease_id, weight=disease['frequency'])

        # Generate positions
        if self.config.layout_algorithm == "force":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif self.config.layout_algorithm == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)

        # Create Plotly figure
        fig = self._create_network_figure(G, pos, "Symptom Clusters Network")

        # Convert to HTML
        html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')

        # Save visualization
        viz_id = str(uuid.uuid4())
        filename = f"symptom_clusters_{viz_id}.html"
        filepath = self.dynamic_dir / filename

        async with aiofiles.open(filepath, 'w') as f:
            await f.write(html_content)

        return {
            "visualization_id": viz_id,
            "filename": filename,
            "filepath": str(filepath),
            "html_content": html_content,
            "graph_stats": {
                "nodes": len(G.nodes()),
                "edges": len(G.edges()),
                "clusters": len(symptom_clusters)
            }
        }

    async def visualize_drug_interactions(self, drug_interactions: List) -> Dict[str, Any]:
        """
        Visualize drug interaction network

        Args:
            drug_interactions: List of drug interaction objects

        Returns:
            Dictionary containing visualization data and HTML
        """
        if not drug_interactions:
            return {"error": "No drug interactions to visualize"}

        # Create network graph
        G = nx.Graph()

        # Add nodes for drugs
        drugs_added = set()
        for interaction in drug_interactions:
            drug1 = interaction.drug1
            drug2 = interaction.drug2

            # Add drug nodes
            if drug1 not in drugs_added:
                G.add_node(drug1, label=drug1, type="drug", color="#45B7D1")
                drugs_added.add(drug1)

            if drug2 not in drugs_added:
                G.add_node(drug2, label=drug2, type="drug", color="#45B7D1")
                drugs_added.add(drug2)

            # Add interaction edge
            edge_color = self._get_interaction_color(interaction.severity)
            G.add_edge(drug1, drug2,
                      weight=self._get_severity_weight(interaction.severity),
                      color=edge_color,
                      severity=interaction.severity,
                      description=interaction.description)

        # Generate layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Create figure
        fig = self._create_network_figure(G, pos, "Drug Interaction Network")

        # Add severity legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='#FF6B6B'),
            name='Severe'
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='#FFEAA7'),
            name='Moderate'
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='#4ECDC4'),
            name='Minor'
        ))

        # Convert to HTML
        html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')

        # Save visualization
        viz_id = str(uuid.uuid4())
        filename = f"drug_interactions_{viz_id}.html"
        filepath = self.dynamic_dir / filename

        async with aiofiles.open(filepath, 'w') as f:
            await f.write(html_content)

        return {
            "visualization_id": viz_id,
            "filename": filename,
            "filepath": str(filepath),
            "html_content": html_content,
            "graph_stats": {
                "drugs": len(drugs_added),
                "interactions": len(drug_interactions)
            }
        }

    async def visualize_comorbidities(self, comorbidity_analysis: Any) -> Dict[str, Any]:
        """
        Visualize comorbidity relationships

        Args:
            comorbidity_analysis: Comorbidity analysis object

        Returns:
            Dictionary containing visualization data and HTML
        """
        if not comorbidity_analysis or not comorbidity_analysis.comorbidities:
            return {"error": "No comorbidity data to visualize"}

        # Create network graph
        G = nx.Graph()

        # Add primary condition
        primary = comorbidity_analysis.primary_condition
        G.add_node(primary,
                  label=primary,
                  type="primary_condition",
                  color="#FF6B6B",
                  size=20)

        # Add comorbidity nodes and edges
        for comorbidity in comorbidity_analysis.comorbidities:
            condition_name = comorbidity["name"]
            prevalence = comorbidity["prevalence"]

            G.add_node(condition_name,
                      label=condition_name,
                      type="comorbidity",
                      color="#4ECDC4",
                      size=prevalence * 15)

            G.add_edge(primary, condition_name,
                      weight=prevalence,
                      prevalence=prevalence,
                      evidence=comorbidity["evidence_strength"])

        # Add risk factor nodes
        for risk_factor in comorbidity_analysis.risk_factors[:5]:  # Limit to top 5
            G.add_node(risk_factor,
                      label=risk_factor,
                      type="risk_factor",
                      color="#FFEAA7",
                      size=10)

            # Connect risk factors to primary condition
            G.add_edge(primary, risk_factor,
                      weight=0.5,
                      relationship="risk_factor")

        # Generate layout
        pos = nx.spring_layout(G, k=1.5, iterations=50)

        # Create figure
        fig = self._create_network_figure(G, pos, "Comorbidity Network")

        # Convert to HTML
        html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')

        # Save visualization
        viz_id = str(uuid.uuid4())
        filename = f"comorbidities_{viz_id}.html"
        filepath = self.dynamic_dir / filename

        async with aiofiles.open(filepath, 'w') as f:
            await f.write(html_content)

        return {
            "visualization_id": viz_id,
            "filename": filename,
            "filepath": str(filepath),
            "html_content": html_content,
            "graph_stats": {
                "primary_condition": primary,
                "comorbidities": len(comorbidity_analysis.comorbidities),
                "risk_factors": len(comorbidity_analysis.risk_factors)
            }
        }

    async def visualize_knowledge_graph_overview(self, max_nodes: int = 50) -> Dict[str, Any]:
        """
        Create an overview visualization of the entire knowledge graph

        Args:
            max_nodes: Maximum number of nodes to include

        Returns:
            Dictionary containing visualization data and HTML
        """
        # Query graph overview
        query = f"""
        MATCH (n)-[r]->(m)
        RETURN DISTINCT labels(n)[0] as source_type,
               type(r) as relationship_type,
               labels(m)[0] as target_type,
               count(*) as relationship_count
        ORDER BY relationship_count DESC
        LIMIT {max_nodes}
        """

        async with self.neo4j_service.driver.session() as session:
            result = await session.run(query)
            relationships = []

            async for record in result:
                relationships.append({
                    "source": record["source_type"] or "Unknown",
                    "relationship": record["relationship_type"],
                    "target": record["target_type"] or "Unknown",
                    "count": record["relationship_count"]
                })

        if not relationships:
            return {"error": "No graph data to visualize"}

        # Create aggregated network
        G = nx.DiGraph()

        # Add nodes for each type
        node_types = set()
        for rel in relationships:
            node_types.add(rel["source"])
            node_types.add(rel["target"])

        for node_type in node_types:
            G.add_node(node_type,
                      label=node_type,
                      type="entity_type",
                      color=self.node_colors.get(node_type, "#95A5A6"),
                      size=15)

        # Add edges
        for rel in relationships:
            G.add_edge(rel["source"], rel["target"],
                      weight=rel["count"],
                      label=f"{rel['relationship']} ({rel['count']})",
                      count=rel["count"])

        # Generate layout
        try:
            pos = nx.spring_layout(G, k=2, iterations=100)
        except:
            pos = nx.random_layout(G)

        # Create figure
        fig = go.Figure()

        # Add edges
        edge_x = []
        edge_y = []
        edge_text = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(edge[2].get('label', ''))

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=2, color='#888'),
            hoverinfo='text',
            text=edge_text,
            showlegend=False
        ))

        # Add nodes
        for node_type in G.nodes():
            x, y = pos[node_type]
            node_data = G.nodes[node_type]

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text' if self.config.show_labels else 'markers',
                marker=dict(
                    size=node_data.get('size', 15),
                    color=node_data.get('color', '#95A5A6'),
                    line=dict(width=2, color='#fff')
                ),
                text=[node_data.get('label', node_type)] if self.config.show_labels else [],
                textposition="top center",
                hovertext=f"{node_type}",
                name=node_type,
                showlegend=True
            ))

        # Update layout
        fig.update_layout(
            title="Knowledge Graph Overview",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )

        # Convert to HTML
        html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')

        # Save visualization
        viz_id = str(uuid.uuid4())
        filename = f"graph_overview_{viz_id}.html"
        filepath = self.dynamic_dir / filename

        async with aiofiles.open(filepath, 'w') as f:
            await f.write(html_content)

        return {
            "visualization_id": viz_id,
            "filename": filename,
            "filepath": str(filepath),
            "html_content": html_content,
            "graph_stats": {
                "node_types": len(node_types),
                "relationships": len(relationships),
                "total_connections": sum(r["count"] for r in relationships)
            }
        }

    def _create_network_figure(self, G: nx.Graph, pos: Dict, title: str) -> go.Figure:
        """Create a Plotly figure from NetworkX graph"""
        fig = go.Figure()

        # Add edges
        edge_x = []
        edge_y = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            showlegend=False
        ))

        # Add nodes
        for node in G.nodes():
            x, y = pos[node]
            node_data = G.nodes[node]

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text' if self.config.show_labels else 'markers',
                marker=dict(
                    size=node_data.get('size', 10),
                    color=node_data.get('color', '#95A5A6'),
                    line=dict(width=2, color='#fff')
                ),
                text=[node_data.get('label', node)] if self.config.show_labels else [],
                textposition="top center",
                hovertext=f"{node}: {node_data.get('type', 'unknown')}",
                name=node_data.get('type', 'unknown'),
                showlegend=True
            ))

        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )

        return fig

    def _get_interaction_color(self, severity: str) -> str:
        """Get color for drug interaction severity"""
        color_map = {
            "severe": "#FF6B6B",
            "major": "#FF8C69",
            "moderate": "#FFEAA7",
            "minor": "#4ECDC4"
        }
        return color_map.get(severity.lower(), "#95A5A6")

    def _get_severity_weight(self, severity: str) -> float:
        """Get edge weight for drug interaction severity"""
        weight_map = {
            "severe": 5.0,
            "major": 3.0,
            "moderate": 2.0,
            "minor": 1.0
        }
        return weight_map.get(severity.lower(), 1.0)

    async def create_performance_dashboard(self, metrics_data: List[Dict]) -> Dict[str, Any]:
        """
        Create performance dashboard visualization

        Args:
            metrics_data: List of performance metrics

        Returns:
            Dictionary containing dashboard visualization
        """
        if not metrics_data:
            return {"error": "No metrics data to visualize"}

        # Convert to DataFrame
        df = pd.DataFrame(metrics_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Memory & CPU Usage', 'Neo4j Performance',
                          'Cache Performance', 'Error Rates'),
            specs=[[{"secondary_y": True}, {}],
                   [{}, {}]]
        )

        # Memory and CPU usage
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['memory_usage'],
                      name="Memory Usage", line=dict(color='#FF6B6B')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cpu_usage'],
                      name="CPU Usage", line=dict(color='#4ECDC4')),
            row=1, col=1, secondary_y=True
        )

        # Neo4j performance
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['neo4j_heap_used'],
                      name="Neo4j Heap", line=dict(color='#45B7D1')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['active_connections'],
                      name="Active Connections", line=dict(color='#96CEB4')),
            row=1, col=2
        )

        # Cache performance
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cache_hit_ratio'],
                      name="Cache Hit Ratio", line=dict(color='#FFEAA7')),
            row=2, col=1
        )

        # Error rates
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['error_count'],
                      name="Error Count", line=dict(color='#DDA0DD')),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title="CortexMD Performance Dashboard",
            height=800,
            showlegend=True
        )

        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)

        fig.update_yaxes(title_text="Usage (%)", row=1, col=1)
        fig.update_yaxes(title_text="Usage (%)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Usage (%)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Hit Ratio (%)", row=2, col=1)
        fig.update_yaxes(title_text="Errors", row=2, col=2)

        # Convert to HTML
        html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')

        # Save dashboard
        viz_id = str(uuid.uuid4())
        filename = f"performance_dashboard_{viz_id}.html"
        filepath = self.dynamic_dir / filename

        async with aiofiles.open(filepath, 'w') as f:
            await f.write(html_content)

        return {
            "visualization_id": viz_id,
            "filename": filename,
            "filepath": str(filepath),
            "html_content": html_content,
            "dashboard_type": "performance",
            "metrics_count": len(metrics_data)
        }

    async def create_reasoning_flowchart(self, reasoning_paths: List[str]) -> Dict[str, Any]:
        """
        Create a flowchart visualization of the reasoning process

        Args:
            reasoning_paths: List of reasoning steps

        Returns:
            Dictionary containing flowchart visualization
        """
        if not reasoning_paths:
            return {"error": "No reasoning paths to visualize"}

        # Create flowchart figure
        fig = go.Figure()

        # Create nodes for each reasoning step
        y_positions = list(range(len(reasoning_paths)))
        x_positions = [0] * len(reasoning_paths)

        # Add nodes
        for i, step in enumerate(reasoning_paths):
            fig.add_trace(go.Scatter(
                x=[x_positions[i]],
                y=[y_positions[i]],
                mode='markers+text',
                marker=dict(size=40, color='#4ECDC4', symbol='circle'),
                text=[f"Step {i+1}"],
                textposition="middle center",
                hovertext=step,
                showlegend=False
            ))

        # Add edges (connecting lines)
        for i in range(len(reasoning_paths) - 1):
            fig.add_trace(go.Scatter(
                x=[x_positions[i], x_positions[i+1]],
                y=[y_positions[i], y_positions[i+1]],
                mode='lines',
                line=dict(width=3, color='#888'),
                showlegend=False
            ))

        # Add step descriptions as annotations
        annotations = []
        for i, step in enumerate(reasoning_paths):
            annotations.append(dict(
                x=x_positions[i] + 0.1,
                y=y_positions[i],
                text=step[:50] + "..." if len(step) > 50 else step,
                showarrow=False,
                xanchor='left',
                font=dict(size=10)
            ))

        fig.update_layout(
            title="Clinical Reasoning Flow",
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=max(400, len(reasoning_paths) * 80)
        )

        # Convert to HTML
        html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')

        # Save flowchart
        viz_id = str(uuid.uuid4())
        filename = f"reasoning_flow_{viz_id}.html"
        filepath = self.dynamic_dir / filename

        async with aiofiles.open(filepath, 'w') as f:
            await f.write(html_content)

        return {
            "visualization_id": viz_id,
            "filename": filename,
            "filepath": str(filepath),
            "html_content": html_content,
            "flowchart_type": "reasoning",
            "steps_count": len(reasoning_paths)
        }

    async def export_visualization_report(self, visualizations: List[Dict]) -> str:
        """
        Export a comprehensive visualization report

        Args:
            visualizations: List of visualization data

        Returns:
            Path to exported report
        """
        report_data = {
            "report_title": "CortexMD Knowledge Graph Visualization Report",
            "generated_at": datetime.now().isoformat(),
            "visualizations": visualizations,
            "summary": {
                "total_visualizations": len(visualizations),
                "visualization_types": list(set(v.get("type", "unknown") for v in visualizations))
            }
        }

        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CortexMD Visualization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2563eb; color: white; padding: 20px; border-radius: 10px; }}
                .viz-section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 10px; }}
                .stats {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 CortexMD Knowledge Graph Visualization Report</h1>
                <p>Generated: {report_data['generated_at']}</p>
            </div>

            <div class="stats">
                <h3>Report Summary</h3>
                <p>Total Visualizations: {report_data['summary']['total_visualizations']}</p>
                <p>Visualization Types: {', '.join(report_data['summary']['visualization_types'])}</p>
            </div>

            <h2>Visualizations</h2>
        """

        for viz in visualizations:
            html_content += f"""
            <div class="viz-section">
                <h3>{viz.get('title', 'Visualization')}</h3>
                <p><strong>Type:</strong> {viz.get('type', 'Unknown')}</p>
                <p><strong>Generated:</strong> {viz.get('generated_at', 'Unknown')}</p>
                <iframe src="{viz.get('filepath', '')}" width="100%" height="600px"></iframe>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        # Save report
        report_filename = f"visualization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_filepath = self.visualizations_dir / report_filename

        async with aiofiles.open(report_filepath, 'w') as f:
            await f.write(html_content)

        return str(report_filepath)
