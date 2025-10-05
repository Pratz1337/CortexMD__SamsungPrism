"""
Advanced Real-time CONCERN Early Warning System
Enhanced with deep pattern analysis, predictive modeling, and sophisticated alerting
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import random
from collections import deque

try:
    from ..core.database_manager import get_database
    from ..utils.enhanced_redis_service import get_redis_service
except ImportError:
    from core.database_manager import get_database
    from utils.enhanced_redis_service import get_redis_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class TrendDirection(Enum):
    RAPIDLY_INCREASING = "rapidly_increasing"
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"
    RAPIDLY_DECREASING = "rapidly_decreasing"

@dataclass
class VitalSigns:
    """Enhanced vital signs data structure"""
    heart_rate: float = 72.0
    blood_pressure_systolic: float = 120.0
    blood_pressure_diastolic: float = 80.0
    temperature: float = 98.6
    oxygen_saturation: float = 98.0
    respiratory_rate: float = 16.0
    timestamp: str = ""
    source: str = "simulated"
    confidence: float = 1.0

@dataclass
class ClinicalIndicators:
    """Clinical indicators for advanced analysis"""
    pain_score: float = 0.0
    consciousness_level: str = "alert"
    mobility_status: str = "independent"
    infection_markers: Dict[str, float] = None
    lab_abnormalities: List[str] = None
    medication_compliance: float = 1.0
    
    def __post_init__(self):
        if self.infection_markers is None:
            self.infection_markers = {}
        if self.lab_abnormalities is None:
            self.lab_abnormalities = []

@dataclass
class AdvancedRiskFactors:
    """Comprehensive risk factor analysis"""
    # Clinical Risk Factors
    vital_instability: bool = False
    deteriorating_vitals: bool = False
    sepsis_risk: bool = False
    cardiac_risk: bool = False
    respiratory_distress: bool = False
    neurological_changes: bool = False
    
    # Behavioral Risk Factors
    visit_frequency_high: bool = False
    off_hours_activity: bool = False
    prolonged_visits: bool = False
    multiple_notes: bool = False
    urgent_consultations: bool = False
    
    # Temporal Pattern Risk Factors
    weekend_activity: bool = False
    night_shift_activity: bool = False
    holiday_complications: bool = False
    rapid_status_changes: bool = False
    
    # Predictive Risk Factors
    early_deterioration_signs: bool = False
    pattern_anomalies: bool = False
    trending_decline: bool = False
    multi_system_involvement: bool = False
    
    # Confidence and Analysis Depth
    analysis_confidence: float = 1.0
    analysis_depth_score: float = 0.0
    data_quality_score: float = 1.0

@dataclass
class DepthAnalysisMetrics:
    """Metrics for analysis depth and confidence"""
    data_points_analyzed: int = 0
    temporal_coverage_hours: float = 0.0
    pattern_recognition_score: float = 0.0
    predictive_confidence: float = 0.0
    clinical_correlation_score: float = 0.0
    multi_modal_integration: float = 0.0
    analysis_completeness: float = 0.0

@dataclass
class ConcernAssessment:
    """Enhanced CONCERN assessment with depth analysis"""
    patient_id: str
    concern_score: float
    risk_level: str
    confidence_score: float
    
    # Risk factors and analysis
    risk_factors: List[str]
    advanced_risk_factors: AdvancedRiskFactors
    depth_metrics: DepthAnalysisMetrics
    
    # Temporal analysis
    trend_direction: str
    trend_velocity: float
    predicted_trajectory: str
    
    # Clinical data
    vital_signs: VitalSigns
    clinical_indicators: ClinicalIndicators
    
    # Activity metrics
    visits_24h: int = 0
    notes_24h: int = 0
    alerts_24h: int = 0
    
    # Metadata
    assessment_timestamp: str = ""
    next_assessment_due: str = ""
    alert_triggered: bool = False
    alert_severity: AlertSeverity = AlertSeverity.INFO
    
    # Analysis metadata
    analysis_duration_ms: float = 0.0
    data_sources: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = []
        if self.recommendations is None:
            self.recommendations = []

class AdvancedRealtimeConcernEWS:
    """Advanced Real-time CONCERN Early Warning System with deep analysis"""
    
    def __init__(self):
        try:
            self.db = get_database()
            self.redis = get_redis_service()
            logger.info("✅ Advanced Real-time CONCERN EWS initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Advanced CONCERN EWS: {e}")
            raise
        
        # Enhanced risk thresholds with depth consideration
        self.RISK_THRESHOLDS = {
            'critical': 0.85,
            'high': 0.65,
            'medium': 0.35,
            'low': 0.0
        }
        
        # Enhanced scoring weights
        self.SCORING_WEIGHTS = {
            # Clinical factors (higher weight)
            'vital_instability': 0.25,
            'deteriorating_vitals': 0.20,
            'sepsis_risk': 0.30,
            'cardiac_risk': 0.25,
            'respiratory_distress': 0.22,
            'neurological_changes': 0.28,
            
            # Activity patterns
            'visit_frequency': 0.12,
            'off_hours_activity': 0.10,
            'visit_duration': 0.08,
            'note_frequency': 0.10,
            'urgent_consultations': 0.15,
            
            # Temporal patterns
            'time_patterns': 0.08,
            'rapid_changes': 0.12,
            'pattern_anomalies': 0.15,
            
            # Predictive factors
            'early_warning_signs': 0.20,
            'trending_decline': 0.18,
            'multi_system_risk': 0.25
        }
        
        # Real-time data streams
        self.active_streams = {}
        self.assessment_cache = {}
        self.trend_history = {}
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Start real-time monitoring
        self.start_realtime_monitoring()
    
    def start_realtime_monitoring(self):
        """Start background real-time monitoring"""
        # Temporarily disable background monitoring to prevent loops
        logger.info("⏸️ Real-time monitoring disabled to prevent excessive processing")
        # if not self.monitoring_active:
        #     self.monitoring_active = True
        #     self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        #     self.monitoring_thread.start()
        #     logger.info("🔄 Real-time monitoring started")
    
    def stop_realtime_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("⏹️ Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop for continuous assessment"""
        while self.monitoring_active:
            try:
                # Get all patients for monitoring
                patients = self._get_monitored_patients()
                
                for patient_id in patients:
                    # Calculate real-time assessment
                    assessment = self.calculate_advanced_concern_score(patient_id)
                    
                    # Store in cache and history
                    self._cache_assessment(assessment)
                    self._update_trend_history(assessment)
                    
                    # Check for alerts
                    if assessment.alert_triggered:
                        self._handle_real_time_alert(assessment)

            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
    
    def calculate_advanced_concern_score(self, patient_id: str) -> ConcernAssessment:
        """Calculate advanced CONCERN score with deep analysis"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Starting advanced CONCERN analysis for {patient_id}")
            
            # 1. Gather comprehensive data
            patient_data = self._gather_comprehensive_data(patient_id)
            
            # 2. Generate real-time vital signs (in production, get from medical devices)
            vital_signs = self._get_realtime_vitals(patient_id)
            
            # 3. Analyze clinical indicators
            clinical_indicators = self._analyze_clinical_indicators(patient_data)
            
            # 4. Advanced risk factor analysis
            advanced_risk_factors = self._analyze_advanced_risk_factors(
                patient_data, vital_signs, clinical_indicators
            )
            
            # 5. Calculate base concern score
            base_score = self._calculate_enhanced_base_score(
                advanced_risk_factors, vital_signs, clinical_indicators
            )
            
            # 6. Apply temporal and predictive analysis
            temporal_score = self._analyze_temporal_patterns(patient_id, patient_data)
            predictive_score = self._calculate_predictive_risk(patient_id, patient_data)
            
            # 7. Calculate final score with confidence weighting
            confidence_score = self._calculate_analysis_confidence(patient_data, vital_signs)
            final_score = min(1.0, base_score + temporal_score + predictive_score)
            
            # Apply confidence adjustment
            final_score = final_score * confidence_score
            
            # 8. Determine risk level and trend
            risk_level = self._determine_risk_level(final_score)
            trend_direction, trend_velocity = self._analyze_trend_direction(patient_id, final_score)
            predicted_trajectory = self._predict_trajectory(patient_id, final_score, trend_velocity)
            
            # 9. Generate depth analysis metrics
            depth_metrics = self._calculate_depth_metrics(patient_data, vital_signs)
            
            # 10. Generate recommendations
            recommendations = self._generate_clinical_recommendations(
                advanced_risk_factors, vital_signs, risk_level
            )
            
            # 11. Determine alert status
            alert_triggered = final_score >= self.RISK_THRESHOLDS['high']
            alert_severity = self._determine_alert_severity(final_score, advanced_risk_factors)
            
            # 12. Create comprehensive assessment
            assessment = ConcernAssessment(
                patient_id=patient_id,
                concern_score=final_score,
                risk_level=risk_level,
                confidence_score=confidence_score,
                risk_factors=self._get_active_risk_factors(advanced_risk_factors),
                advanced_risk_factors=advanced_risk_factors,
                depth_metrics=depth_metrics,
                trend_direction=trend_direction.value,
                trend_velocity=trend_velocity,
                predicted_trajectory=predicted_trajectory,
                vital_signs=vital_signs,
                clinical_indicators=clinical_indicators,
                visits_24h=len(patient_data.get('visits', [])),
                notes_24h=len(patient_data.get('notes', [])),
                alerts_24h=len(patient_data.get('alerts', [])),
                assessment_timestamp=datetime.now().isoformat(),
                next_assessment_due=(datetime.now() + timedelta(minutes=5)).isoformat(),
                alert_triggered=alert_triggered,
                alert_severity=alert_severity,
                analysis_duration_ms=(time.time() - start_time) * 1000,
                data_sources=list(patient_data.keys()),
                recommendations=recommendations
            )
            
            logger.info(f"✅ Advanced CONCERN analysis complete: {patient_id} - {risk_level.upper()} ({final_score:.3f}) - Confidence: {confidence_score:.3f}")
            return assessment
            
        except Exception as e:
            logger.error(f"❌ Failed advanced CONCERN analysis for {patient_id}: {e}")
            return self._create_default_assessment(patient_id)
    
    def _gather_comprehensive_data(self, patient_id: str) -> Dict[str, Any]:
        """Gather comprehensive patient data from all sources"""
        data = {}
        
        try:
            # Get patient basic info
            patient = self.db.get_patient(patient_id)
            data['patient'] = patient or {}
            
            # Get recent visits (24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            data['visits'] = self._get_patient_visits_24h(patient_id, cutoff_time)
            
            # Get recent notes
            data['notes'] = self._get_patient_notes_24h(patient_id, cutoff_time)
            
            # Get diagnosis history
            data['diagnoses'] = self.db.get_patient_diagnosis_sessions(patient_id) or []
            
            # Get alerts history
            data['alerts'] = self._get_recent_alerts(patient_id, cutoff_time)
            
            # Get trend history from cache (ensure list for slicing)
            cached_trend = self.trend_history.get(patient_id, [])
            try:
                data['trend_history'] = list(cached_trend)
            except TypeError:
                data['trend_history'] = cached_trend if isinstance(cached_trend, list) else []
            
        except Exception as e:
            logger.error(f"Error gathering data for {patient_id}: {e}")
            
        return data
    
    def _get_realtime_vitals(self, patient_id: str) -> VitalSigns:
        """Generate realistic real-time vital signs based on patient status"""
        # In production, this would connect to medical devices
        # For now, generate realistic data based on patient context
        
        try:
            # Get previous assessment for continuity
            cached_data = self.redis.get_data(f"concern_current:{patient_id}")
            previous_vitals = None
            
            if cached_data and 'vital_signs' in cached_data:
                prev_data = cached_data['vital_signs']
                previous_vitals = VitalSigns(**prev_data) if isinstance(prev_data, dict) else prev_data
            
            # Generate realistic vitals with some variation
            base_hr = 72
            base_bp_sys = 120
            base_bp_dia = 80
            base_temp = 98.6
            base_o2 = 98
            base_rr = 16
            
            # Add realistic variation
            current_time = datetime.now()
            
            # Simulate some daily patterns and variations
            hour_factor = np.sin(current_time.hour * np.pi / 12) * 0.1
            random_factor = np.random.normal(0, 0.05)
            
            # Generate vitals with continuity if previous data exists
            if previous_vitals:
                # Smooth transition from previous values
                hr = previous_vitals.heart_rate + np.random.normal(0, 2)
                bp_sys = previous_vitals.blood_pressure_systolic + np.random.normal(0, 3)
                bp_dia = previous_vitals.blood_pressure_diastolic + np.random.normal(0, 2)
                temp = previous_vitals.temperature + np.random.normal(0, 0.2)
                o2_sat = previous_vitals.oxygen_saturation + np.random.normal(0, 0.5)
                rr = previous_vitals.respiratory_rate + np.random.normal(0, 1)
            else:
                # Generate new baseline
                hr = base_hr + hour_factor * 10 + random_factor * 15
                bp_sys = base_bp_sys + hour_factor * 15 + random_factor * 20
                bp_dia = base_bp_dia + hour_factor * 10 + random_factor * 15
                temp = base_temp + hour_factor * 0.5 + random_factor * 1.0
                o2_sat = base_o2 + random_factor * 2
                rr = base_rr + hour_factor * 2 + random_factor * 3
            
            # Ensure realistic ranges
            hr = max(45, min(150, hr))
            bp_sys = max(80, min(200, bp_sys))
            bp_dia = max(50, min(120, bp_dia))
            temp = max(95.0, min(106.0, temp))
            o2_sat = max(85, min(100, o2_sat))
            rr = max(8, min(30, rr))
            
            return VitalSigns(
                heart_rate=round(hr, 1),
                blood_pressure_systolic=round(bp_sys, 1),
                blood_pressure_diastolic=round(bp_dia, 1),
                temperature=round(temp, 1),
                oxygen_saturation=round(o2_sat, 1),
                respiratory_rate=round(rr, 1),
                timestamp=current_time.isoformat(),
                source="realtime_simulation",
                confidence=0.95
            )
            
        except Exception as e:
            logger.error(f"Error generating vitals for {patient_id}: {e}")
            return VitalSigns(timestamp=datetime.now().isoformat())
    
    def _analyze_clinical_indicators(self, patient_data: Dict[str, Any]) -> ClinicalIndicators:
        """Analyze clinical indicators from patient data"""
        
        # Simulate clinical indicators based on available data
        indicators = ClinicalIndicators()
        
        try:
            visits = patient_data.get('visits', [])
            notes = patient_data.get('notes', [])
            diagnoses = patient_data.get('diagnoses', [])
            
            # Analyze visit patterns for pain/distress indicators
            if len(visits) > 3:
                indicators.pain_score = min(8.0, len(visits) * 1.5)
            
            # Analyze notes for clinical concerns
            concerning_keywords = ['pain', 'distress', 'concern', 'unstable', 'deterioration']
            concern_count = 0
            
            for note in notes:
                note_text = str(note.get('content', '')).lower()
                for keyword in concerning_keywords:
                    if keyword in note_text:
                        concern_count += 1
            
            if concern_count > 2:
                indicators.consciousness_level = "confused"
                indicators.mobility_status = "assisted"
            
            # Analyze recent diagnoses for infection markers
            for diagnosis in diagnoses[-3:]:  # Last 3 diagnoses
                primary_diag = str(diagnosis.get('primary_diagnosis', '')).lower()
                if any(term in primary_diag for term in ['infection', 'fever', 'sepsis']):
                    indicators.infection_markers['elevated_wbc'] = 12000
                    indicators.infection_markers['elevated_crp'] = 50
                    indicators.lab_abnormalities.append('elevated_inflammatory_markers')
            
            # Calculate medication compliance based on visit frequency
            if len(visits) > 5:  # Too many visits might indicate non-compliance
                indicators.medication_compliance = max(0.3, 1.0 - (len(visits) - 3) * 0.1)
            
        except Exception as e:
            logger.error(f"Error analyzing clinical indicators: {e}")
        
        return indicators
    
    def _analyze_advanced_risk_factors(self, patient_data: Dict[str, Any], 
                                     vitals: VitalSigns, 
                                     clinical: ClinicalIndicators) -> AdvancedRiskFactors:
        """Advanced risk factor analysis with clinical depth"""
        
        factors = AdvancedRiskFactors()
        
        try:
            visits = patient_data.get('visits', [])
            notes = patient_data.get('notes', [])
            current_time = datetime.now()
            
            # 1. Vital sign analysis
            factors.vital_instability = (
                vitals.heart_rate > 100 or vitals.heart_rate < 60 or
                vitals.blood_pressure_systolic > 160 or vitals.blood_pressure_systolic < 90 or
                vitals.temperature > 100.4 or vitals.temperature < 96.0 or
                vitals.oxygen_saturation < 95
            )
            
            # 2. Clinical deterioration signs
            factors.sepsis_risk = (
                clinical.infection_markers.get('elevated_wbc', 0) > 12000 or
                vitals.temperature > 100.4 or vitals.temperature < 96.0 or
                vitals.heart_rate > 90
            )
            
            factors.cardiac_risk = (
                vitals.heart_rate > 100 or 
                vitals.blood_pressure_systolic > 160 or
                vitals.blood_pressure_systolic < 90
            )
            
            factors.respiratory_distress = (
                vitals.oxygen_saturation < 95 or
                vitals.respiratory_rate > 24 or vitals.respiratory_rate < 8
            )
            
            factors.neurological_changes = (
                clinical.consciousness_level != "alert" or
                clinical.pain_score > 6
            )
            
            # 3. Activity pattern analysis
            factors.visit_frequency_high = len(visits) > 4
            factors.multiple_notes = len(notes) > 3
            
            # 4. Temporal pattern analysis
            factors.off_hours_activity = any(
                datetime.fromisoformat(v['created_at'].replace('Z', '')).hour in range(22, 24) or
                datetime.fromisoformat(v['created_at'].replace('Z', '')).hour in range(0, 6)
                for v in visits
            )
            
            factors.weekend_activity = current_time.weekday() >= 5
            factors.night_shift_activity = current_time.hour >= 22 or current_time.hour <= 6
            
            # 5. Pattern anomaly detection
            factors.rapid_status_changes = len(visits) > 2 and len(notes) > 2
            factors.pattern_anomalies = factors.visit_frequency_high and factors.off_hours_activity
            
            # 6. Predictive indicators
            factors.early_deterioration_signs = (
                factors.vital_instability and 
                (factors.sepsis_risk or factors.cardiac_risk or factors.respiratory_distress)
            )
            
            factors.trending_decline = len(visits) > 3 and clinical.pain_score > 4
            
            factors.multi_system_involvement = sum([
                factors.sepsis_risk,
                factors.cardiac_risk,
                factors.respiratory_distress,
                factors.neurological_changes
            ]) > 1
            
            # 7. Calculate analysis confidence and depth
            data_points = len(visits) + len(notes) + (10 if vitals.confidence > 0.5 else 0)
            factors.analysis_confidence = min(1.0, data_points / 20.0)
            
            factors.analysis_depth_score = self._calculate_analysis_depth(
                patient_data, vitals, clinical
            )
            
            factors.data_quality_score = min(1.0, vitals.confidence * factors.analysis_confidence)
            
        except Exception as e:
            logger.error(f"Error in advanced risk factor analysis: {e}")
        
        return factors
    
    def _calculate_enhanced_base_score(self, risk_factors: AdvancedRiskFactors, 
                                     vitals: VitalSigns, 
                                     clinical: ClinicalIndicators) -> float:
        """Calculate enhanced base score with clinical depth"""
        
        score = 0.0
        
        try:
            # Clinical risk factors (higher weight)
            if risk_factors.sepsis_risk:
                score += self.SCORING_WEIGHTS['sepsis_risk']
            if risk_factors.cardiac_risk:
                score += self.SCORING_WEIGHTS['cardiac_risk']
            if risk_factors.respiratory_distress:
                score += self.SCORING_WEIGHTS['respiratory_distress']
            if risk_factors.neurological_changes:
                score += self.SCORING_WEIGHTS['neurological_changes']
            if risk_factors.vital_instability:
                score += self.SCORING_WEIGHTS['vital_instability']
            
            # Activity patterns
            if risk_factors.visit_frequency_high:
                score += self.SCORING_WEIGHTS['visit_frequency']
            if risk_factors.off_hours_activity:
                score += self.SCORING_WEIGHTS['off_hours_activity']
            if risk_factors.multiple_notes:
                score += self.SCORING_WEIGHTS['note_frequency']
            
            # Pattern anomalies
            if risk_factors.pattern_anomalies:
                score += self.SCORING_WEIGHTS['pattern_anomalies']
            if risk_factors.rapid_status_changes:
                score += self.SCORING_WEIGHTS['rapid_changes']
            
            # Predictive factors
            if risk_factors.early_deterioration_signs:
                score += self.SCORING_WEIGHTS['early_warning_signs']
            if risk_factors.trending_decline:
                score += self.SCORING_WEIGHTS['trending_decline']
            if risk_factors.multi_system_involvement:
                score += self.SCORING_WEIGHTS['multi_system_risk']
            
            # Apply confidence weighting
            score = score * risk_factors.data_quality_score
            
        except Exception as e:
            logger.error(f"Error calculating enhanced base score: {e}")
        
        return min(1.0, score)
    
    def _analyze_temporal_patterns(self, patient_id: str, patient_data: Dict[str, Any]) -> float:
        """Advanced temporal pattern analysis"""
        
        score = 0.0
        
        try:
            visits = patient_data.get('visits', [])
            notes = patient_data.get('notes', [])
            trend_history = patient_data.get('trend_history', [])
            # Ensure list for slicing
            try:
                trend_list = list(trend_history)
            except TypeError:
                trend_list = trend_history if isinstance(trend_history, list) else []
            
            # Recent activity clustering
            if len(visits) > 1 and len(notes) > 0:
                score += 0.1
            
            # Accelerating pattern detection
            if len(trend_list) >= 3:
                recent_scores = [t.get('score', 0) for t in trend_list[-3:]]
                if len(recent_scores) == 3:
                    trend = np.polyfit(range(3), recent_scores, 1)[0]
                    if trend > 0.05:  # Increasing trend
                        score += 0.15
            
            # Time-based risk factors
            current_time = datetime.now()
            if current_time.weekday() >= 5:  # Weekend
                score += 0.05
            if current_time.hour >= 22 or current_time.hour <= 6:  # Night
                score += 0.08
            
        except Exception as e:
            logger.error(f"Error in temporal pattern analysis: {e}")
        
        return min(0.3, score)
    
    def _calculate_predictive_risk(self, patient_id: str, patient_data: Dict[str, Any]) -> float:
        """Calculate predictive risk based on patterns and trends"""
        
        score = 0.0
        
        try:
            visits = patient_data.get('visits', [])
            diagnoses = patient_data.get('diagnoses', [])
            trend_history = patient_data.get('trend_history', [])
            # Ensure list for slicing
            try:
                trend_list = list(trend_history)
            except TypeError:
                trend_list = trend_history if isinstance(trend_history, list) else []
            
            # Predictive indicators based on visit patterns
            if len(visits) > 2:
                visit_times = [datetime.fromisoformat(v['created_at'].replace('Z', '')) 
                              for v in visits]
                visit_times.sort()
                
                # Calculate visit frequency acceleration
                if len(visit_times) >= 3:
                    intervals = [(visit_times[i+1] - visit_times[i]).total_seconds() 
                               for i in range(len(visit_times)-1)]
                    if len(intervals) >= 2:
                        trend = np.polyfit(range(len(intervals)), intervals, 1)[0]
                        if trend < -3600:  # Visits getting more frequent
                            score += 0.15
            
            # Diagnosis complexity and confidence trends
            if diagnoses:
                recent_diagnoses = diagnoses[-3:]
                low_confidence_count = sum(1 for d in recent_diagnoses 
                                         if d.get('confidence_score', 1.0) < 0.7)
                if low_confidence_count >= 2:
                    score += 0.12
            
            # Trend velocity analysis
            if len(trend_list) >= 4:
                scores = [t.get('score', 0) for t in trend_list[-4:]]
                velocity = np.mean(np.diff(scores))
                if velocity > 0.1:  # Rapid increase
                    score += 0.20
            
        except Exception as e:
            logger.error(f"Error in predictive risk calculation: {e}")
        
        return min(0.25, score)
    
    def _calculate_analysis_confidence(self, patient_data: Dict[str, Any], 
                                     vitals: VitalSigns) -> float:
        """Calculate confidence in the analysis based on data quality and completeness"""
        
        confidence = 0.0
        
        try:
            # Data availability score
            data_sources = len(patient_data.keys())
            data_availability = min(1.0, data_sources / 6.0)  # Expect 6 data sources
            
            # Data recency score
            visits = patient_data.get('visits', [])
            notes = patient_data.get('notes', [])
            
            recent_data_points = len(visits) + len(notes)
            data_recency = min(1.0, recent_data_points / 10.0)
            
            # Vital signs quality
            vitals_quality = vitals.confidence
            
            # Temporal coverage
            if visits:
                visit_times = [datetime.fromisoformat(v['created_at'].replace('Z', '')) 
                              for v in visits]
                if visit_times:
                    time_span = (max(visit_times) - min(visit_times)).total_seconds() / 3600
                    temporal_coverage = min(1.0, time_span / 24.0)  # 24-hour coverage
                else:
                    temporal_coverage = 0.1
            else:
                temporal_coverage = 0.1
            
            # Calculate weighted confidence
            confidence = (
                data_availability * 0.3 +
                data_recency * 0.25 +
                vitals_quality * 0.25 +
                temporal_coverage * 0.2
            )
            
        except Exception as e:
            logger.error(f"Error calculating analysis confidence: {e}")
            confidence = 0.5  # Default moderate confidence
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_depth_metrics(self, patient_data: Dict[str, Any], 
                               vitals: VitalSigns) -> DepthAnalysisMetrics:
        """Calculate metrics indicating the depth and quality of analysis"""
        
        try:
            visits = patient_data.get('visits', [])
            notes = patient_data.get('notes', [])
            diagnoses = patient_data.get('diagnoses', [])
            
            # Data points analyzed
            data_points = len(visits) + len(notes) + len(diagnoses) + 1  # +1 for vitals
            
            # Temporal coverage
            all_times = []
            for v in visits:
                all_times.append(datetime.fromisoformat(v['created_at'].replace('Z', '')))
            for n in notes:
                all_times.append(datetime.fromisoformat(n['created_at'].replace('Z', '')))
            
            if all_times:
                temporal_coverage = (max(all_times) - min(all_times)).total_seconds() / 3600
            else:
                temporal_coverage = 0
            
            # Pattern recognition score
            pattern_score = min(1.0, (len(visits) * 0.1 + len(notes) * 0.15) / 2.0)
            
            # Clinical correlation score
            clinical_data_points = len(diagnoses) + (1 if vitals.confidence > 0.8 else 0)
            clinical_correlation = min(1.0, clinical_data_points / 5.0)
            
            # Multi-modal integration
            modality_count = sum([
                len(visits) > 0,
                len(notes) > 0,
                len(diagnoses) > 0,
                vitals.confidence > 0.5
            ])
            multi_modal = modality_count / 4.0
            
            # Analysis completeness
            completeness = min(1.0, (data_points + temporal_coverage/24) / 10.0)
            
            return DepthAnalysisMetrics(
                data_points_analyzed=data_points,
                temporal_coverage_hours=round(temporal_coverage, 2),
                pattern_recognition_score=round(pattern_score, 3),
                predictive_confidence=min(1.0, data_points / 15.0),
                clinical_correlation_score=round(clinical_correlation, 3),
                multi_modal_integration=round(multi_modal, 3),
                analysis_completeness=round(completeness, 3)
            )
            
        except Exception as e:
            logger.error(f"Error calculating depth metrics: {e}")
            return DepthAnalysisMetrics()
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score with enhanced thresholds"""
        if score >= self.RISK_THRESHOLDS['critical']:
            return 'critical'
        elif score >= self.RISK_THRESHOLDS['high']:
            return 'high'
        elif score >= self.RISK_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_trend_direction(self, patient_id: str, current_score: float) -> Tuple[TrendDirection, float]:
        """Analyze trend direction and velocity"""
        
        try:
            # Get trend history from cache
            trend_history = self.trend_history.get(patient_id, [])
            # Ensure list for slicing
            try:
                trend_list = list(trend_history)
            except TypeError:
                trend_list = trend_history if isinstance(trend_history, list) else []
            
            if len(trend_list) < 2:
                return TrendDirection.STABLE, 0.0
            
            # Get recent scores
            recent_scores = [t['score'] for t in trend_list[-5:]]  # Last 5 assessments
            recent_scores.append(current_score)
            
            if len(recent_scores) < 3:
                return TrendDirection.STABLE, 0.0
            
            # Calculate trend using linear regression
            x = np.arange(len(recent_scores))
            trend_slope = np.polyfit(x, recent_scores, 1)[0]
            
            # Calculate velocity (rate of change)
            velocity = abs(trend_slope)
            
            # Determine trend direction
            if trend_slope > 0.05:
                if velocity > 0.15:
                    return TrendDirection.RAPIDLY_INCREASING, velocity
                else:
                    return TrendDirection.INCREASING, velocity
            elif trend_slope < -0.05:
                if velocity > 0.15:
                    return TrendDirection.RAPIDLY_DECREASING, velocity
                else:
                    return TrendDirection.DECREASING, velocity
            else:
                return TrendDirection.STABLE, velocity
                
        except Exception as e:
            logger.error(f"Error analyzing trend direction: {e}")
            return TrendDirection.STABLE, 0.0
    
    def _predict_trajectory(self, patient_id: str, current_score: float, velocity: float) -> str:
        """Predict future trajectory based on current trends"""
        
        try:
            if velocity > 0.2:
                return "rapid_change_expected"
            elif velocity > 0.1:
                if current_score > 0.6:
                    return "deterioration_likely"
                else:
                    return "improvement_possible"
            elif current_score > 0.8:
                return "critical_monitoring_required"
            elif current_score > 0.6:
                return "close_monitoring_recommended"
            else:
                return "stable_condition_expected"
                
        except Exception as e:
            logger.error(f"Error predicting trajectory: {e}")
            return "uncertain_trajectory"
    
    def _generate_clinical_recommendations(self, risk_factors: AdvancedRiskFactors, 
                                         vitals: VitalSigns, 
                                         risk_level: str) -> List[str]:
        """Generate clinical recommendations based on analysis"""
        
        recommendations = []
        
        try:
            if risk_level == 'critical':
                recommendations.append("🚨 IMMEDIATE medical evaluation required")
                recommendations.append("📋 Consider ICU consultation")
                recommendations.append("🩺 Continuous vital sign monitoring")
            
            if risk_factors.sepsis_risk:
                recommendations.append("🦠 Assess for sepsis - consider blood cultures")
                recommendations.append("🌡️ Monitor temperature closely")
            
            if risk_factors.cardiac_risk:
                recommendations.append("❤️ Cardiac monitoring recommended")
                if vitals.heart_rate > 100:
                    recommendations.append("⚡ Investigate tachycardia causes")
            
            if risk_factors.respiratory_distress:
                recommendations.append("🫁 Respiratory assessment needed")
                if vitals.oxygen_saturation < 95:
                    recommendations.append("💨 Consider oxygen therapy")
            
            if risk_factors.neurological_changes:
                recommendations.append("🧠 Neurological assessment indicated")
            
            if risk_factors.visit_frequency_high:
                recommendations.append("👥 Review care plan - frequent visits indicate concerns")
            
            if risk_factors.off_hours_activity:
                recommendations.append("🌙 Review after-hours care protocols")
            
            if risk_level in ['high', 'critical']:
                recommendations.append("📞 Notify attending physician")
                recommendations.append("📊 Review all recent lab results")
                recommendations.append("💊 Medication review recommended")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _determine_alert_severity(self, score: float, risk_factors: AdvancedRiskFactors) -> AlertSeverity:
        """Determine alert severity based on score and risk factors"""
        
        if score >= 0.9 or risk_factors.multi_system_involvement:
            return AlertSeverity.CRITICAL
        elif score >= 0.7 or risk_factors.early_deterioration_signs:
            return AlertSeverity.HIGH
        elif score >= 0.4:
            return AlertSeverity.MEDIUM
        elif score >= 0.2:
            return AlertSeverity.LOW
        else:
            return AlertSeverity.INFO
    
    def get_realtime_concern_stream(self, patient_id: str):
        """Get real-time concern data stream for a patient"""
        
        try:
            # Get current assessment
            assessment = self.calculate_advanced_concern_score(patient_id)
            
            # Apply post-diagnosis override if present (severity-aware final risk)
            try:
                override = self.redis.get_data(f"concern_override:{patient_id}")
                if override and isinstance(override, dict):
                    override_score = float(override.get('concern_score', assessment.concern_score))
                    override_level = str(override.get('risk_level', assessment.risk_level))
                    assessment.concern_score = override_score
                    assessment.risk_level = override_level
                    assessment.alert_triggered = (
                        override_level in ['high', 'critical'] or
                        override_score >= self.RISK_THRESHOLDS.get('high', 0.65)
                    )
                    assessment.alert_severity = self._determine_alert_severity(
                        override_score, assessment.advanced_risk_factors
                    )
            except Exception as ie:
                logger.warning(f"Override application failed for {patient_id}: {ie}")
            
            # Format for frontend
            stream_data = {
                'patient_id': patient_id,
                'timestamp': assessment.assessment_timestamp,
                'concern_score': assessment.concern_score,
                'risk_level': assessment.risk_level,
                'confidence_score': assessment.confidence_score,
                'trend_direction': assessment.trend_direction,
                'trend_velocity': assessment.trend_velocity,
                'predicted_trajectory': assessment.predicted_trajectory,
                'alert_triggered': assessment.alert_triggered,
                'alert_severity': assessment.alert_severity.value,
                'vital_signs': asdict(assessment.vital_signs),
                'depth_metrics': asdict(assessment.depth_metrics),
                'risk_factors': assessment.risk_factors,
                'recommendations': assessment.recommendations,
                'analysis_duration_ms': assessment.analysis_duration_ms
            }
            
            return stream_data
            
        except Exception as e:
            logger.error(f"Error getting realtime stream for {patient_id}: {e}")
            return None
    
    # Helper methods
    def _get_monitored_patients(self) -> List[str]:
        """Get list of patients being monitored"""
        try:
            patients = self.db.get_all_patients() or []
            return [p['patient_id'] for p in patients]
        except Exception as e:
            logger.error(f"Error getting monitored patients: {e}")
            return []
    
    def _get_patient_visits_24h(self, patient_id: str, cutoff_time: datetime) -> List[Dict]:
        """Get patient visits in the last 24 hours"""
        # Prefer real visits from DB; fallback to simulated
        try:
            visits = self.db.get_patient_visits(patient_id, limit=50) or []
            recent_visits = []
            for v in visits:
                ts_str = v.get('created_at') or v.get('timestamp')
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(str(ts_str).replace('Z', '').replace('+00:00', ''))
                except Exception:
                    continue
                if ts >= cutoff_time:
                    recent_visits.append({
                        'visit_id': v.get('visit_id') or f"VISIT_{patient_id}_{ts.strftime('%Y%m%d_%H%M')}",
                        'patient_id': patient_id,
                        'created_at': ts.isoformat(),
                        'visit_type': v.get('visit_type', 'routine'),
                        'duration_minutes': v.get('duration_minutes', 10),
                        'location': v.get('location', 'Ward')
                    })
            if recent_visits:
                return recent_visits
        except Exception as e:
            logger.error(f"Error fetching visits from DB for {patient_id}: {e}")
        
        # Fallback simulation
        current_time = datetime.now()
        visits = []
        if current_time.hour >= 22 or current_time.hour <= 6:  # Night shift
            visits.append({
                'visit_id': f"VISIT_{patient_id}_{current_time.strftime('%Y%m%d_%H%M')}",
                'patient_id': patient_id,
                'created_at': current_time.isoformat(),
                'visit_type': 'night_check',
                'duration_minutes': random.randint(10, 25),
                'location': 'Ward'
            })
        visit_count = random.randint(1, 4)
        for i in range(visit_count):
            visit_time = current_time - timedelta(hours=random.randint(1, 24))
            visits.append({
                'visit_id': f"VISIT_{patient_id}_{visit_time.strftime('%Y%m%d_%H%M')}_{i}",
                'patient_id': patient_id,
                'created_at': visit_time.isoformat(),
                'visit_type': random.choice(['routine', 'medication', 'assessment']),
                'duration_minutes': random.randint(5, 20),
                'location': 'Ward'
            })
        return visits
    
    def _get_patient_notes_24h(self, patient_id: str, cutoff_time: datetime) -> List[Dict]:
        """Get patient notes in the last 24 hours"""
        # Prefer real notes from DB; fallback to simulated
        try:
            notes = self.db.get_patient_clinical_notes(patient_id, limit=50) or []
            recent_notes = []
            for n in notes:
                ts_str = n.get('created_at') or n.get('timestamp')
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(str(ts_str).replace('Z', '').replace('+00:00', ''))
                except Exception:
                    continue
                if ts >= cutoff_time:
                    recent_notes.append({
                        'note_id': n.get('note_id') or f"NOTE_{patient_id}_{ts.strftime('%Y%m%d_%H%M')}",
                        'patient_id': patient_id,
                        'created_at': ts.isoformat(),
                        'content': n.get('content', ''),
                        'author': n.get('nurse_id', 'Nurse'),
                        'note_type': n.get('note_type', 'nursing')
                    })
            if recent_notes:
                return recent_notes
        except Exception as e:
            logger.error(f"Error fetching notes from DB for {patient_id}: {e}")
        
        # Fallback simulation
        current_time = datetime.now()
        notes = []
        note_count = random.randint(0, 3)
        for i in range(note_count):
            note_time = current_time - timedelta(hours=random.randint(1, 24))
            notes.append({
                'note_id': f"NOTE_{patient_id}_{note_time.strftime('%Y%m%d_%H%M')}_{i}",
                'patient_id': patient_id,
                'created_at': note_time.isoformat(),
                'content': f"Clinical note {i+1} for patient assessment",
                'author': 'Nurse',
                'note_type': random.choice(['assessment', 'medication', 'observation'])
            })
        return notes
    
    def _get_recent_alerts(self, patient_id: str, cutoff_time: datetime) -> List[Dict]:
        """Get recent alerts for patient"""
        # Mock implementation
        return []
    
    def _calculate_analysis_depth(self, patient_data: Dict[str, Any], 
                                vitals: VitalSigns, 
                                clinical: ClinicalIndicators) -> float:
        """Calculate analysis depth score"""
        
        depth_factors = [
            len(patient_data.get('visits', [])) > 2,
            len(patient_data.get('notes', [])) > 1,
            len(patient_data.get('diagnoses', [])) > 0,
            vitals.confidence > 0.8,
            len(clinical.infection_markers) > 0,
            clinical.pain_score > 0
        ]
        
        return sum(depth_factors) / len(depth_factors)
    
    def _get_active_risk_factors(self, risk_factors: AdvancedRiskFactors) -> List[str]:
        """Get list of active risk factors"""
        
        active_factors = []
        
        if risk_factors.vital_instability:
            active_factors.append("vital_signs_instability")
        if risk_factors.sepsis_risk:
            active_factors.append("sepsis_risk_indicators")
        if risk_factors.cardiac_risk:
            active_factors.append("cardiac_risk_factors")
        if risk_factors.respiratory_distress:
            active_factors.append("respiratory_distress")
        if risk_factors.neurological_changes:
            active_factors.append("neurological_changes")
        if risk_factors.visit_frequency_high:
            active_factors.append("high_visit_frequency")
        if risk_factors.off_hours_activity:
            active_factors.append("off_hours_activity")
        if risk_factors.early_deterioration_signs:
            active_factors.append("early_deterioration_signs")
        if risk_factors.pattern_anomalies:
            active_factors.append("pattern_anomalies")
        if risk_factors.multi_system_involvement:
            active_factors.append("multi_system_involvement")
        
        return active_factors
    
    def _cache_assessment(self, assessment: ConcernAssessment):
        """Cache assessment for quick retrieval"""
        try:
            cache_key = f"concern_advanced:{assessment.patient_id}"
            cache_data = asdict(assessment)
            # Cache full advanced assessment
            self.redis.store_data(cache_key, cache_data, ttl=300)  # 5-minute TTL
            # Also cache vital signs under concern_current for continuity
            try:
                self.redis.set_data(
                    f"concern_current:{assessment.patient_id}",
                    { 'vital_signs': asdict(assessment.vital_signs) },
                    expiry=600
                )
            except Exception as ie:
                logger.warning(f"Unable to cache concern_current for {assessment.patient_id}: {ie}")
        except Exception as e:
            logger.error(f"Error caching assessment: {e}")
    
    def _update_trend_history(self, assessment: ConcernAssessment):
        """Update trend history for patient"""
        try:
            patient_id = assessment.patient_id
            
            if patient_id not in self.trend_history:
                self.trend_history[patient_id] = deque(maxlen=50)  # Keep last 50 assessments
            
            self.trend_history[patient_id].append({
                'timestamp': assessment.assessment_timestamp,
                'score': assessment.concern_score,
                'level': assessment.risk_level,
                'confidence': assessment.confidence_score,
                'trend_velocity': assessment.trend_velocity
            })
            
            # Also store in Redis for persistence
            redis_key = f"concern_history:{patient_id}"
            history_data = {
                'scores': [t['score'] for t in self.trend_history[patient_id]],
                'timestamps': [t['timestamp'] for t in self.trend_history[patient_id]],
                'levels': [t['level'] for t in self.trend_history[patient_id]]
            }
            self.redis.store_data(redis_key, history_data, ttl=86400)  # 24-hour TTL
            
        except Exception as e:
            logger.error(f"Error updating trend history: {e}")
    
    def _handle_real_time_alert(self, assessment: ConcernAssessment):
        """Handle real-time alerts"""
        try:
            alert_data = {
                'patient_id': assessment.patient_id,
                'alert_type': 'concern_ews',
                'severity': assessment.alert_severity.value,
                'message': f"CONCERN score {assessment.concern_score:.3f} - {assessment.risk_level.upper()}",
                'recommendations': assessment.recommendations,
                'timestamp': assessment.assessment_timestamp,
                'confidence': assessment.confidence_score
            }
            
            # Store alert
            alert_key = f"alert:{assessment.patient_id}:{int(time.time())}"
            self.redis.store_data(alert_key, alert_data, ttl=86400)
            
            # Publish to real-time channels if needed
            logger.warning(f"🚨 ALERT: {alert_data['message']} for {assessment.patient_id}")
            
        except Exception as e:
            logger.error(f"Error handling real-time alert: {e}")
    
    def _create_default_assessment(self, patient_id: str) -> ConcernAssessment:
        """Create default assessment for errors"""
        return ConcernAssessment(
            patient_id=patient_id,
            concern_score=0.1,
            risk_level='low',
            confidence_score=0.5,
            risk_factors=[],
            advanced_risk_factors=AdvancedRiskFactors(),
            depth_metrics=DepthAnalysisMetrics(),
            trend_direction=TrendDirection.STABLE.value,
            trend_velocity=0.0,
            predicted_trajectory="stable_condition_expected",
            vital_signs=VitalSigns(timestamp=datetime.now().isoformat()),
            clinical_indicators=ClinicalIndicators(),
            assessment_timestamp=datetime.now().isoformat(),
            next_assessment_due=(datetime.now() + timedelta(minutes=10)).isoformat(),
            alert_triggered=False,
            alert_severity=AlertSeverity.INFO,
            recommendations=["Monitor patient status"],
            data_sources=[]
        )

# Global instance
_advanced_concern_ews = None

def get_advanced_realtime_concern_ews() -> AdvancedRealtimeConcernEWS:
    """Get the global advanced realtime CONCERN EWS instance"""
    global _advanced_concern_ews
    
    if _advanced_concern_ews is None:
        _advanced_concern_ews = AdvancedRealtimeConcernEWS()
    
    return _advanced_concern_ews

if __name__ == "__main__":
    # Test the system
    ews = AdvancedRealtimeConcernEWS()
    
    # Test assessment
    assessment = ews.calculate_advanced_concern_score("TEST_PATIENT_001")
    print(f"Assessment: {assessment.concern_score:.3f} - {assessment.risk_level}")
    print(f"Confidence: {assessment.confidence_score:.3f}")
    print(f"Risk factors: {assessment.risk_factors}")
    print(f"Recommendations: {assessment.recommendations}")
