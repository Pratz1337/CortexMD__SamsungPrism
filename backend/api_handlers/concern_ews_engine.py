"""
CONCERN Early Warning System (EWS) Engine
AI-powered analysis of nursing notes metadata to predict patient deterioration
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
import uuid
import logging

# Import the simple database
from simple_database import get_database

logger = logging.getLogger(__name__)

@dataclass
class ClinicalNote:
    note_id: str
    patient_id: str
    nurse_id: str
    content: str
    timestamp: datetime
    location: str
    shift: str
    note_type: str = "nursing"
    
@dataclass
class PatientVisit:
    visit_id: str
    patient_id: str
    nurse_id: str
    timestamp: datetime
    location: str
    visit_type: str = "routine"
    duration_minutes: int = 5
    
@dataclass
class ConcernAlert:
    patient_id: str
    concern_score: float
    risk_level: str
    factors: List[str]
    timestamp: datetime
    metadata_patterns: Dict[str, Any]

class ConcernEWSEngine:
    """CONCERN Early Warning System - AI engine for patient deterioration prediction"""
    
    def __init__(self):
        self.db = get_database()
        
        # Risk thresholds
        self.RISK_THRESHOLDS = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }
        
        # Pattern weights for metadata analysis
        self.PATTERN_WEIGHTS = {
            'visit_frequency_increase': 0.25,
            'off_hours_visits': 0.20,
            'visit_duration_increase': 0.15,
            'medication_delays': 0.20,
            'urgent_visit_type': 0.20
        }
    
    def init_database(self):
        """Initialize SQLite database for CONCERN EWS"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clinical notes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clinical_notes (
                note_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                nurse_id TEXT,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                location TEXT,
                shift TEXT,
                note_type TEXT DEFAULT 'nursing',
                metadata TEXT DEFAULT '{}'
            )
        ''')
        
        # Patient visits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_visits (
                visit_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                nurse_id TEXT,
                timestamp TEXT NOT NULL,
                location TEXT NOT NULL,
                visit_type TEXT DEFAULT 'routine',
                duration_minutes INTEGER DEFAULT 5
            )
        ''')
        
        # Concern scores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS concern_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                concern_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                factors TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata_patterns TEXT NOT NULL
            )
        ''')
        
        # Patient monitoring table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_monitoring (
                patient_id TEXT PRIMARY KEY,
                admission_date TEXT NOT NULL,
                current_status TEXT DEFAULT 'stable',
                latest_concern_score REAL,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_clinical_note(self, patient_id: str, nurse_id: str, content: str, 
                         location: str = "Ward", shift: str = "Day") -> str:
        """Add a clinical note and trigger CONCERN analysis"""
        note_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO clinical_notes 
            (note_id, patient_id, nurse_id, content, timestamp, location, shift)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (note_id, patient_id, nurse_id, content, timestamp, location, shift))
        
        conn.commit()
        conn.close()
        
        # Trigger CONCERN analysis
        self.analyze_patient_concern(patient_id)
        
        return note_id
    
    def add_patient_visit(self, patient_id: str, nurse_id: str, location: str,
                         visit_type: str = "routine", duration_minutes: int = 5) -> str:
        """Add a patient visit record"""
        visit_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patient_visits 
            (visit_id, patient_id, nurse_id, timestamp, location, visit_type, duration_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (visit_id, patient_id, nurse_id, timestamp, location, visit_type, duration_minutes))
        
        conn.commit()
        conn.close()
        
        # Trigger CONCERN analysis
        self.analyze_patient_concern(patient_id)
        
        return visit_id
    
    def analyze_patient_concern(self, patient_id: str) -> ConcernAlert:
        """Main CONCERN analysis - analyze metadata patterns for early warning"""
        
        # Get recent data (last 24 hours)
        cutoff_time = (datetime.now() - timedelta(hours=24)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent visits
        cursor.execute('''
            SELECT * FROM patient_visits 
            WHERE patient_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
        ''', (patient_id, cutoff_time))
        recent_visits = cursor.fetchall()
        
        # Get recent notes
        cursor.execute('''
            SELECT * FROM clinical_notes 
            WHERE patient_id = ? AND timestamp > ?
            ORDER BY timestamp DESC
        ''', (patient_id, cutoff_time))
        recent_notes = cursor.fetchall()
        
        conn.close()
        
        # Analyze metadata patterns
        concern_score, risk_factors, metadata_patterns = self._analyze_metadata_patterns(
            recent_visits, recent_notes
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(concern_score)
        
        # Save concern score
        self._save_concern_score(patient_id, concern_score, risk_level, risk_factors, metadata_patterns)
        
        # Create alert
        alert = ConcernAlert(
            patient_id=patient_id,
            concern_score=concern_score,
            risk_level=risk_level,
            factors=risk_factors,
            timestamp=datetime.now(),
            metadata_patterns=metadata_patterns
        )
        
        return alert
    
    def _analyze_metadata_patterns(self, visits: List, notes: List) -> Tuple[float, List[str], Dict[str, Any]]:
        """Analyze metadata patterns from visits and notes"""
        
        concern_score = 0.0
        risk_factors = []
        metadata_patterns = {}
        
        if not visits and not notes:
            return 0.0, [], {}
        
        # Analyze visit frequency patterns
        if visits:
            visit_times = [datetime.fromisoformat(v[3]) for v in visits]  # timestamp is index 3
            visit_frequency_score, freq_factors = self._analyze_visit_frequency(visit_times)
            concern_score += visit_frequency_score * self.PATTERN_WEIGHTS['visit_frequency_increase']
            risk_factors.extend(freq_factors)
            metadata_patterns['visit_frequency'] = len(visits)
        
        # Analyze off-hours visits
        if visits:
            off_hours_score, off_hours_factors = self._analyze_off_hours_activity(visits)
            concern_score += off_hours_score * self.PATTERN_WEIGHTS['off_hours_visits']
            risk_factors.extend(off_hours_factors)
            metadata_patterns['off_hours_visits'] = off_hours_score > 0.5
        
        # Analyze visit duration patterns
        if visits:
            duration_score, duration_factors = self._analyze_visit_duration(visits)
            concern_score += duration_score * self.PATTERN_WEIGHTS['visit_duration_increase']
            risk_factors.extend(duration_factors)
            metadata_patterns['avg_visit_duration'] = np.mean([v[6] for v in visits])  # duration is index 6
        
        # Analyze note content for medication delays/concerns
        if notes:
            med_score, med_factors = self._analyze_medication_patterns(notes)
            concern_score += med_score * self.PATTERN_WEIGHTS['medication_delays']
            risk_factors.extend(med_factors)
            metadata_patterns['medication_concerns'] = med_score > 0.3
        
        # Analyze urgent visit types
        if visits:
            urgent_visits = [v for v in visits if v[5] in ['urgent', 'emergency']]  # visit_type is index 5
            if urgent_visits:
                urgent_score = min(len(urgent_visits) * 0.3, 1.0)
                concern_score += urgent_score * self.PATTERN_WEIGHTS['urgent_visit_type']
                risk_factors.append(f"Urgent visits detected: {len(urgent_visits)}")
                metadata_patterns['urgent_visits'] = len(urgent_visits)
        
        # Cap concern score at 1.0
        concern_score = min(concern_score, 1.0)
        
        return concern_score, risk_factors, metadata_patterns
    
    def _analyze_visit_frequency(self, visit_times: List[datetime]) -> Tuple[float, List[str]]:
        """Analyze if visit frequency is increasing (concerning pattern)"""
        if len(visit_times) < 3:
            return 0.0, []
        
        # Calculate visits per hour
        time_span_hours = (max(visit_times) - min(visit_times)).total_seconds() / 3600
        if time_span_hours == 0:
            return 0.0, []
        
        visits_per_hour = len(visit_times) / time_span_hours
        
        # Normal frequency is about 1 visit per 4-8 hours
        if visits_per_hour > 0.5:  # More than 1 visit per 2 hours
            score = min(visits_per_hour / 2.0, 1.0)
            return score, [f"High visit frequency: {visits_per_hour:.1f} visits/hour"]
        
        return 0.0, []
    
    def _analyze_off_hours_activity(self, visits: List) -> Tuple[float, List[str]]:
        """Analyze visits during off-hours (nights/weekends)"""
        off_hours_count = 0
        
        for visit in visits:
            timestamp = datetime.fromisoformat(visit[3])
            hour = timestamp.hour
            
            # Define off-hours: 10 PM to 6 AM
            if hour >= 22 or hour <= 6:
                off_hours_count += 1
        
        if off_hours_count > 0:
            score = min(off_hours_count * 0.4, 1.0)
            return score, [f"Off-hours visits: {off_hours_count}"]
        
        return 0.0, []
    
    def _analyze_visit_duration(self, visits: List) -> Tuple[float, List[str]]:
        """Analyze if visit durations are increasing"""
        if len(visits) < 2:
            return 0.0, []
        
        durations = [v[6] for v in visits]  # duration_minutes is index 6
        avg_duration = np.mean(durations)
        
        # Normal visit duration is 5-10 minutes
        if avg_duration > 15:
            score = min((avg_duration - 10) / 20.0, 1.0)
            return score, [f"Extended visit durations: avg {avg_duration:.1f} minutes"]
        
        return 0.0, []
    
    def _analyze_medication_patterns(self, notes: List) -> Tuple[float, List[str]]:
        """Analyze notes for medication-related concerns"""
        med_keywords = [
            'hold medication', 'delayed medication', 'refused medication',
            'medication concern', 'side effect', 'adverse reaction',
            'bp too low', 'bp too high', 'heart rate', 'oxygen sat'
        ]
        
        concern_count = 0
        factors = []
        
        for note in notes:
            content = note[3].lower()  # content is index 3
            for keyword in med_keywords:
                if keyword in content:
                    concern_count += 1
                    factors.append(f"Medication concern: {keyword}")
                    break
        
        if concern_count > 0:
            score = min(concern_count * 0.3, 1.0)
            return score, factors
        
        return 0.0, []
    
    def _determine_risk_level(self, concern_score: float) -> str:
        """Determine risk level based on concern score"""
        if concern_score >= self.RISK_THRESHOLDS['critical']:
            return 'critical'
        elif concern_score >= self.RISK_THRESHOLDS['high']:
            return 'high'
        elif concern_score >= self.RISK_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _save_concern_score(self, patient_id: str, score: float, risk_level: str, 
                           factors: List[str], metadata_patterns: Dict[str, Any]):
        """Save concern score to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO concern_scores 
            (patient_id, concern_score, risk_level, factors, timestamp, metadata_patterns)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (patient_id, score, risk_level, json.dumps(factors), timestamp, json.dumps(metadata_patterns)))
        
        # Update patient monitoring
        cursor.execute('''
            INSERT OR REPLACE INTO patient_monitoring 
            (patient_id, admission_date, latest_concern_score, last_updated)
            VALUES (?, ?, ?, ?)
        ''', (patient_id, timestamp, score, timestamp))
        
        conn.commit()
        conn.close()
    
    def get_patient_dashboard(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive dashboard data for a patient"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get latest concern score
        cursor.execute('''
            SELECT * FROM concern_scores 
            WHERE patient_id = ? 
            ORDER BY timestamp DESC LIMIT 1
        ''', (patient_id,))
        latest_score = cursor.fetchone()
        
        # Get recent scores for trend
        cursor.execute('''
            SELECT concern_score, risk_level, timestamp FROM concern_scores 
            WHERE patient_id = ? 
            ORDER BY timestamp DESC LIMIT 10
        ''', (patient_id,))
        score_history = cursor.fetchall()
        
        # Get recent visits
        cursor.execute('''
            SELECT COUNT(*) FROM patient_visits 
            WHERE patient_id = ? AND timestamp > ?
        ''', (patient_id, (datetime.now() - timedelta(hours=24)).isoformat()))
        visits_24h = cursor.fetchone()[0]
        
        # Get recent notes
        cursor.execute('''
            SELECT COUNT(*) FROM clinical_notes 
            WHERE patient_id = ? AND timestamp > ?
        ''', (patient_id, (datetime.now() - timedelta(hours=24)).isoformat()))
        notes_24h = cursor.fetchone()[0]
        
        conn.close()
        
        dashboard = {
            'patient_id': patient_id,
            'current_concern_score': latest_score[2] if latest_score else 0.0,
            'current_risk_level': latest_score[3] if latest_score else 'low',
            'risk_factors': json.loads(latest_score[4]) if latest_score else [],
            'metadata_patterns': json.loads(latest_score[6]) if latest_score else {},
            'score_trend': [{'score': s[0], 'level': s[1], 'timestamp': s[2]} for s in score_history],
            'visits_24h': visits_24h,
            'notes_24h': notes_24h,
            'last_updated': latest_score[5] if latest_score else None
        }
        
        return dashboard
    
    def get_all_patients_status(self) -> List[Dict[str, Any]]:
        """Get status overview for all monitored patients"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT patient_id FROM patient_monitoring
        ''')
        patient_ids = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        patients_status = []
        for patient_id in patient_ids:
            dashboard = self.get_patient_dashboard(patient_id)
            patients_status.append({
                'patient_id': patient_id,
                'concern_score': dashboard['current_concern_score'],
                'risk_level': dashboard['current_risk_level'],
                'last_updated': dashboard['last_updated']
            })
        
        # Sort by concern score (highest first)
        patients_status.sort(key=lambda x: x['concern_score'], reverse=True)
        
        return patients_status

# Global instance
concern_engine = ConcernEWSEngine()
