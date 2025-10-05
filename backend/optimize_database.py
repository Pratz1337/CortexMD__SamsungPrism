#!/usr/bin/env python3
"""
Database Index Optimization for Ultra-Fast Patient Data Retrieval
Creates optimized indexes for instant patient data loading
"""

import logging
import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from core.database_manager import get_database
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

def optimize_database_indexes():
    """Create optimized indexes for ultra-fast patient queries"""
    try:
        db = get_database()
        
        # Get the database connection
        if hasattr(db, 'engine'):
            engine = db.engine
        else:
            print("❌ Database engine not available")
            return False
        
        print("🔧 Optimizing database indexes for ultra-fast patient loading...")
        
        # List of indexes to create for optimal performance
        indexes = [
            # Patient table optimizations
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patients_id_status ON patients(patient_id, current_status)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patients_created_at ON patients(created_at DESC)",
            
            # Diagnosis sessions optimizations
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_diagnosis_sessions_patient_created ON diagnosis_sessions(patient_id, created_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_diagnosis_sessions_status_patient ON diagnosis_sessions(status, patient_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_diagnosis_sessions_confidence ON diagnosis_sessions(patient_id, confidence_score DESC)",
            
            # Concern scores optimizations
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_concern_scores_patient_created ON concern_scores(patient_id, created_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_concern_scores_risk_level ON concern_scores(patient_id, risk_level)",
            
            # Concern severity tracking optimizations
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_concern_severity_patient_risk ON concern_severity_tracking(patient_id, current_risk_level)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_concern_severity_risk_score ON concern_severity_tracking(current_risk_score DESC)",
            
            # Chat messages optimizations
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chat_messages_patient_timestamp ON chat_messages(patient_id, timestamp DESC)",
            
            # Clinical notes optimizations
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_clinical_notes_patient_timestamp ON clinical_notes(patient_id, timestamp DESC)",
            
            # Patient visits optimizations
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_patient_visits_patient_timestamp ON patient_visits(patient_id, timestamp DESC)"
        ]
        
        successful_indexes = 0
        failed_indexes = 0
        
        with engine.connect() as conn:
            # Start a transaction
            trans = conn.begin()
            
            try:
                for index_sql in indexes:
                    try:
                        print(f"Creating index: {index_sql.split('idx_')[1].split(' ')[0] if 'idx_' in index_sql else 'unknown'}")
                        conn.execute(text(index_sql))
                        successful_indexes += 1
                        print("  ✅ Success")
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            print("  ✅ Already exists")
                            successful_indexes += 1
                        else:
                            print(f"  ❌ Failed: {e}")
                            failed_indexes += 1
                
                # Commit the transaction
                trans.commit()
                print(f"\n🎯 Index optimization complete:")
                print(f"   ✅ Successful: {successful_indexes}")
                print(f"   ❌ Failed: {failed_indexes}")
                
                # Update table statistics for better query planning
                print("\n📊 Updating table statistics...")
                
                tables = ['patients', 'diagnosis_sessions', 'concern_scores', 
                         'concern_severity_tracking', 'chat_messages', 
                         'clinical_notes', 'patient_visits']
                
                for table in tables:
                    try:
                        conn.execute(text(f"ANALYZE {table}"))
                        print(f"  ✅ {table}")
                    except Exception as e:
                        print(f"  ⚠️ {table}: {e}")
                
                print("\n🚀 Database optimization complete! Patient loading should now be ultra-fast.")
                return successful_indexes > failed_indexes
                
            except Exception as e:
                trans.rollback()
                print(f"❌ Transaction failed: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Database optimization failed: {e}")
        return False

def check_database_performance():
    """Check current database performance metrics"""
    try:
        db = get_database()
        
        if hasattr(db, 'engine'):
            engine = db.engine
        else:
            print("❌ Database engine not available")
            return
        
        print("📈 Checking database performance...")
        
        with engine.connect() as conn:
            # Check index usage
            result = conn.execute(text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes 
                WHERE schemaname = 'public'
                ORDER BY idx_tup_read DESC
                LIMIT 10
            """))
            
            print("\n📊 Top 10 most used indexes:")
            for row in result:
                print(f"  {row.tablename}.{row.indexname}: {row.idx_tup_read:,} reads")
            
            # Check table sizes
            result = conn.execute(text("""
                SELECT 
                    tablename,
                    pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes
                FROM pg_stat_user_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(tablename::regclass) DESC
            """))
            
            print("\n💾 Table sizes and activity:")
            for row in result:
                print(f"  {row.tablename}: {row.size} ({row.inserts:,} ins, {row.updates:,} upd, {row.deletes:,} del)")
            
    except Exception as e:
        print(f"❌ Performance check failed: {e}")

if __name__ == "__main__":
    print("🚀 CortexMD Database Performance Optimization")
    print("=" * 50)
    
    # Check performance first
    check_database_performance()
    
    print("\n" + "=" * 50)
    
    # Optimize indexes
    success = optimize_database_indexes()
    
    if success:
        print("\n🎉 Database optimization successful!")
        print("   Patient data loading should now be instant!")
        
        # Check performance after optimization
        print("\n" + "=" * 50)
        check_database_performance()
    else:
        print("\n😞 Database optimization failed!")
        print("   Please check the error messages above.")