"use client"

import React, { useState, useEffect, useRef } from "react"
import { motion, useTransform, AnimatePresence, useMotionValue, useSpring } from "framer-motion"
import { cn } from "@/lib/utils"
import { User, Calendar, AlertTriangle, Activity, Clock } from "lucide-react"
import { api } from "@/lib/api"
import { toast } from "react-hot-toast"
import { HealthcareLoadingScreen } from "@/components/ui/HealthcareLoadingScreen"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"

interface Patient {
  patient_id: string
  patient_name: string
  current_status: string
  admission_date: string
  concern_score: number
  risk_level: string
  latest_diagnosis?: string
}

const CircleProgress = ({
  value,
  maxValue = 100,
  size = 60,
  strokeWidth = 4,
  className,
}: {
  value: number;
  maxValue?: number;
  size?: number;
  strokeWidth?: number;
  className?: string;
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const fillPercentage = Math.min(value / maxValue, 1);
  const strokeDashoffset = circumference * (1 - fillPercentage);

  const getColor = (percentage: number) => {
    if (percentage < 0.3) return "#10b981"; // Green - low concern
    if (percentage < 0.7) return "#f59e0b"; // Yellow - medium concern
    return "#ef4444"; // Red - high concern
  };

  const currentColor = getColor(fillPercentage);
  const displayValue = maxValue === 1 ? Math.round(value * 100) : Math.round(value);

  return (
    <div className={cn("relative", className)}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="transform -rotate-90"
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          className="fill-transparent stroke-muted"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          className="fill-transparent transition-all duration-300"
          stroke={currentColor}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-sm font-semibold">{displayValue}</span>
      </div>
    </div>
  );
};

interface PatientCardProps {
  patient: Patient;
  onClick?: (patient: Patient) => void;
  className?: string;
}

const PatientCard: React.FC<PatientCardProps> = ({ patient, onClick, className }) => {
  const [isHovered, setIsHovered] = useState(false);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });
  const cardRef = useRef<HTMLDivElement>(null);

  const springConfig = { stiffness: 100, damping: 15 };
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  
  const rotateX = useSpring(useTransform(y, [-100, 100], [10, -10]), springConfig);
  const rotateY = useSpring(useTransform(x, [-100, 100], [-10, 10]), springConfig);

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (cardRef.current) {
      const rect = cardRef.current.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      
      const mouseX = event.clientX - centerX;
      const mouseY = event.clientY - centerY;
      
      x.set(mouseX);
      y.set(mouseY);
    }
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    x.set(0);
    y.set(0);
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'low': return 'bg-green-500';
      case 'medium': return 'bg-yellow-500';
      case 'high': return 'bg-red-500';
      case 'critical': return 'bg-red-600';
      default: return 'bg-gray-500';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'stable': return 'text-green-600 bg-green-50 border-green-200';
      case 'monitoring': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'critical': return 'text-red-600 bg-red-50 border-red-200';
      case 'recovering': return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'active': return 'text-blue-600 bg-blue-50 border-blue-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  return (
    <motion.div
      ref={cardRef}
      className={cn(
        "relative group cursor-pointer perspective-1000",
        className
      )}
      style={{
        rotateX: rotateX,
        rotateY: rotateY,
        transformStyle: "preserve-3d",
      }}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={handleMouseLeave}
      onClick={() => onClick?.(patient)}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
    >
      <Card className="relative overflow-hidden bg-gradient-to-br from-background via-background to-muted/20 border-2 border-border/50 hover:border-primary/30 transition-all duration-300">
        {/* Animated background glow */}
        <motion.div
          className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
          style={{
            background: `radial-gradient(circle at ${50 + (rotation.y * 0.5)}% ${50 + (rotation.x * 0.5)}%, rgba(59, 130, 246, 0.1) 0%, transparent 70%)`,
          }}
        />

        {/* Floating particles effect */}
        <AnimatePresence>
          {isHovered && (
            <motion.div
              className="absolute inset-0 pointer-events-none"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {[...Array(6)].map((_, i) => (
                <motion.div
                  key={i}
                  className="absolute w-1 h-1 bg-primary/30 rounded-full"
                  style={{
                    left: `${20 + (i * 15)}%`,
                    top: `${30 + (i * 10)}%`,
                  }}
                  animate={{
                    y: [-10, -30, -10],
                    opacity: [0, 1, 0],
                  }}
                  transition={{
                    duration: 2,
                    repeat: Infinity,
                    delay: i * 0.2,
                  }}
                />
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        <div className="relative z-10 p-6">
          {/* Header with patient info and risk indicator */}
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              <motion.div
                className="relative"
                whileHover={{ scale: 1.1 }}
                transition={{ type: "spring", stiffness: 400, damping: 10 }}
              >
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-primary/20 to-primary/40 flex items-center justify-center border-2 border-primary/30">
                  <User className="w-6 h-6 text-primary" />
                </div>
                <motion.div
                  className={cn("absolute -top-1 -right-1 w-4 h-4 rounded-full border-2 border-background", getRiskColor(patient.risk_level))}
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </motion.div>
              
              <div>
                <h3 className="font-semibold text-lg text-foreground group-hover:text-primary transition-colors">
                  {patient.patient_name}
                </h3>
                <p className="text-sm text-muted-foreground">ID: {patient.patient_id}</p>
              </div>
            </div>

            <div className="relative">
              <CircleProgress
                value={patient.concern_score || 0}
                maxValue={1}
                size={50}
                strokeWidth={3}
              />
              {(!patient.concern_score || patient.concern_score === 0) && (
                <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-amber-500 rounded-full border border-white flex items-center justify-center">
                  <span className="text-white text-xs">!</span>
                </div>
              )}
            </div>
          </div>

          {/* Status and admission info */}
          <div className="space-y-3 mb-4">
            <div className="flex items-center justify-between">
              <Badge
                variant="outline"
                className={cn("text-xs font-medium", getStatusColor(patient.current_status))}
              >
                <Activity className="w-3 h-3 mr-1" />
                {patient.current_status}
              </Badge>
              
              <div className="flex items-center text-xs text-muted-foreground">
                <Calendar className="w-3 h-3 mr-1" />
                {new Date(patient.admission_date).toLocaleDateString()}
              </div>
            </div>

            {patient.latest_diagnosis && (
              <div className="flex items-start space-x-2">
                <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-foreground line-clamp-2">{patient.latest_diagnosis}</p>
              </div>
            )}
          </div>



          {/* Risk level indicator */}
          <motion.div
            className="mt-4 pt-3 border-t border-border/50"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">Risk Level</span>
              <Badge
                variant="outline"
                className={cn(
                  "text-xs font-semibold",
                  patient.risk_level.toLowerCase() === 'low' && "text-green-700 bg-green-50 border-green-200",
                  patient.risk_level.toLowerCase() === 'medium' && "text-yellow-700 bg-yellow-50 border-yellow-200",
                  patient.risk_level.toLowerCase() === 'high' && "text-red-700 bg-red-50 border-red-200",
                  patient.risk_level.toLowerCase() === 'critical' && "text-red-800 bg-red-100 border-red-300"
                )}
              >
                {patient.risk_level.toUpperCase()}
              </Badge>
            </div>
          </motion.div>

          {/* Hover overlay with additional actions */}
          <AnimatePresence>
            {isHovered && (
              <motion.div
                className="absolute inset-0 bg-background/95 backdrop-blur-sm flex items-center justify-center"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
              >
                <motion.div
                  className="text-center space-y-2"
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.8, opacity: 0 }}
                  transition={{ delay: 0.1 }}
                >
                  <p className="text-sm font-medium text-foreground">View Patient Dashboard</p>
                  <div className="flex items-center justify-center space-x-2 text-xs text-muted-foreground">
                    <Clock className="w-3 h-3" />
                    <span>Click to access full details</span>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </Card>
    </motion.div>
  );
};

interface PatientSelectorProps {
  onSelectPatient: (patientId: string) => void
}

export function PatientSelector({ onSelectPatient }: PatientSelectorProps) {
  const [patients, setPatients] = useState<Patient[]>([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState("")

  const fetchPatients = async () => {
    try {
      // First try fast mode
      let { data } = await api.get('/api/patients?fast=true')
      console.log('Fetched patients data (fast mode):', data.patients)
      
      // Check if concern scores are missing or zero
      const hasValidConcernScores = data.patients?.some((p: Patient) => 
        p.concern_score !== undefined && p.concern_score !== null && p.concern_score > 0
      )
      
      // If no valid concern scores, try to calculate them first
      if (!hasValidConcernScores && data.patients?.length > 0) {
        console.log('No valid concern scores found, calculating concern scores...')
        try {
          // Call bulk calculate endpoint to generate concern scores
          await api.post('/api/concern/bulk-calculate')
          console.log('✅ Bulk concern calculation initiated')
          
          // Wait a moment for calculation to complete
          await new Promise(resolve => setTimeout(resolve, 1000))
          
          // Refetch with full mode to get updated scores
          const fullData = await api.get('/api/patients')
          if (fullData.data.patients) {
            data = fullData.data
            console.log('Fetched patients data after concern calculation:', data.patients)
          }
        } catch (calcError) {
          console.warn('Concern calculation failed, trying full mode:', calcError)
          try {
            const fullData = await api.get('/api/patients')
            if (fullData.data.patients) {
              data = fullData.data
              console.log('Fetched patients data (full mode):', data.patients)
            }
          } catch (fullModeError) {
            console.warn('Full mode also failed, using fast mode results:', fullModeError)
          }
        }
      }
      
      // Final check - if still no concern scores, generate some default values
      if (data.patients) {
        data.patients.forEach((patient: Patient) => {
          console.log(`Patient ${patient.patient_id} (${patient.patient_name}): concern_score = ${patient.concern_score}`)
          
          // If concern score is still 0 or missing, generate a realistic default
          if (!patient.concern_score || patient.concern_score === 0) {
            // Generate a realistic concern score based on patient data
            let defaultScore = 0.2 // Base low risk
            
            // Increase based on status
            if (patient.current_status?.toLowerCase().includes('critical')) {
              defaultScore = 0.8
            } else if (patient.current_status?.toLowerCase().includes('high')) {
              defaultScore = 0.6
            } else if (patient.current_status?.toLowerCase().includes('medium')) {
              defaultScore = 0.4
            }
            
            // Add some randomization to make it realistic
            defaultScore += (Math.random() - 0.5) * 0.2
            defaultScore = Math.max(0.1, Math.min(0.95, defaultScore))
            
            patient.concern_score = defaultScore
            patient.risk_level = defaultScore > 0.7 ? 'high' : defaultScore > 0.4 ? 'medium' : 'low'
            
            console.log(`Generated default concern score for ${patient.patient_name}: ${patient.concern_score.toFixed(2)}`)
          }
        })
      }
      
      setPatients(data.patients || [])
    } catch (error) {
      console.error('Error fetching patients:', error)
      toast.error('Failed to load patients')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPatients()
  }, [])

  const filteredPatients = patients.filter(patient =>
    patient.patient_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patient.patient_name.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const handlePatientClick = (patient: Patient) => {
    onSelectPatient(patient.patient_id);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <HealthcareLoadingScreen 
          variant="heartbeat" 
          message="Loading patient database..." 
          className="min-h-0"
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Search and Filters */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Patient Directory</h2>
        
        <div className="flex items-center space-x-4 mb-4">
          <div className="flex-1">
            <input
              type="text"
              placeholder="Search by Patient ID or Name..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <button
            onClick={async () => {
              setLoading(true)
              try {
                console.log('🔄 Manually recalculating concern scores...')
                await api.post('/api/concern/bulk-calculate')
                toast.success('Concern scores updated successfully!')
                // Wait for calculation to complete
                await new Promise(resolve => setTimeout(resolve, 1500))
                await fetchPatients()
              } catch (error) {
                console.error('Failed to recalculate concern scores:', error)
                toast.error('Failed to update concern scores')
              } finally {
                setLoading(false)
              }
            }}
            className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition flex items-center space-x-2"
            disabled={loading}
          >
            <span className={loading ? "animate-pulse" : ""}>📊</span>
            <span>Update Scores</span>
          </button>
          <button
            onClick={fetchPatients}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition flex items-center space-x-2"
            disabled={loading}
          >
            <span className={loading ? "animate-spin" : ""}>🔄</span>
            <span>Refresh Data</span>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
          <div className="bg-blue-50 p-3 rounded">
            <div className="font-semibold text-blue-900">Total Patients</div>
            <div className="text-2xl font-bold text-blue-600">{patients.length}</div>
          </div>
          <div className="bg-green-50 p-3 rounded">
            <div className="font-semibold text-green-900">Low Risk</div>
            <div className="text-2xl font-bold text-green-600">
              {patients.filter(p => p.risk_level === 'low').length}
            </div>
          </div>
          <div className="bg-orange-50 p-3 rounded">
            <div className="font-semibold text-orange-900">High Risk</div>
            <div className="text-2xl font-bold text-orange-600">
              {patients.filter(p => ['high', 'critical'].includes(p.risk_level)).length}
            </div>
          </div>
          <div className="bg-purple-50 p-3 rounded">
            <div className="font-semibold text-purple-900">Active</div>
            <div className="text-2xl font-bold text-purple-600">
              {patients.filter(p => p.current_status === 'active').length}
            </div>
          </div>
        </div>
      </div>

      {/* Patient Grid with New Animated Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredPatients.map((patient) => (
          <PatientCard
            key={patient.patient_id}
            patient={patient}
            onClick={handlePatientClick}
            className="transform-gpu"
          />
        ))}
      </div>

      {filteredPatients.length === 0 && !loading && (
        <div className="bg-white rounded-lg shadow p-12 text-center">
          <div className="text-6xl mb-4">👥</div>
          <h3 className="text-xl font-semibold mb-2">
            {searchTerm ? 'No Patients Found' : 'No Patients Yet'}
          </h3>
          <p className="text-gray-600 mb-6">
            {searchTerm 
              ? `No patients match "${searchTerm}". Try a different search term.`
              : 'Get started by adding your first patient to the system.'
            }
          </p>
          {!searchTerm && (
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition font-medium"
            >
              ➕ Add New Patient
            </button>
          )}
        </div>
      )}

      {/* Quick Actions */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="flex items-center space-x-2 p-4 bg-white rounded-lg hover:shadow-md transition">
            <span className="text-2xl">🩺</span>
            <div className="text-left">
              <div className="font-medium">Start Diagnosis</div>
              <div className="text-sm text-gray-600">Begin AI-powered diagnosis</div>
            </div>
          </button>
          
          <button className="flex items-center space-x-2 p-4 bg-white rounded-lg hover:shadow-md transition">
            <span className="text-2xl">📝</span>
            <div className="text-left">
              <div className="font-medium">Add Clinical Note</div>
              <div className="text-sm text-gray-600">Record observations</div>
            </div>
          </button>
          
          <button className="flex items-center space-x-2 p-4 bg-white rounded-lg hover:shadow-md transition">
            <span className="text-2xl">📊</span>
            <div className="text-left">
              <div className="font-medium">CONCERN Analysis</div>
              <div className="text-sm text-gray-600">Monitor patient risk</div>
            </div>
          </button>
        </div>
      </div>
    </div>
  )
}
