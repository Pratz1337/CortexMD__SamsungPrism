"use client"

import { useState, useEffect, useRef } from "react"
import { api, DiagnosisAPI } from "@/lib/api"
import { toast } from "react-hot-toast"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { HealthcareLoadingScreen } from "@/components/ui/HealthcareLoadingScreen"

interface Patient {
  patient_id: string
  concern_score: number
  risk_level: string
  last_updated: string
}

interface DashboardData {
  patients: Patient[]
  total_patients: number
  high_risk_count: number
  last_updated: string
}

interface ChartDataPoint {
  timestamp: string
  time: string
  criticalCount: number
  highCount: number
  mediumCount: number
  lowCount: number
  avgConcernScore: number
}

export function ConcernDashboard() {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedPatient, setSelectedPatient] = useState<string | null>(null)
  const [patientDetails, setPatientDetails] = useState<any>(null)
  const [alerts, setAlerts] = useState<any[]>([])
  const [criticalPatients, setCriticalPatients] = useState<any[]>([])
  const [showAlerts, setShowAlerts] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  const [chartData, setChartData] = useState<ChartDataPoint[]>([])
  const [patientTrends, setPatientTrends] = useState<{
    [key: string]: { timestamp: string; score: number; time: string }[]
  }>({})

  const fetchDashboardData = async () => {
    try {
      const { data } = await api.get("/api/concern/patients?fast=true")
      setDashboardData(data)

      const now = new Date()
      const timeString = now.toLocaleTimeString()

      const newDataPoint: ChartDataPoint = {
        timestamp: now.toISOString(),
        time: timeString,
        criticalCount: data.patients.filter((p:any) => p.risk_level === "critical").length,
        highCount: data.patients.filter((p:any) => p.risk_level === "high").length,
        mediumCount: data.patients.filter((p:any) => p.risk_level === "medium").length,
        lowCount: data.patients.filter((p:any) => p.risk_level === "low").length,
        avgConcernScore:
          data.patients.length > 0
            ? (data.patients.reduce((sum:any, p:any) => sum + p.concern_score, 0) / data.patients.length) * 100
            : 0,
      }

      setChartData((prev) => {
        const updated = [...prev, newDataPoint]
        return updated.slice(-20)
      })

      const newPatientTrends = { ...patientTrends }
      data.patients.forEach((patient:any) => {
        if (!newPatientTrends[patient.patient_id]) {
          newPatientTrends[patient.patient_id] = []
        }
        newPatientTrends[patient.patient_id].push({
          timestamp: now.toISOString(),
          time: timeString,
          score: patient.concern_score * 100,
        })
        newPatientTrends[patient.patient_id] = newPatientTrends[patient.patient_id].slice(-10)
      })
      setPatientTrends(newPatientTrends)

      const criticalCount = data.patients.filter((p:any) => p.risk_level === "critical").length
      if (criticalCount > 0) {
        toast.error(`⚠️ ${criticalCount} patients in CRITICAL condition!`, {
          duration: 6000,
          style: {
            background: "#DC2626",
            color: "white",
          },
        })
      }
    } catch (error) {
      console.error("Error fetching dashboard:", error)
      toast.error("Failed to load dashboard data")
    } finally {
      setLoading(false)
    }
  }

  const fetchPatientDetails = async (patientId: string) => {
    try {
      const { data } = await api.get(`/api/concern/patient/${patientId}`)
      setPatientDetails(data)
    } catch (error) {
      console.error("Error fetching patient details:", error)
      toast.error("Failed to load patient details")
    }
  }

  const fetchCriticalPatients = async () => {
    try {
      const data = await DiagnosisAPI.getCriticalPatients()
      setCriticalPatients(data.patients || [])
    } catch (error) {
      console.error("Error fetching critical patients:", error)
    }
  }

  const fetchAlerts = async () => {
    try {
      const data = await DiagnosisAPI.getAlerts()
      setAlerts(data.alerts || [])

      const unacknowledged = (data.alerts || []).filter((a:any) => !a.acknowledged)
      if (unacknowledged.length > 0) {
        toast.error(`${unacknowledged.length} unacknowledged alerts!`, {
          duration: 4000,
          style: {
            background: "#EF4444",
            color: "white",
          },
        })
      }
    } catch (error) {
      console.error("Error fetching alerts:", error)
    }
  }

  const acknowledgeAlert = async (alertId: string) => {
    try {
      await DiagnosisAPI.acknowledgeAlert(alertId)
      toast.success("Alert acknowledged")
      fetchAlerts()
    } catch (error) {
      console.error("Error acknowledging alert:", error)
      toast.error("Failed to acknowledge alert")
    }
  }

  const populateDemoData = async () => {
    try {
      const { data } = await api.post("/api/concern/demo/populate")
      toast.success(data.message || "Demo data populated")
      fetchDashboardData()
    } catch (error) {
      console.error("Error populating demo data:", error)
      toast.error("Failed to populate demo data")
    }
  }

  useEffect(() => {
    fetchDashboardData()
    fetchCriticalPatients()
    fetchAlerts()
  }, [])

  useEffect(() => {
    if (selectedPatient) {
      fetchPatientDetails(selectedPatient)
    }
  }, [selectedPatient])

  useEffect(() => {
    if (autoRefresh && !loading) {
      intervalRef.current = setInterval(() => {
        if (!loading) {
          fetchDashboardData()
          fetchCriticalPatients()
          fetchAlerts()
        }
      }, 60000) // Increased to 1 minute to reduce load
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [autoRefresh, loading])

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case "critical":
        return "text-red-800 bg-red-100"
      case "high":
        return "text-orange-800 bg-orange-100"
      case "medium":
        return "text-yellow-800 bg-yellow-100"
      case "low":
        return "text-green-800 bg-green-100"
      default:
        return "text-gray-800 bg-gray-100"
    }
  }

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case "critical":
        return "🚨"
      case "high":
        return "⚠️"
      case "medium":
        return "⚡"
      case "low":
        return "✅"
      default:
        return "❓"
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <HealthcareLoadingScreen 
          variant="pulse" 
          message="Loading concern dashboard..." 
          className="min-h-0"
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {criticalPatients.length > 0 && (
        <div className="bg-red-600 text-white rounded-lg p-4 mb-4 animate-pulse">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <span className="text-2xl">🚨</span>
              <div>
                <div className="font-bold text-lg">CRITICAL ALERT</div>
                <div>{criticalPatients.length} patients require immediate attention</div>
              </div>
            </div>
            <button
              onClick={() => setShowAlerts(!showAlerts)}
              className="px-4 py-2 bg-red-700 hover:bg-red-800 rounded-lg transition"
            >
              View Details
            </button>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">CONCERN Early Warning System</h2>
            <p className="text-gray-600">Real-time patient risk monitoring</p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-4 py-2 rounded-lg transition flex items-center space-x-2 ${
                autoRefresh
                  ? "bg-green-600 text-white hover:bg-green-700"
                  : "bg-gray-300 text-gray-700 hover:bg-gray-400"
              }`}
            >
              <span>{autoRefresh ? "🔄" : "⏸️"}</span>
              <span>{autoRefresh ? "Auto-refresh ON" : "Auto-refresh OFF"}</span>
            </button>
            <button
              onClick={populateDemoData}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
            >
              Populate Demo Data
            </button>
          </div>
        </div>

        {dashboardData && (
          <div className="space-y-6">
            {chartData.length > 1 && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Risk Level Trends</CardTitle>
                    <CardDescription>Real-time patient risk distribution over time</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="time" tick={{ fontSize: 12 }} stroke="#666" />
                          <YAxis tick={{ fontSize: 12 }} stroke="#666" />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "white",
                              border: "1px solid #ccc",
                              borderRadius: "8px",
                            }}
                          />
                          <Area
                            type="monotone"
                            dataKey="criticalCount"
                            stackId="1"
                            stroke="#DC2626"
                            fill="#DC2626"
                            fillOpacity={0.8}
                            name="Critical"
                            animationDuration={1500}
                            animationEasing="ease-in-out"
                          />
                          <Area
                            type="monotone"
                            dataKey="highCount"
                            stackId="1"
                            stroke="#EA580C"
                            fill="#EA580C"
                            fillOpacity={0.8}
                            name="High"
                            animationDuration={1500}
                            animationEasing="ease-in-out"
                          />
                          <Area
                            type="monotone"
                            dataKey="mediumCount"
                            stackId="1"
                            stroke="#D97706"
                            fill="#D97706"
                            fillOpacity={0.8}
                            name="Medium"
                            animationDuration={1500}
                            animationEasing="ease-in-out"
                          />
                          <Area
                            type="monotone"
                            dataKey="lowCount"
                            stackId="1"
                            stroke="#16A34A"
                            fill="#16A34A"
                            fillOpacity={0.8}
                            name="Low"
                            animationDuration={1500}
                            animationEasing="ease-in-out"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Average Concern Score</CardTitle>
                    <CardDescription>Overall patient concern level over time</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="time" tick={{ fontSize: 12 }} stroke="#666" />
                          <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} stroke="#666" />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "white",
                              border: "1px solid #ccc",
                              borderRadius: "8px",
                            }}
                            formatter={(value: number) => [`${value.toFixed(1)}%`, "Avg Concern Score"]}
                          />
                          <Line
                            type="monotone"
                            dataKey="avgConcernScore"
                            stroke="#8B5CF6"
                            strokeWidth={3}
                            dot={{ fill: "#8B5CF6", strokeWidth: 2, r: 4 }}
                            activeDot={{ r: 6, stroke: "#8B5CF6", strokeWidth: 2 }}
                            animationDuration={1500}
                            animationEasing="ease-in-out"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-red-50 border-2 border-red-200 p-4 rounded-lg transform transition-all duration-500 hover:scale-105">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-2xl font-bold text-red-600 transition-all duration-1000">
                      {dashboardData.patients.filter((p) => p.risk_level === "critical").length}
                    </div>
                    <div className="text-sm text-gray-600">Critical Risk</div>
                  </div>
                  <span className="text-3xl animate-pulse">🚨</span>
                </div>
              </div>
              <div className="bg-orange-50 border-2 border-orange-200 p-4 rounded-lg transform transition-all duration-500 hover:scale-105">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-2xl font-bold text-orange-600 transition-all duration-1000">
                      {dashboardData.patients.filter((p) => p.risk_level === "high").length}
                    </div>
                    <div className="text-sm text-gray-600">High Risk</div>
                  </div>
                  <span className="text-3xl">⚠️</span>
                </div>
              </div>
              <div className="bg-yellow-50 border-2 border-yellow-200 p-4 rounded-lg transform transition-all duration-500 hover:scale-105">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-2xl font-bold text-yellow-600 transition-all duration-1000">
                      {dashboardData.patients.filter((p) => p.risk_level === "medium").length}
                    </div>
                    <div className="text-sm text-gray-600">Medium Risk</div>
                  </div>
                  <span className="text-3xl">⚡</span>
                </div>
              </div>
              <div className="bg-green-50 border-2 border-green-200 p-4 rounded-lg transform transition-all duration-500 hover:scale-105">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-2xl font-bold text-green-600 transition-all duration-1000">
                      {dashboardData.patients.filter((p) => p.risk_level === "low").length}
                    </div>
                    <div className="text-sm text-gray-600">Low Risk</div>
                  </div>
                  <span className="text-3xl">✅</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg transform transition-all duration-500 hover:scale-105">
                <div className="text-2xl font-bold text-blue-600 transition-all duration-1000">
                  {dashboardData.total_patients}
                </div>
                <div className="text-sm text-gray-600">Total Patients Monitored</div>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg transform transition-all duration-500 hover:scale-105">
                <div className="text-2xl font-bold text-purple-600 transition-all duration-1000">{alerts.length}</div>
                <div className="text-sm text-gray-600">Active Alerts</div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="text-2xl font-bold text-gray-600">{new Date().toLocaleTimeString()}</div>
                <div className="text-sm text-gray-600">Last Updated</div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold">Patient Risk Status</h3>
        </div>

        <div className="divide-y">
          {dashboardData?.patients.map((patient) => (
            <div
              key={patient.patient_id}
              className={`p-4 cursor-pointer hover:bg-gray-50 transition-all duration-300 ${
                selectedPatient === patient.patient_id ? "bg-blue-50 border-l-4 border-blue-500" : ""
              }`}
              onClick={() => setSelectedPatient(patient.patient_id)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <span className="text-xl">{getRiskIcon(patient.risk_level)}</span>
                  <div>
                    <div className="font-medium">{patient.patient_id}</div>
                    <div className="text-sm text-gray-500">
                      Last updated: {new Date(patient.last_updated || "").toLocaleString()}
                    </div>
                    {patientTrends[patient.patient_id] && patientTrends[patient.patient_id].length > 1 && (
                      <div className="mt-2 h-8 w-32">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={patientTrends[patient.patient_id]}>
                            <Line
                              type="monotone"
                              dataKey="score"
                              stroke="#8B5CF6"
                              strokeWidth={2}
                              dot={false}
                              animationDuration={1000}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <div className="text-right">
                    <div className="text-lg font-bold transition-all duration-1000">
                      {(patient.concern_score * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-500">Concern Score</div>
                  </div>
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-medium transition-all duration-500 ${getRiskColor(patient.risk_level)}`}
                  >
                    {patient.risk_level.toUpperCase()}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {(!dashboardData?.patients || dashboardData.patients.length === 0) && (
          <div className="p-8 text-center text-gray-500">
            <div className="text-4xl mb-2">🏥</div>
            <div>No patients currently monitored</div>
            <div className="text-sm">Click "Populate Demo Data" to add sample patients</div>
          </div>
        )}
      </div>

      {showAlerts && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b flex justify-between items-center bg-red-50">
              <CardTitle className="text-lg">Critical Alerts & Warnings</CardTitle>
              <button onClick={() => setShowAlerts(false)} className="text-red-400 hover:text-red-600 text-2xl">
                ×
              </button>
            </div>

            <div className="p-6 space-y-4">
              {criticalPatients.length > 0 && (
                <div>
                  <h4 className="font-semibold text-red-700 mb-3">Critical Patients Requiring Immediate Attention</h4>
                  <div className="space-y-3">
                    {criticalPatients.map((patient, index) => (
                      <div key={index} className="bg-red-50 border-2 border-red-200 p-4 rounded-lg">
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-bold text-lg">{patient.patient_id}</div>
                            <div className="text-red-600">Risk Score: {(patient.concern_score * 100).toFixed(1)}%</div>
                            <div className="text-sm text-gray-600 mt-1">
                              {patient.risk_factors?.join(", ") || "Multiple risk factors detected"}
                            </div>
                          </div>
                          <button
                            onClick={() => {
                              setSelectedPatient(patient.patient_id)
                              setShowAlerts(false)
                            }}
                            className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
                          >
                            View Details
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {alerts.length > 0 && (
                <div>
                  <h4 className="font-semibold text-orange-700 mb-3">Recent Alerts</h4>
                  <div className="space-y-3">
                    {alerts.slice(0, 10).map((alert) => (
                      <div key={alert.id} className="bg-orange-50 border border-orange-200 p-4 rounded-lg">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2">
                              <span
                                className={`px-2 py-1 rounded text-xs font-medium ${
                                  alert.severity === "critical"
                                    ? "bg-red-100 text-red-800"
                                    : alert.severity === "high"
                                      ? "bg-orange-100 text-orange-800"
                                      : alert.severity === "medium"
                                        ? "bg-yellow-100 text-yellow-800"
                                        : "bg-blue-100 text-blue-800"
                                }`}
                              >
                                {alert.severity?.toUpperCase() || "ALERT"}
                              </span>
                              <span className="font-medium">{alert.patient_id}</span>
                              <span className="text-sm text-gray-500">
                                {new Date(alert.timestamp || Date.now()).toLocaleString()}
                              </span>
                            </div>
                            <div className="mt-2 text-gray-700">{alert.message}</div>
                          </div>
                          {!alert.acknowledged && (
                            <button
                              onClick={() => acknowledgeAlert(alert.id)}
                              className="ml-4 px-3 py-1 bg-orange-600 text-white rounded hover:bg-orange-700 text-sm"
                            >
                              Acknowledge
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {selectedPatient && patientDetails && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b flex justify-between items-center">
              <h3 className="text-xl font-semibold">Patient Details: {selectedPatient}</h3>
              <button onClick={() => setSelectedPatient(null)} className="text-gray-400 hover:text-gray-600">
                ×
              </button>
            </div>

            <div className="p-6 space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">Current Risk Assessment</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Concern Score:</span>
                      <span className="font-bold">{(patientDetails.current_concern_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Risk Level:</span>
                      <span className={`px-2 py-1 rounded text-sm ${getRiskColor(patientDetails.current_risk_level)}`}>
                        {patientDetails.current_risk_level.toUpperCase()}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">Activity (24h)</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Visits:</span>
                      <span className="font-bold">{patientDetails.visits_24h}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Notes:</span>
                      <span className="font-bold">{patientDetails.notes_24h}</span>
                    </div>
                  </div>
                </div>
              </div>

              {patientDetails.risk_factors && patientDetails.risk_factors.length > 0 && (
                <div>
                  <h4 className="font-semibold mb-2">Risk Factors</h4>
                  <div className="space-y-2">
                    {patientDetails.risk_factors.map((factor: string, index: number) => (
                      <div key={index} className="flex items-center space-x-2">
                        <span className="text-orange-500">⚠️</span>
                        <span>{factor}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {patientDetails.metadata_patterns && Object.keys(patientDetails.metadata_patterns).length > 0 && (
                <div>
                  <h4 className="font-semibold mb-2">Metadata Patterns</h4>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <pre className="text-sm">{JSON.stringify(patientDetails.metadata_patterns, null, 2)}</pre>
                  </div>
                </div>
              )}

              {patientDetails.score_trend && patientDetails.score_trend.length > 0 && (
                <div>
                  <h4 className="font-semibold mb-2">Score Trend</h4>
                  <div className="space-y-2">
                    {patientDetails.score_trend.slice(0, 5).map((entry: any, index: number) => (
                      <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                        <span className="text-sm">{new Date(entry.timestamp).toLocaleString()}</span>
                        <div className="flex items-center space-x-2">
                          <span className="font-bold">{(entry.score * 100).toFixed(1)}%</span>
                          <span className={`px-2 py-1 rounded text-xs ${getRiskColor(entry.level)}`}>
                            {entry.level.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
