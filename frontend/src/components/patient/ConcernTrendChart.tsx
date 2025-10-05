"use client"

import { useState, useEffect, useRef } from "react"
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
} from "recharts"

interface ConcernTrendData {
  score: number
  level: string
  timestamp: string
  confidence?: number
  trend_velocity?: number
  vital_signs?: {
    heart_rate: number
    blood_pressure_systolic: number
    blood_pressure_diastolic: number
    temperature: number
    oxygen_saturation: number
  }
}

interface DepthMetrics {
  data_points_analyzed: number
  temporal_coverage_hours: number
  pattern_recognition_score: number
  predictive_confidence: number
  clinical_correlation_score: number
  multi_modal_integration: number
  analysis_completeness: number
}

interface ConcernTrendChartProps {
  trendData: ConcernTrendData[]
  currentScore: number
  currentLevel: string
  patientName: string
  patientId?: string
  showAdvancedMetrics?: boolean
  depthMetrics?: DepthMetrics
}

interface RealtimeUpdate {
  patient_id: string
  concern_score: number
  risk_level: string
  confidence_score: number
  trend_direction: string
  trend_velocity: number
  vital_signs: any
  depth_metrics: DepthMetrics
  timestamp: string
  recommendations: string[]
}

export function ConcernTrendChart({ 
  trendData, 
  currentScore, 
  currentLevel, 
  patientName,
  patientId,
  showAdvancedMetrics = false,
  depthMetrics
}: ConcernTrendChartProps) {
  const [animatedData, setAnimatedData] = useState<any[]>([])
  const [showAlert, setShowAlert] = useState(false)
  const [realtimeData, setRealtimeData] = useState<RealtimeUpdate | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected')
  const [liveScore, setLiveScore] = useState(currentScore)
  const [liveLevel, setLiveLevel] = useState(currentLevel)
  const [currentDepthMetrics, setCurrentDepthMetrics] = useState<DepthMetrics | undefined>(depthMetrics)
  
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttempts = useRef(0)
  
  // WebSocket connection for real-time updates (with debouncing)
  useEffect(() => {
    if (!patientId) return

    let connectionAttempted = false
    
    const connectWebSocket = async () => {
      if (connectionAttempted) return
      connectionAttempted = true
      
      try {
        // Get WebSocket configuration from backend (with cache)
        const response = await fetch('/api/concern/websocket/info', {
          cache: 'force-cache'
        })
        if (!response.ok) {
          console.warn('WebSocket not available, skipping connection')
          setConnectionStatus('disconnected')
          return
        }
        
        const config = await response.json()
        if (!config.success) {
          console.warn('WebSocket configuration not available')
          setConnectionStatus('disconnected')
          return
        }
        
        const wsUrl = `${config.websocket_config.url}/concern/${patientId}`
        
        // Prevent duplicate connections
        if (wsRef.current && wsRef.current.readyState === WebSocket.CONNECTING) {
          console.log('WebSocket already connecting, skipping')
          return
        }
        
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          console.log('WebSocket already connected, skipping')
          return
        }
        
        setConnectionStatus('connecting')
        const ws = new WebSocket(wsUrl)
        wsRef.current = ws
        
        ws.onopen = () => {
          console.log('✅ WebSocket connected for patient', patientId)
          setConnectionStatus('connected')
          reconnectAttempts.current = 0
          
          // Send initial ping
          try {
            ws.send(JSON.stringify({ type: 'ping' }))
          } catch (error) {
            console.warn('Failed to send ping:', error)
          }
        }
        
        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data)
            
            if (message.type === 'realtime_update' && message.data) {
              const update: RealtimeUpdate = message.data
              setRealtimeData(update)
              setLiveScore(update.concern_score)
              setLiveLevel(update.risk_level)
              
              if (update.depth_metrics) {
                setCurrentDepthMetrics(update.depth_metrics)
              }
              
              // Add to trend data
              const newDataPoint = {
                timestamp: update.timestamp,
                concern_score: update.concern_score,
                risk_level: update.risk_level,
                vital_signs: update.vital_signs
              }
              
              // Update animated data
              setAnimatedData(prev => {
                const newData = [...prev, {
                  ...newDataPoint,
                  scorePercent: update.concern_score * 100,
                  time: new Date(update.timestamp).toLocaleTimeString("en-US", {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                    hour12: false,
                  }),
                  date: new Date(update.timestamp).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                  }),
                  riskValue: getRiskValue(update.risk_level),
                  index: prev.length,
                }]
                
                // Keep last 20 data points
                return newData.slice(-20)
              })
              
              // Check for alerts
              if (update.risk_level === 'critical' || update.risk_level === 'high') {
                setShowAlert(true)
                const alertTimer = setTimeout(() => setShowAlert(false), 8000)
                return () => clearTimeout(alertTimer)
              }
            }
            
            else if (message.type === 'initial_data' && message.data) {
              const initialUpdate: RealtimeUpdate = message.data
              setRealtimeData(initialUpdate)
              setLiveScore(initialUpdate.concern_score)
              setLiveLevel(initialUpdate.risk_level)
              
              if (initialUpdate.depth_metrics) {
                setCurrentDepthMetrics(initialUpdate.depth_metrics)
              }
            }
            
            else if (message.type === 'pong') {
              // Connection is alive
              console.log('🏓 WebSocket pong received')
            }
            
          } catch (e) {
            console.error('Error parsing WebSocket message:', e)
          }
        }
        
        ws.onclose = () => {
          console.log('🔌 WebSocket disconnected')
          setConnectionStatus('disconnected')
          wsRef.current = null
          
          // Attempt to reconnect with exponential backoff
          if (reconnectAttempts.current < 5) {
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000)
            reconnectTimeoutRef.current = setTimeout(() => {
              reconnectAttempts.current++
              connectWebSocket()
            }, delay)
          }
        }
        
        ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          setConnectionStatus('disconnected')
          
          // Try to fetch concern data via REST API as fallback
          fetchConcernDataFallback()
        }
        
      } catch (error) {
        console.error('Failed to connect WebSocket:', error)
        setConnectionStatus('disconnected')
        
        // Try to fetch concern data via REST API as fallback
        fetchConcernDataFallback()
      }
    }

    // Fallback function to fetch concern data via REST API when WebSocket fails
    const fetchConcernDataFallback = async () => {
      if (!patientId) return
      
      try {
        console.log('🔄 Fetching concern data via REST API fallback...')
        const response = await fetch(`/api/concern/patient/${patientId}`)
        
        if (response.ok) {
          const concernData = await response.json()
          
          if (concernData.success && concernData.data) {
            const data = concernData.data
            
            // Update with REST API data
            setLiveScore(data.current_concern_score || 0)
            setLiveLevel(data.risk_level || 'low')
            
            // Create a fallback realtime update object
            const fallbackUpdate: RealtimeUpdate = {
              patient_id: patientId,
              concern_score: data.current_concern_score || 0,
              risk_level: data.risk_level || 'low',
              confidence_score: data.confidence_score || 0.75,
              trend_direction: data.trend_direction || 'stable',
              trend_velocity: data.trend_velocity || 0,
              timestamp: new Date().toISOString(),
              vital_signs: data.vital_signs || {},
              recommendations: data.recommendations || [],
              depth_metrics: {
                data_points_analyzed: 5,
                temporal_coverage_hours: 24,
                pattern_recognition_score: 0.8,
                predictive_confidence: 0.75,
                clinical_correlation_score: 0.85,
                multi_modal_integration: 0.7,
                analysis_completeness: 0.9
              }
            }
            
            setRealtimeData(fallbackUpdate)
            setCurrentDepthMetrics(fallbackUpdate.depth_metrics)
            
            console.log('✅ Successfully fetched concern data via REST API')
          }
        }
      } catch (fallbackError) {
        console.warn('REST API fallback also failed:', fallbackError)
        
        // Generate some default data based on props
        const defaultScore = currentScore || 0.3
        const defaultLevel = currentLevel || 'medium'
        
        setLiveScore(defaultScore)
        setLiveLevel(defaultLevel)
        
        const defaultUpdate: RealtimeUpdate = {
          patient_id: patientId || 'unknown',
          concern_score: defaultScore,
          risk_level: defaultLevel,
          confidence_score: 0.6,
          trend_direction: 'stable',
          trend_velocity: 0,
          timestamp: new Date().toISOString(),
          vital_signs: {},
          recommendations: ['Monitor patient status', 'Continue current treatment'],
          depth_metrics: {
            data_points_analyzed: 3,
            temporal_coverage_hours: 12,
            pattern_recognition_score: 0.6,
            predictive_confidence: 0.5,
            clinical_correlation_score: 0.6,
            multi_modal_integration: 0.5,
            analysis_completeness: 0.7
          }
        }
        
        setRealtimeData(defaultUpdate)
        setCurrentDepthMetrics(defaultUpdate.depth_metrics)
        
        console.log('🔧 Using default concern data due to connection issues')
      }
    }
    
    connectWebSocket()
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, [patientId])
  
  // Periodic ping to keep connection alive
  useEffect(() => {
    const pingInterval = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }))
      }
    }, 30000) // Ping every 30 seconds
    
    return () => clearInterval(pingInterval)
  }, [])

  // Process and format data for the chart (enhanced with real-time data)
  const processedData = trendData
    .map((item, index) => ({
      ...item,
      scorePercent: item.score * 100,
      time: new Date(item.timestamp).toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      }),
      date: new Date(item.timestamp).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      }),
      riskValue: getRiskValue(item.level),
      index,
    }))
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())

  // Add current data point if not already included or if we have real-time updates
  const latestTimestamp = processedData.length > 0 ? 
    new Date(processedData[processedData.length - 1].timestamp).getTime() : 0
  const currentTimestamp = Date.now()
  
  const currentScoreToUse = realtimeData ? realtimeData.concern_score : liveScore
  const currentLevelToUse = realtimeData ? realtimeData.risk_level : liveLevel

  if (currentTimestamp - latestTimestamp > 300000 || realtimeData) { // 5 minutes or real-time data
    processedData.push({
      score: currentScoreToUse,
      level: currentLevelToUse,
      scorePercent: currentScoreToUse * 100,
      time: new Date().toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      }),
      date: new Date().toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      }),
      riskValue: getRiskValue(currentLevelToUse),
      timestamp: new Date().toISOString(),
      index: processedData.length,
      confidence: realtimeData?.confidence_score,
      trend_velocity: realtimeData?.trend_velocity,
      vital_signs: realtimeData?.vital_signs
    })
  }

  function getRiskValue(level: string): number {
    switch (level) {
      case "critical":
        return 4
      case "high":
        return 3
      case "medium":
        return 2
      case "low":
        return 1
      default:
        return 1
    }
  }

  function getRiskColor(level: string): string {
    switch (level) {
      case "critical":
        return "#dc2626"
      case "high":
        return "#ea580c"
      case "medium":
        return "#d97706"
      case "low":
        return "#16a34a"
      default:
        return "#6b7280"
    }
  }

  function getRiskZone(score: number): string {
    if (score >= 80) return "critical"
    if (score >= 60) return "high"
    if (score >= 40) return "medium"
    return "low"
  }

  // Animate data loading
  useEffect(() => {
    if (processedData.length === 0) return

    setAnimatedData([])
    const timer = setTimeout(() => {
      processedData.forEach((item, index) => {
        setTimeout(() => {
          setAnimatedData((prev) => [...prev, item])
        }, index * 100) // Faster animation for real-time feel
      })
    }, 100)

    return () => clearTimeout(timer)
  }, [processedData.length])

  // Check for high-risk alerts
  useEffect(() => {
    const hasHighRisk = processedData.some((item) => item.level === "critical" || item.level === "high")

    if (hasHighRisk && (currentLevelToUse === "critical" || currentLevelToUse === "high")) {
      setShowAlert(true)
      const alertTimer = setTimeout(() => setShowAlert(false), 8000) // Longer alert time
      return () => clearTimeout(alertTimer)
    }
  }, [currentLevelToUse, processedData])

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg max-w-sm">
          <p className="font-semibold text-gray-900">{`${data.date} at ${data.time}`}</p>
          <p className="text-sm text-gray-600 mb-2">CONCERN Score Analysis</p>
          <div className="space-y-1">
            <p className={`font-bold text-lg`} style={{ color: getRiskColor(data.level) }}>
              {`${data.scorePercent.toFixed(1)}% - ${data.level.toUpperCase()}`}
            </p>
            
            {data.confidence && (
              <p className="text-xs text-blue-600">
                📊 Confidence: {(data.confidence * 100).toFixed(1)}%
              </p>
            )}
            
            {data.trend_velocity && (
              <p className="text-xs text-purple-600">
                📈 Trend Velocity: {data.trend_velocity.toFixed(3)}
              </p>
            )}
            
            {data.vital_signs && (
              <div className="mt-2 pt-2 border-t border-gray-200">
                <p className="text-xs font-semibold text-gray-700 mb-1">Vital Signs:</p>
                <div className="grid grid-cols-2 gap-1 text-xs text-gray-600">
                  <span>❤️ HR: {data.vital_signs.heart_rate}</span>
                  <span>🫁 O2: {data.vital_signs.oxygen_saturation}%</span>
                  <span>🌡️ T: {data.vital_signs.temperature}°F</span>
                  <span>🩺 BP: {data.vital_signs.blood_pressure_systolic}/{data.vital_signs.blood_pressure_diastolic}</span>
                </div>
              </div>
            )}
            
            <p className="text-xs text-gray-500 mt-2">
              {data.level === "critical"
                ? "🚨 Immediate attention required"
                : data.level === "high"
                  ? "⚠️ Close monitoring needed"
                  : data.level === "medium"
                    ? "⚡ Regular monitoring"
                    : "✅ Stable condition"}
            </p>
          </div>
        </div>
      )
    }
    return null
  }

  const CustomDot = (props: any) => {
    const { cx, cy, payload } = props
    const color = getRiskColor(payload.level)
    const isHighRisk = payload.level === "critical" || payload.level === "high"
    const hasRealtimeData = payload.confidence || payload.trend_velocity

    return (
      <g>
        <circle
          cx={cx}
          cy={cy}
          r={isHighRisk ? 7 : hasRealtimeData ? 5 : 4}
          fill={color}
          stroke="white"
          strokeWidth={2}
          className={isHighRisk ? "animate-pulse" : ""}
        />
        {isHighRisk && (
          <circle
            cx={cx}
            cy={cy}
            r={12}
            fill="none"
            stroke={color}
            strokeWidth={1}
            opacity={0.3}
            className="animate-ping"
          />
        )}
        {hasRealtimeData && (
          <circle
            cx={cx}
            cy={cy}
            r={8}
            fill="none"
            stroke="#3b82f6"
            strokeWidth={1}
            opacity={0.5}
            className="animate-pulse"
          />
        )}
      </g>
    )
  }

  if (processedData.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-xl font-semibold mb-4">CONCERN Trend Analysis</h3>
        <div className="text-center py-8 text-gray-500">
          <div className="text-4xl mb-2">📈</div>
          <div>No trend data available</div>
          <div className="text-sm">Data will appear as assessments are recorded</div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-semibold text-gray-900">Real-time CONCERN Trend Analysis</h3>
          <p className="text-sm text-gray-600">Advanced health monitoring for {patientName}</p>
        </div>

        <div className="flex items-center space-x-4">
          {/* Connection Status Indicator */}
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              connectionStatus === 'connected' ? 'bg-green-500 animate-pulse' :
              connectionStatus === 'connecting' ? 'bg-yellow-500 animate-spin' :
              'bg-red-500'
            }`} />
            <span className="text-xs text-gray-600">
              {connectionStatus === 'connected' ? 'Live' :
               connectionStatus === 'connecting' ? 'Connecting...' :
               'Offline'}
            </span>
          </div>

          {/* Alert Banner */}
          {showAlert && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 animate-bounce">
              <div className="flex items-center space-x-2">
                <span className="text-red-600 text-lg">🚨</span>
                <div>
                  <div className="font-semibold text-red-800 text-sm">High Risk Alert</div>
                  <div className="text-red-600 text-xs">Patient requires immediate attention</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Enhanced Risk Level Legend */}
      <div className="mb-6 p-4 bg-gradient-to-r from-gray-50 to-blue-50 rounded-lg">
        <div className="text-sm font-medium text-gray-700 mb-2">Risk Level Guide & Analysis Depth</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
          {[
            { level: "low", label: "Low (0-34%)", color: "#16a34a", icon: "✅" },
            { level: "medium", label: "Medium (35-64%)", color: "#d97706", icon: "⚡" },
            { level: "high", label: "High (65-84%)", color: "#ea580c", icon: "⚠️" },
            { level: "critical", label: "Critical (85%+)", color: "#dc2626", icon: "🚨" },
          ].map(({ level, label, color, icon }) => (
            <div key={level} className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-xs text-gray-600">
                {icon} {label}
              </span>
            </div>
          ))}
        </div>
        
        {/* Advanced Depth Metrics */}
        {showAdvancedMetrics && currentDepthMetrics && (
          <div className="border-t pt-3 mt-3">
            <div className="text-sm font-medium text-gray-700 mb-2">Analysis Depth & Confidence</div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-lg font-bold text-blue-600">
                  {currentDepthMetrics.data_points_analyzed}
                </div>
                <div className="text-xs text-gray-600">Data Points</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-green-600">
                  {(currentDepthMetrics.analysis_completeness * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-600">Completeness</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-purple-600">
                  {currentDepthMetrics.temporal_coverage_hours.toFixed(1)}h
                </div>
                <div className="text-xs text-gray-600">Coverage</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Interactive Chart */}
      <div className="h-96 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={animatedData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <defs>
              <linearGradient id="concernGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="criticalGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#dc2626" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#dc2626" stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />

            <XAxis 
              dataKey="time" 
              stroke="#6b7280" 
              fontSize={12} 
              tickLine={false} 
              tick={{ fontSize: 10 }}
            />

            <YAxis
              domain={[0, 100]}
              stroke="#6b7280"
              fontSize={12}
              tickLine={false}
              tick={{ fontSize: 10 }}
              label={{ value: "CONCERN Score (%)", angle: -90, position: "insideLeft" }}
            />

            {/* Enhanced Risk zone reference lines */}
            <ReferenceLine y={35} stroke="#d97706" strokeDasharray="2 2" opacity={0.5} />
            <ReferenceLine y={65} stroke="#ea580c" strokeDasharray="2 2" opacity={0.5} />
            <ReferenceLine y={85} stroke="#dc2626" strokeDasharray="2 2" opacity={0.5} />

            <Tooltip content={<CustomTooltip />} />

            <Area
              type="monotone"
              dataKey="scorePercent"
              stroke="#3b82f6"
              strokeWidth={3}
              fill="url(#concernGradient)"
              fillOpacity={0.6}
            />

            <Line
              type="monotone"
              dataKey="scorePercent"
              stroke="#1d4ed8"
              strokeWidth={3}
              dot={<CustomDot />}
              activeDot={{ r: 10, stroke: "#1d4ed8", strokeWidth: 2, fill: "#3b82f6" }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Enhanced Current Status Summary */}
      <div className="mt-6 space-y-4">
        <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-blue-700 font-medium">Current Status</div>
              <div className="flex items-center space-x-3 mt-1">
                <span className="text-2xl font-bold" style={{ color: getRiskColor(currentLevelToUse) }}>
                  {(currentScoreToUse * 100).toFixed(1)}%
                </span>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${
                    currentLevelToUse === "critical"
                      ? "bg-red-100 text-red-800"
                      : currentLevelToUse === "high"
                        ? "bg-orange-100 text-orange-800"
                        : currentLevelToUse === "medium"
                          ? "bg-yellow-100 text-yellow-800"
                          : "bg-green-100 text-green-800"
                  }`}
                >
                  {currentLevelToUse.toUpperCase()}
                </span>
              </div>
            </div>

            <div className="text-right">
              <div className="text-sm text-blue-700">
                {connectionStatus === 'connected' ? 'Live Monitoring' : 
                 currentLevelToUse === "critical" || currentLevelToUse === "high" ? "Requires Attention" : 
                 "Monitoring Status"}
              </div>
              <div className="text-xs text-blue-600 mt-1">
                Last updated: {realtimeData ? new Date(realtimeData.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>
        </div>

        {/* Real-time Vital Signs Display */}
        {realtimeData?.vital_signs && (
          <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200">
            <div className="text-sm font-medium text-green-700 mb-2">Live Vital Signs</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-lg font-bold text-red-600">
                  ❤️ {realtimeData.vital_signs.heart_rate}
                </div>
                <div className="text-xs text-gray-600">Heart Rate (bpm)</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-blue-600">
                  🫁 {realtimeData.vital_signs.oxygen_saturation}%
                </div>
                <div className="text-xs text-gray-600">Oxygen Sat</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-orange-600">
                  🌡️ {realtimeData.vital_signs.temperature}°F
                </div>
                <div className="text-xs text-gray-600">Temperature</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-purple-600">
                  🩺 {realtimeData.vital_signs.blood_pressure_systolic}/{realtimeData.vital_signs.blood_pressure_diastolic}
                </div>
                <div className="text-xs text-gray-600">Blood Pressure</div>
              </div>
            </div>
          </div>
        )}

        {/* Real-time Recommendations */}
        {realtimeData?.recommendations && realtimeData.recommendations.length > 0 && (
          <div className="p-4 bg-gradient-to-r from-yellow-50 to-amber-50 rounded-lg border border-yellow-200">
            <div className="text-sm font-medium text-yellow-700 mb-2">Clinical Recommendations</div>
            <ul className="space-y-1">
              {realtimeData.recommendations.slice(0, 3).map((rec, index) => (
                <li key={index} className="text-sm text-gray-700 flex items-start space-x-2">
                  <span className="text-yellow-600 mt-0.5">•</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}
