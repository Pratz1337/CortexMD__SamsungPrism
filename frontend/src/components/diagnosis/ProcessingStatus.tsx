"use client"

import { useDiagnosisStore } from "@/store/diagnosisStore"
import { DebugVisualization } from "./DebugVisualization"
import { createPortal } from "react-dom"
import { useEffect, useState } from "react"

export function ProcessingStatus() {
  const { processingStatus, setProcessingStatus, setIsLoading } = useDiagnosisStore()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    return () => setMounted(false)
  }, [])

  if (!processingStatus || !mounted) return null

  return createPortal(
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        <div className="p-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-bold text-gray-800">🔍 Backend Debug Stream</h3>
              <p className="text-sm text-gray-600">Real-time processing visualization</p>
            </div>
            <button
              onClick={() => {
                setProcessingStatus(null)
                setIsLoading(false)
              }}
              className="text-gray-500 hover:text-gray-700"
            >
              ✕
            </button>
          </div>

          <DebugVisualization />
        </div>
      </div>
    </div>,
    document.body
  )
}
