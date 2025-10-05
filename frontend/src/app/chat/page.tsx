"use client"

import { useState } from "react"
import { ChatInterface } from "@/components/chat/ChatInterface"
import { ConfidenceVisualization } from "@/components/chat/ConfidenceVisualization"
import { ReasoningBreakdown } from "@/components/chat/ReasoningBreakdown"
import { DetailedExplanations } from "@/components/chat/DetailedExplanations"

// Sample data for testing the explainability features
const sampleMetrics = {
  confidence: 0.87,
  fol_verified: true,
  explainability: 0.92,
  response_time: 2.4,
  medical_accuracy: 0.84,
  source_reliability: 0.91,
}

const sampleReasoningSteps = [
  {
    id: 1,
    step_type: "premise" as const,
    content: "Patient presents with chest pain, shortness of breath, and elevated cardiac enzymes.",
    confidence: 0.95,
    verified: true,
    reasoning: "These symptoms form the foundational evidence for cardiac analysis.",
    evidence_sources: ["ECG results", "Troponin levels", "Patient symptoms"],
    medical_context: "Classic presentation of acute coronary syndrome requires immediate evaluation.",
  },
  {
    id: 2,
    step_type: "inference" as const,
    content: "The combination of symptoms and biomarkers suggests acute myocardial infarction.",
    confidence: 0.89,
    verified: true,
    reasoning: "Clinical correlation between symptoms and diagnostic markers indicates cardiac tissue damage.",
    dependencies: [1],
    evidence_sources: ["Cardiology guidelines", "Clinical studies"],
    medical_context: "Elevated troponin with typical chest pain has high diagnostic value for MI.",
  },
  {
    id: 3,
    step_type: "verification" as const,
    content: "ECG changes confirm ST-elevation myocardial infarction (STEMI).",
    confidence: 0.92,
    verified: true,
    reasoning: "ST-segment elevation in multiple leads provides definitive diagnostic confirmation.",
    dependencies: [1, 2],
    evidence_sources: ["12-lead ECG", "AHA guidelines"],
    medical_context: "STEMI diagnosis requires immediate reperfusion therapy.",
  },
  {
    id: 4,
    step_type: "conclusion" as const,
    content: "Recommended immediate cardiac catheterization and PCI within 90 minutes.",
    confidence: 0.94,
    verified: true,
    reasoning: "Time-sensitive intervention is critical for optimal patient outcomes in STEMI.",
    dependencies: [1, 2, 3],
    evidence_sources: ["ACC/AHA guidelines", "Door-to-balloon protocols"],
    medical_context: "Primary PCI is the gold standard for STEMI treatment when available.",
  },
]

const sampleExplanations = [
  {
    type: "Clinical Analysis",
    content:
      "The patient's presentation is highly consistent with acute ST-elevation myocardial infarction (STEMI). The combination of typical chest pain, elevated cardiac enzymes (troponin), and characteristic ECG changes provides strong diagnostic evidence.",
    confidence: 0.91,
    icon: "🩺",
  },
  {
    type: "Diagnostic Reasoning",
    content:
      "The ST-segment elevation pattern in leads II, III, and aVF indicates inferior wall MI, likely due to right coronary artery occlusion. The timeline and enzyme elevation pattern support acute presentation requiring immediate intervention.",
    confidence: 0.88,
    icon: "📊",
  },
]

const sampleFOLVerification = {
  total_predicates: 12,
  verified_count: 10,
  verification_score: 83,
  status: "VERIFIED" as const,
  reasoning:
    "The logical chain from symptoms → biomarkers → ECG changes → diagnosis has been verified using first-order logic predicates. High confidence in the diagnostic conclusion.",
  predicate_breakdown: [
    {
      id: 1,
      predicate: "has_chest_pain(patient) ∧ elevated_troponin(patient) → cardiac_event_likely(patient)",
      verified: true,
      confidence: 0.92,
      reasoning: "Strong correlation between chest pain and elevated troponin indicates cardiac tissue damage.",
      medical_context: "Troponin is a highly specific biomarker for myocardial injury.",
      status: "VERIFIED" as const,
    },
    {
      id: 2,
      predicate: "st_elevation(ecg) ∧ cardiac_enzymes_elevated(patient) → stemi_diagnosis(patient)",
      verified: true,
      confidence: 0.89,
      reasoning: "ECG evidence combined with biomarkers confirms STEMI diagnosis.",
      medical_context: "STEMI requires both ECG changes and clinical/biochemical evidence.",
      status: "VERIFIED" as const,
    },
  ],
}

export default function ChatExplainabilityPage() {
  const [showDemoComponents, setShowDemoComponents] = useState(false)

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="bg-gradient-to-r from-cyan-800 via-cyan-700 to-emerald-600 text-white py-12">
        <div className="container mx-auto px-6">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl font-bold mb-6 text-balance">CortexMD Explainable AI Interface</h1>
            <p className="text-xl text-cyan-100 mb-8 text-pretty leading-relaxed">
              Experience transparent medical AI reasoning with First-Order Logic verification, confidence scoring, and
              step-by-step diagnostic explanations
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <div className="bg-white/20 backdrop-blur-sm px-6 py-3 rounded-full border border-white/30">
                <span className="font-semibold">FOL Verification</span>
              </div>
              <div className="bg-white/20 backdrop-blur-sm px-6 py-3 rounded-full border border-white/30">
                <span className="font-semibold">Confidence Metrics</span>
              </div>
              <div className="bg-white/20 backdrop-blur-sm px-6 py-3 rounded-full border border-white/30">
                <span className="font-semibold">Reasoning Analysis</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-12 max-w-7xl">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          <div className="xl:col-span-2">
            <div className="bg-white rounded-2xl shadow-xl border border-slate-200 overflow-hidden">
              <div className="p-6 border-b border-slate-200 bg-gradient-to-r from-cyan-800 to-emerald-600 text-white">
                <h2 className="text-2xl font-bold mb-2">Medical AI Diagnostic Assistant</h2>
                <p className="text-cyan-100 leading-relaxed">
                  Ask medical questions and receive detailed AI reasoning with transparent verification
                </p>
              </div>
              <div className="h-[650px]">
                <ChatInterface />
              </div>
            </div>
          </div>

          <div className="space-y-8">
            <div className="bg-white rounded-2xl shadow-xl border border-slate-200 p-8">
              <h3 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-3">
                <div className="w-8 h-8 bg-cyan-100 rounded-lg flex items-center justify-center">
                  <div className="w-4 h-4 bg-cyan-600 rounded"></div>
                </div>
                Explainability Features
              </h3>
              <div className="space-y-4">
                <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                  <div className="w-3 h-3 bg-emerald-500 rounded-full"></div>
                  <span className="text-slate-700 font-medium">Real-time confidence scoring</span>
                </div>
                <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                  <div className="w-3 h-3 bg-cyan-500 rounded-full"></div>
                  <span className="text-slate-700 font-medium">First-Order Logic verification</span>
                </div>
                <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span className="text-slate-700 font-medium">Step-by-step reasoning breakdown</span>
                </div>
                <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                  <div className="w-3 h-3 bg-indigo-500 rounded-full"></div>
                  <span className="text-slate-700 font-medium">Medical accuracy scoring</span>
                </div>
                <div className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                  <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                  <span className="text-slate-700 font-medium">Source reliability tracking</span>
                </div>
              </div>

              <div className="mt-8 pt-6 border-t border-slate-200">
                <button
                  onClick={() => setShowDemoComponents(!showDemoComponents)}
                  className="w-full px-6 py-3 bg-gradient-to-r from-cyan-600 to-emerald-600 text-white rounded-xl hover:from-cyan-700 hover:to-emerald-700 transition-all duration-200 font-semibold shadow-lg hover:shadow-xl"
                >
                  {showDemoComponents ? "Hide" : "Show"} Demo Components
                </button>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-xl border border-slate-200 p-8">
              <h3 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-3">
                <div className="w-8 h-8 bg-emerald-100 rounded-lg flex items-center justify-center">
                  <div className="w-4 h-4 bg-emerald-600 rounded"></div>
                </div>
                How to Use
              </h3>
              <div className="space-y-6">
                <div className="border-l-4 border-cyan-500 pl-4">
                  <h4 className="font-semibold text-slate-900 mb-2">1. Ask Medical Questions</h4>
                  <p className="text-slate-600 text-sm leading-relaxed">
                    "What could cause chest pain and shortness of breath?"
                  </p>
                </div>
                <div className="border-l-4 border-emerald-500 pl-4">
                  <h4 className="font-semibold text-slate-900 mb-2">2. View Confidence Metrics</h4>
                  <p className="text-slate-600 text-sm leading-relaxed">
                    Check AI confidence, FOL verification, and explainability scores
                  </p>
                </div>
                <div className="border-l-4 border-blue-500 pl-4">
                  <h4 className="font-semibold text-slate-900 mb-2">3. Explore Reasoning</h4>
                  <p className="text-slate-600 text-sm leading-relaxed">
                    Click on expandable sections to see step-by-step logic
                  </p>
                </div>
                <div className="border-l-4 border-indigo-500 pl-4">
                  <h4 className="font-semibold text-slate-900 mb-2">4. Verify Sources</h4>
                  <p className="text-slate-600 text-sm leading-relaxed">Review evidence sources and medical context</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-xl border border-slate-200 p-8">
              <h3 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-3">
                <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                  <div className="w-4 h-4 bg-blue-600 rounded"></div>
                </div>
                Sample Questions
              </h3>
              <div className="space-y-4">
                <div className="p-4 bg-gradient-to-r from-cyan-50 to-cyan-100 rounded-xl border border-cyan-200 hover:shadow-md transition-shadow cursor-pointer">
                  <p className="text-cyan-800 font-medium">"What are the symptoms of a heart attack?"</p>
                </div>
                <div className="p-4 bg-gradient-to-r from-emerald-50 to-emerald-100 rounded-xl border border-emerald-200 hover:shadow-md transition-shadow cursor-pointer">
                  <p className="text-emerald-800 font-medium">"How is diabetes diagnosed?"</p>
                </div>
                <div className="p-4 bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl border border-blue-200 hover:shadow-md transition-shadow cursor-pointer">
                  <p className="text-blue-800 font-medium">"What causes high blood pressure?"</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {showDemoComponents && (
          <div className="mt-12 space-y-8">
            <div className="bg-white rounded-2xl shadow-xl border border-slate-200 p-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-slate-900 mb-4">Explainability Components Demo</h2>
                <p className="text-slate-600 text-lg leading-relaxed max-w-3xl mx-auto">
                  These components demonstrate how AI reasoning and confidence are displayed to medical professionals,
                  ensuring transparency and trust in diagnostic assistance.
                </p>
              </div>

              <div className="space-y-12">
                <div className="border border-slate-200 rounded-xl p-6">
                  <h3 className="text-xl font-semibold text-slate-800 mb-6 flex items-center gap-3">
                    <div className="w-8 h-8 bg-emerald-100 rounded-lg flex items-center justify-center">
                      <div className="w-4 h-4 bg-emerald-600 rounded"></div>
                    </div>
                    Confidence Visualization
                  </h3>
                  <ConfidenceVisualization metrics={sampleMetrics} compact={false} showDetails={true} />
                </div>

                <div className="border border-slate-200 rounded-xl p-6">
                  <h3 className="text-xl font-semibold text-slate-800 mb-6 flex items-center gap-3">
                    <div className="w-8 h-8 bg-cyan-100 rounded-lg flex items-center justify-center">
                      <div className="w-4 h-4 bg-cyan-600 rounded"></div>
                    </div>
                    Step-by-Step Reasoning
                  </h3>
                  <ReasoningBreakdown
                    steps={sampleReasoningSteps}
                    overallConfidence={0.89}
                    verificationStatus="verified"
                    processingTime={2.4}
                    title="Cardiac Diagnosis Reasoning"
                  />
                </div>

                <div className="border border-slate-200 rounded-xl p-6">
                  <h3 className="text-xl font-semibold text-slate-800 mb-6 flex items-center gap-3">
                    <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                      <div className="w-4 h-4 bg-blue-600 rounded"></div>
                    </div>
                    Comprehensive Explanations
                  </h3>
                  <DetailedExplanations
                    explanations={sampleExplanations}
                    folVerification={sampleFOLVerification}
                    confidenceMetrics={sampleMetrics}
                    reasoningSteps={sampleReasoningSteps}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
