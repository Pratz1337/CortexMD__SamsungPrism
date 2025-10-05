"use client"

import { useState, useEffect } from "react"
import { ConcernDashboard } from "@/components/concern/ConcernDashboard"
import { PatientMonitoring } from "@/components/concern/PatientMonitoring"
import { AddNoteForm } from "@/components/concern/AddNoteForm"
import { AddVisitForm } from "@/components/concern/AddVisitForm"

export default function ConcernPage() {
  const [activeTab, setActiveTab] = useState<"dashboard" | "monitoring" | "add-note" | "add-visit">("dashboard")

  const renderContent = () => {
    switch (activeTab) {
      case "dashboard":
        return <ConcernDashboard />
      case "monitoring":
        return <PatientMonitoring />
      case "add-note":
        return <AddNoteForm />
      case "add-visit":
        return <AddVisitForm />
      default:
        return <ConcernDashboard />
    }
  }

  return (
    <div className="min-h-screen">
      <section className="bg-red-600 text-white py-12">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-4xl font-bold mb-2">CONCERN Early Warning System</h1>
          <p className="text-lg opacity-90">AI-Powered Patient Deterioration Detection</p>
        </div>
      </section>

      <div className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <nav className="flex space-x-1 bg-white p-1 rounded-lg border">
            {[
              { id: "dashboard", label: "Dashboard", icon: "📊" },
              { id: "monitoring", label: "Patient Monitoring", icon: "👥" },
              { id: "add-note", label: "Add Clinical Note", icon: "📝" },
              { id: "add-visit", label: "Add Patient Visit", icon: "🏥" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-md font-medium transition ${
                  activeTab === tab.id 
                    ? "bg-red-600 text-white" 
                    : "text-gray-600 hover:bg-gray-100"
                }`}
              >
                <span>{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>

        <div>{renderContent()}</div>
      </div>
    </div>
  )
}
