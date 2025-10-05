"use client"

import React, { useState } from 'react'
import { Dialog, Transition } from '@headlessui/react'
import { Fragment } from 'react'
import { 
  XMarkIcon, 
  DocumentArrowDownIcon, 
  DocumentTextIcon,
  PrinterIcon,
  GlobeAltIcon,
  DocumentIcon,
  CheckCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { DiagnosisResult } from '@/types'
import { generatePDFReport, generateHTMLReport, generateWordReport, generateMedicalReport, ReportOptions } from '@/utils/reportGenerator'
import toast from 'react-hot-toast'

interface Props {
  isOpen: boolean
  onClose: () => void
  diagnosisData: DiagnosisResult
}

interface FormatOption {
  id: string
  name: string
  description: string
  icon: React.ElementType
  extension: string
  size: string
  recommended?: boolean
}

interface TemplateOption {
  id: string
  name: string
  description: string
  features: string[]
}

const formatOptions: FormatOption[] = [
  {
    id: 'pdf',
    name: 'PDF Report',
    description: 'Professional PDF document with medical formatting',
    icon: DocumentTextIcon,
    extension: 'PDF',
    size: '~200-500 KB',
    recommended: true
  },
  {
    id: 'html',
    name: 'HTML Report',
    description: 'Web page format for viewing and sharing online',
    icon: GlobeAltIcon,
    extension: 'HTML',
    size: '~50-100 KB'
  },
  {
    id: 'docx',
    name: 'Word Document',
    description: 'Microsoft Word format for editing and collaboration',
    icon: DocumentIcon,
    extension: 'DOCX',
    size: '~100-200 KB'
  },
  {
    id: 'print',
    name: 'Print Preview',
    description: 'Direct printing with optimized layout',
    icon: PrinterIcon,
    extension: 'PRINT',
    size: 'Browser Print'
  }
]

const templateOptions: TemplateOption[] = [
  {
    id: 'standard',
    name: 'Standard Report',
    description: 'Essential diagnosis information with clean formatting',
    features: ['Primary diagnosis', 'Differential diagnoses', 'Recommendations', 'Patient info']
  },
  {
    id: 'detailed',
    name: 'Detailed Report',
    description: 'Comprehensive report with AI verification details',
    features: ['All standard features', 'FOL verification', 'Enhanced verification', 'Online verification', 'Source citations']
  },
  {
    id: 'summary',
    name: 'Executive Summary',
    description: 'Concise report focusing on key findings',
    features: ['Primary diagnosis only', 'Key recommendations', 'Confidence scores', 'Urgency level']
  }
]

export default function ReportDownloadModal({ isOpen, onClose, diagnosisData }: Props) {
  const [selectedFormat, setSelectedFormat] = useState<string>('pdf')
  const [selectedTemplate, setSelectedTemplate] = useState<string>('standard')
  const [isGenerating, setIsGenerating] = useState(false)
  const [hospitalInfo, setHospitalInfo] = useState({
    name: 'CortexMD Medical AI System',
    address: 'AI-Powered Medical Diagnosis Platform',
    phone: '1-800-CORTEX'
  })
  const [physicianInfo, setPhysicianInfo] = useState({
    name: 'Dr. AI Assistant',
    title: 'AI Medical Specialist',
    license: 'AI-LICENSE-001'
  })
  const [includeCharts, setIncludeCharts] = useState(true)
  const [includeDiagnosticImages, setIncludeDiagnosticImages] = useState(true)

  const generateReport = async () => {
    setIsGenerating(true)
    
    try {
      const options: ReportOptions = {
        format: selectedFormat as any,
        template: selectedTemplate as any,
        includeCharts,
        includeDiagnosticImages,
        hospitalInfo,
        physicianInfo
      }

      await generateMedicalReport(diagnosisData, options)
      
      const formatName = formatOptions.find(f => f.id === selectedFormat)?.name || 'Report'
      toast.success(`${formatName} generated successfully!`)
      onClose()
    } catch (error: any) {
      console.error('Report generation error:', error)
      toast.error(`Failed to generate report: ${error.message}`)
    } finally {
      setIsGenerating(false)
    }
  }

  const selectedFormatOption = formatOptions.find(f => f.id === selectedFormat)
  const selectedTemplateOption = templateOptions.find(t => t.id === selectedTemplate)

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black bg-opacity-25 backdrop-blur-sm" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4 text-center">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <Dialog.Panel className="w-full max-w-4xl transform overflow-hidden rounded-2xl bg-white p-6 text-left align-middle shadow-xl transition-all">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <DocumentArrowDownIcon className="w-6 h-6 text-blue-600" />
                    </div>
                    <div>
                      <Dialog.Title as="h3" className="text-xl font-bold text-gray-900">
                        Download Medical Report
                      </Dialog.Title>
                      <p className="text-sm text-gray-600">
                        Generate professional medical reports in multiple formats
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={onClose}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    <XMarkIcon className="w-5 h-5 text-gray-500" />
                  </button>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Left Column - Options */}
                  <div className="lg:col-span-2 space-y-6">
                    {/* Format Selection */}
                    <div>
                      <h4 className="text-lg font-semibold text-gray-900 mb-3">Select Format</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {formatOptions.map((format) => {
                          const IconComponent = format.icon
                          return (
                            <div
                              key={format.id}
                              className={`relative p-4 border-2 rounded-lg cursor-pointer transition-all ${
                                selectedFormat === format.id
                                  ? 'border-blue-500 bg-blue-50'
                                  : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                              }`}
                              onClick={() => setSelectedFormat(format.id)}
                            >
                              {format.recommended && (
                                <div className="absolute -top-2 -right-2 bg-green-500 text-white text-xs px-2 py-1 rounded-full">
                                  Recommended
                                </div>
                              )}
                              <div className="flex items-start space-x-3">
                                <IconComponent className="w-6 h-6 text-gray-700 mt-0.5" />
                                <div className="flex-1">
                                  <h5 className="font-medium text-gray-900">{format.name}</h5>
                                  <p className="text-sm text-gray-600 mb-2">{format.description}</p>
                                  <div className="flex items-center space-x-2 text-xs text-gray-500">
                                    <span className="bg-gray-100 px-2 py-1 rounded">{format.extension}</span>
                                    <span>{format.size}</span>
                                  </div>
                                </div>
                                {selectedFormat === format.id && (
                                  <CheckCircleIcon className="w-5 h-5 text-blue-600" />
                                )}
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>

                    {/* Template Selection */}
                    <div>
                      <h4 className="text-lg font-semibold text-gray-900 mb-3">Report Template</h4>
                      <div className="space-y-3">
                        {templateOptions.map((template) => (
                          <div
                            key={template.id}
                            className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                              selectedTemplate === template.id
                                ? 'border-blue-500 bg-blue-50'
                                : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                            }`}
                            onClick={() => setSelectedTemplate(template.id)}
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <h5 className="font-medium text-gray-900 mb-1">{template.name}</h5>
                                <p className="text-sm text-gray-600 mb-3">{template.description}</p>
                                <div className="flex flex-wrap gap-1">
                                  {template.features.map((feature, index) => (
                                    <span
                                      key={index}
                                      className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded"
                                    >
                                      {feature}
                                    </span>
                                  ))}
                                </div>
                              </div>
                              {selectedTemplate === template.id && (
                                <CheckCircleIcon className="w-5 h-5 text-blue-600 ml-3" />
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Additional Options */}
                    <div>
                      <h4 className="text-lg font-semibold text-gray-900 mb-3">Additional Options</h4>
                      <div className="space-y-3">
                        <label className="flex items-center space-x-3">
                          <input
                            type="checkbox"
                            checked={includeCharts}
                            onChange={(e) => setIncludeCharts(e.target.checked)}
                            className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                          />
                          <span className="text-sm text-gray-700">Include charts and visualizations</span>
                        </label>
                        <label className="flex items-center space-x-3">
                          <input
                            type="checkbox"
                            checked={includeDiagnosticImages}
                            onChange={(e) => setIncludeDiagnosticImages(e.target.checked)}
                            className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                          />
                          <span className="text-sm text-gray-700">Include diagnostic images</span>
                        </label>
                      </div>
                    </div>

                    {/* Hospital Information */}
                    <div>
                      <h4 className="text-lg font-semibold text-gray-900 mb-3">Hospital Information</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Hospital Name
                          </label>
                          <input
                            type="text"
                            value={hospitalInfo.name}
                            onChange={(e) => setHospitalInfo(prev => ({ ...prev, name: e.target.value }))}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Address
                          </label>
                          <input
                            type="text"
                            value={hospitalInfo.address}
                            onChange={(e) => setHospitalInfo(prev => ({ ...prev, address: e.target.value }))}
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Right Column - Preview & Summary */}
                  <div className="space-y-6">
                    {/* Report Preview */}
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h4 className="text-lg font-semibold text-gray-900 mb-3">Report Preview</h4>
                      
                      {/* Selected Options Summary */}
                      <div className="space-y-3">
                        <div className="flex items-center justify-between p-3 bg-white rounded border">
                          <div className="flex items-center space-x-2">
                            {selectedFormatOption && <selectedFormatOption.icon className="w-4 h-4 text-gray-600" />}
                            <span className="text-sm font-medium">{selectedFormatOption?.name}</span>
                          </div>
                          <span className="text-xs text-gray-500">{selectedFormatOption?.extension}</span>
                        </div>

                        <div className="p-3 bg-white rounded border">
                          <div className="text-sm font-medium mb-1">{selectedTemplateOption?.name}</div>
                          <div className="text-xs text-gray-600">{selectedTemplateOption?.description}</div>
                        </div>
                      </div>

                      {/* Report Stats */}
                      <div className="mt-4 p-3 bg-white rounded border">
                        <h5 className="text-sm font-medium text-gray-900 mb-2">Report Statistics</h5>
                        <div className="space-y-2 text-xs text-gray-600">
                          <div className="flex justify-between">
                            <span>Session ID:</span>
                            <span className="font-mono">{diagnosisData.session_id.substring(0, 8)}...</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Diagnosis:</span>
                            <span>{diagnosisData.primary_diagnosis.condition.substring(0, 20)}...</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Confidence:</span>
                            <span>{(diagnosisData.primary_diagnosis.confidence * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Processing Time:</span>
                            <span>{diagnosisData.processing_time?.toFixed(2) || 'N/A'}s</span>
                          </div>
                        </div>
                      </div>

                      {/* Features Included */}
                      <div className="mt-4 p-3 bg-white rounded border">
                        <h5 className="text-sm font-medium text-gray-900 mb-2">Included Features</h5>
                        <div className="space-y-1">
                          {selectedTemplateOption?.features.map((feature, index) => (
                            <div key={index} className="flex items-center space-x-2 text-xs text-gray-600">
                              <CheckCircleIcon className="w-3 h-3 text-green-500" />
                              <span>{feature}</span>
                            </div>
                          ))}
                          {includeCharts && (
                            <div className="flex items-center space-x-2 text-xs text-gray-600">
                              <CheckCircleIcon className="w-3 h-3 text-green-500" />
                              <span>Charts & visualizations</span>
                            </div>
                          )}
                          {includeDiagnosticImages && (
                            <div className="flex items-center space-x-2 text-xs text-gray-600">
                              <CheckCircleIcon className="w-3 h-3 text-green-500" />
                              <span>Diagnostic images</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Medical Disclaimer */}
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                      <div className="flex items-start space-x-2">
                        <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 mt-0.5" />
                        <div>
                          <h5 className="text-sm font-medium text-yellow-800 mb-1">Medical Disclaimer</h5>
                          <p className="text-xs text-yellow-700">
                            This AI-generated report should be reviewed by a qualified medical professional 
                            before clinical use. It is not intended to replace professional medical advice.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Footer */}
                <div className="mt-6 pt-6 border-t border-gray-200">
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-gray-600">
                      <ClockIcon className="w-4 h-4 inline mr-1" />
                      Estimated generation time: 3-10 seconds
                    </div>
                    <div className="flex space-x-3">
                      <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={generateReport}
                        disabled={isGenerating}
                        className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
                      >
                        {isGenerating ? (
                          <>
                            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                            <span>Generating...</span>
                          </>
                        ) : (
                          <>
                            <DocumentArrowDownIcon className="w-4 h-4" />
                            <span>Generate Report</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  )
}
