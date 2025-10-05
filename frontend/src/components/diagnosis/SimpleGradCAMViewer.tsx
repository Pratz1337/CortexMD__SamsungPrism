import { PhotoIcon, FireIcon } from "@heroicons/react/24/outline"

interface GradCAMViewerProps {
  heatmapData: any[]
  imagePaths: string[]
}

export default function SimpleGradCAMViewer({ heatmapData, imagePaths }: GradCAMViewerProps) {
  if (!heatmapData || heatmapData.length === 0) {
    return (
      <div className="w-full h-64 flex items-center justify-center text-blue-400 bg-blue-50 rounded-lg border border-blue-200">
        <div className="text-center">
          <FireIcon className="w-12 h-12 mx-auto mb-2" />
          <p className="font-medium">GradCAM Analysis</p>
          <p className="text-sm">Heatmap visualization will appear here</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {heatmapData.map((heatmap, index) => (
        <div key={index} className="bg-white rounded-lg border border-blue-200 overflow-hidden">
          <div className="p-3 bg-blue-50 border-b border-blue-200">
            <h4 className="font-semibold text-blue-900 flex items-center">
              <FireIcon className="w-4 h-4 mr-2" />
              GradCAM Heatmap {index + 1}
            </h4>
          </div>

          <div className="p-4">
            {heatmap.visualizations?.heatmap_image || heatmap.visualizations?.overlay_image || heatmap.visualizations?.volume_image || heatmap.gradcam_url ? (
              <div className="relative">
                <img
                  src={heatmap.visualizations?.heatmap_image ? 
                    `data:image/png;base64,${heatmap.visualizations.heatmap_image}` : 
                    heatmap.visualizations?.overlay_image ? 
                    `data:image/png;base64,${heatmap.visualizations.overlay_image}` :
                    heatmap.visualizations?.volume_image ? 
                    `data:image/png;base64,${heatmap.visualizations.volume_image}` :
                    heatmap.gradcam_url || "/placeholder.svg"}
                  alt={`GradCAM heatmap ${index + 1}`}
                  className="w-full h-64 object-contain rounded-lg bg-gray-50"
                />
                <div className="absolute top-2 right-2 bg-blue-600 text-white px-2 py-1 rounded text-xs font-semibold">
                  AI Focus Areas
                </div>
              </div>
            ) : (
              <div className="w-full h-64 flex items-center justify-center text-gray-400 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200">
                <div className="text-center">
                  <PhotoIcon className="w-12 h-12 mx-auto mb-2" />
                  <p>GradCAM image not available</p>
                </div>
              </div>
            )}

            {heatmap.explanation && (
              <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <p className="text-sm text-blue-800">{heatmap.explanation}</p>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}
