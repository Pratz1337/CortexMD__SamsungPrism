#!/usr/bin/env python3
"""
Quick integration script to add 3D GradCAM to your existing CortexMD app.
This shows exactly how to modify your current app.py to include 3D GradCAM.
"""

# Add this to your existing app.py imports
try:
    from ..medical_processing.integration_3d_gradcam import integrate_3d_gradcam_with_diagnosis, create_heatmap_api_response
except ImportError:
    from medical_processing.integration_3d_gradcam import integrate_3d_gradcam_with_diagnosis, create_heatmap_api_response

def integrate_gradcam_with_existing_app():
    """
    Integration code to add to your existing Flask app
    """
    
    integration_code = '''
# =============================================================================
# ADD THIS TO YOUR EXISTING app.py FILE
# =============================================================================

# 1. ADD THESE IMPORTS AT THE TOP
from medical_processing.integration_3d_gradcam import integrate_3d_gradcam_with_diagnosis, create_heatmap_api_response

# 2. MODIFY YOUR EXISTING /submit ROUTE
@app.route('/submit', methods=['POST'])
def submit_diagnosis():
    """Your existing diagnosis submission route - ENHANCED with 3D GradCAM"""
    
    # ... your existing code for handling form data ...
    
    # ADD THIS SECTION after file upload handling:
    uploaded_file_paths = []  # List of uploaded image file paths
    
    # Your existing file handling code should populate this list
    # For example:
    # for file in request.files.getlist('files'):
    #     filename = secure_filename(file.filename)
    #     file_path = os.path.join('uploads', filename)
    #     file.save(file_path)
    #     uploaded_file_paths.append(file_path)
    
    # NEW: Generate 3D GradCAM for uploaded images
    heatmap_results = None
    if uploaded_file_paths:
        try:
            print("🔥 Generating 3D GradCAM heatmaps...")
            heatmap_results = integrate_3d_gradcam_with_diagnosis(
                image_files=uploaded_file_paths,
                model_path="3d_image_classification.h5",  # Your model
                output_dir=f"uploads/heatmaps_{session_id}"
            )
            print(f"✅ GradCAM completed: {heatmap_results['successful_heatmaps']} heatmaps generated")
        except Exception as e:
            print(f"❌ GradCAM generation failed: {e}")
            heatmap_results = {'success': False, 'error': str(e), 'heatmap_data': []}
    
    # Store heatmap results in session data
    session_data = {
        # ... your existing session data ...
        'heatmap_results': heatmap_results,  # NEW: Add this line
        'image_paths': uploaded_file_paths   # Make sure this exists
    }
    
    # ... rest of your existing submit logic ...

# 3. MODIFY YOUR EXISTING /results/<session_id> ROUTE  
@app.route('/results/<session_id>')
def get_results(session_id):
    """Your existing results route - ENHANCED with 3D GradCAM data"""
    
    # ... your existing results retrieval logic ...
    
    # NEW: Add heatmap data to response
    response_data = {
        # ... your existing response data ...
    }
    
    # Add 3D GradCAM results if available
    if 'heatmap_results' in session_data and session_data['heatmap_results']:
        heatmap_response = create_heatmap_api_response(session_data['heatmap_results'])
        
        # Add to your existing response
        response_data.update({
            'heatmap_visualization': heatmap_response.get('heatmap_visualization', {}),
            'heatmap_data': heatmap_response.get('heatmap_data', []),
            'gradcam_available': True  # Flag for frontend
        })
    else:
        response_data.update({
            'heatmap_visualization': {'available': False},
            'heatmap_data': [],
            'gradcam_available': False
        })
    
    return jsonify(response_data)

# 4. NEW: Add dedicated GradCAM API endpoint (optional)
@app.route('/api/gradcam/generate', methods=['POST'])
def generate_gradcam_api():
    """Dedicated API endpoint for 3D GradCAM generation"""
    
    try:
        # Get uploaded files
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        # Save files temporarily
        temp_files = []
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join('uploads/temp', filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                file.save(file_path)
                temp_files.append(file_path)
        
        # Generate GradCAM
        heatmap_results = integrate_3d_gradcam_with_diagnosis(
            image_files=temp_files,
            model_path="3d_image_classification.h5"
        )
        
        # Create API response
        api_response = create_heatmap_api_response(heatmap_results)
        
        # Cleanup
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        return jsonify(api_response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 5. NEW: Serve generated heatmap images
@app.route('/api/heatmap/image/<path:image_path>')
def serve_heatmap_image(image_path):
    """Serve generated heatmap images"""
    try:
        # Security check - only serve from heatmap directories
        if 'heatmap' not in image_path:
            abort(404)
        
        return send_file(image_path, as_attachment=False)
    except:
        abort(404)
'''

    return integration_code

def test_integration():
    """Test the integration with a real example"""
    
    print("🧪 Testing 3D GradCAM Integration")
    print("=" * 50)
    
    # Simulate what happens in your app
    uploaded_files = ["uploads/PATIENT_001_20250905_134726_sarcoma.JPG"]
    
    print("📤 Simulating file upload...")
    print(f"   Files: {uploaded_files}")
    
    # Test the integration function
    print("\n🔥 Testing GradCAM integration...")
    results = integrate_3d_gradcam_with_diagnosis(
        image_files=uploaded_files,
        model_path="3d_image_classification.h5"
    )
    
    print(f"✅ Integration test results:")
    print(f"   Success: {results['success']}")
    print(f"   Total images: {results['total_images']}")
    print(f"   Successful heatmaps: {results['successful_heatmaps']}")
    
    # Test API response creation
    api_response = create_heatmap_api_response(results)
    print(f"\n📡 API Response created:")
    print(f"   Success: {api_response['success']}")
    print(f"   Heatmap available: {api_response['heatmap_visualization']['available']}")
    print(f"   Number of heatmap data entries: {len(api_response['heatmap_data'])}")
    
    if api_response['heatmap_data']:
        first_heatmap = api_response['heatmap_data'][0]
        if first_heatmap['success']:
            print(f"   First heatmap prediction: {first_heatmap['analysis']['predicted_class']}")
            print(f"   First heatmap confidence: {first_heatmap['analysis']['confidence_score']:.2%}")
            print(f"   Medical interpretation: {first_heatmap['medical_interpretation']['primary_finding']}")
    
    return results, api_response

def show_frontend_integration():
    """Show how to integrate with your frontend"""
    
    frontend_code = '''
<!-- ADD THIS TO YOUR FRONTEND (e.g., DiagnosisResults.tsx) -->

// 1. Update your results interface to include heatmap data
interface DiagnosisResult {
  // ... your existing fields ...
  heatmap_visualization?: {
    available: boolean;
    total_images: number;
    successful_heatmaps: number;
    model_type: string;
  };
  heatmap_data?: Array<{
    success: boolean;
    image_file: string;
    analysis?: {
      predicted_class: string;
      confidence_score: number;
      processing_time: number;
    };
    visualizations?: {
      heatmap_image: string;  // base64 encoded
      overlay_image: string;  // base64 encoded
      volume_image: string;   // base64 encoded
    };
    medical_interpretation?: {
      primary_finding: string;
      confidence_level: string;
      clinical_notes: string[];
    };
  }>;
  gradcam_available?: boolean;
}

// 2. Add GradCAM visualization component
const GradCAMVisualization = ({ heatmapData }: { heatmapData: any }) => {
  if (!heatmapData || !heatmapData.success) return null;

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center">
        🔥 AI Attention Heatmap
        <span className="ml-2 text-sm bg-green-100 text-green-800 px-2 py-1 rounded">
          {heatmapData.analysis.predicted_class} ({(heatmapData.analysis.confidence_score * 100).toFixed(1)}%)
        </span>
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Heatmap */}
        <div>
          <h4 className="font-medium mb-2">Model Attention</h4>
          <img 
            src={`data:image/png;base64,${heatmapData.visualizations.heatmap_image}`}
            alt="GradCAM Heatmap" 
            className="w-full rounded border"
          />
        </div>
        
        {/* Overlay */}
        <div>
          <h4 className="font-medium mb-2">Overlay View</h4>
          <img 
            src={`data:image/png;base64,${heatmapData.visualizations.overlay_image}`}
            alt="Overlay Visualization" 
            className="w-full rounded border"
          />
        </div>
        
        {/* Volume */}
        <div>
          <h4 className="font-medium mb-2">3D Volume View</h4>
          <img 
            src={`data:image/png;base64,${heatmapData.visualizations.volume_image}`}
            alt="3D Volume Visualization" 
            className="w-full rounded border"
          />
        </div>
      </div>
      
      {/* Medical Interpretation */}
      <div className="mt-4 p-4 bg-blue-50 rounded">
        <h4 className="font-medium text-blue-800 mb-2">Medical Interpretation</h4>
        <p className="text-sm text-blue-700">
          <strong>Finding:</strong> {heatmapData.medical_interpretation.primary_finding}
        </p>
        <p className="text-sm text-blue-700">
          <strong>Confidence:</strong> {heatmapData.medical_interpretation.confidence_level}
        </p>
        <div className="mt-2">
          <strong className="text-blue-800">Clinical Notes:</strong>
          <ul className="list-disc list-inside text-sm text-blue-700 mt-1">
            {heatmapData.medical_interpretation.clinical_notes.map((note, idx) => (
              <li key={idx}>{note}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

// 3. Use in your main results component
const DiagnosisResults = ({ sessionId }: { sessionId: string }) => {
  const [results, setResults] = useState<DiagnosisResult | null>(null);
  
  // Your existing useEffect to fetch results...
  
  return (
    <div>
      {/* Your existing results display */}
      
      {/* NEW: Add GradCAM visualizations */}
      {results?.gradcam_available && results.heatmap_data && (
        <div className="mt-8">
          <h2 className="text-xl font-bold mb-4">🧠 AI Explainability</h2>
          {results.heatmap_data.map((heatmapData, index) => (
            <GradCAMVisualization key={index} heatmapData={heatmapData} />
          ))}
        </div>
      )}
    </div>
  );
};
'''

    print("🎨 Frontend Integration Code:")
    print("=" * 50)
    print(frontend_code)

if __name__ == "__main__":
    print("🔧 CortexMD 3D GradCAM Integration Helper")
    print("=" * 60)
    
    # Show integration code
    print("\n1. 📝 Backend Integration Code:")
    integration_code = integrate_gradcam_with_existing_app()
    print(integration_code)
    
    # Test the integration
    print("\n2. 🧪 Testing Integration:")
    try:
        results, api_response = test_integration()
        print("✅ Integration test PASSED!")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
    
    # Show frontend code
    print("\n3. 🎨 Frontend Integration:")
    show_frontend_integration()
    
    print("\n" + "=" * 60)
    print("🎉 Integration Complete!")
    print("Your 3D GradCAM is ready for production use!")




