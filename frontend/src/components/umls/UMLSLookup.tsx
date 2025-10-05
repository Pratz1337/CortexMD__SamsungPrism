'use client';

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { MagnifyingGlassIcon, DocumentArrowUpIcon, DocumentArrowDownIcon } from '@heroicons/react/24/outline';
import { api } from '@/lib/api';
import toast from 'react-hot-toast';

interface UMLSConcept {
  cui: string;
  preferred_name: string;
  synonyms: string[];
  semantic_types: string[];
  definitions: string[];
  sources: string[];
}

interface UMLSSearchResult {
  concepts: UMLSConcept[];
  total_results: number;
  search_term: string;
  normalized_term?: string;
}

interface ConceptDetails {
  cui: string;
  preferred_name: string;
  synonyms: string[];
  semantic_types: string[];
  definitions: string[];
  sources: string[];
  relationships: any[];
}

export function UMLSLookup() {
  const [searchResults, setSearchResults] = useState<UMLSSearchResult | null>(null);
  const [selectedConcept, setSelectedConcept] = useState<ConceptDetails | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'search' | 'batch' | 'details'>('search');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const { register, handleSubmit, formState: { errors } } = useForm();

  const handleSearch = async (data: any) => {
    setIsLoading(true);
    try {
      const response = await api.post('/api/umls/search', {
        query: data.searchTerm,
        search_type: data.searchType || 'exact',
        max_results: data.maxResults || 20
      });
      setSearchResults(response.data);
      setActiveTab('search');
      toast.success(`Found ${response.data.total_results} results`);
    } catch (error) {
      toast.error('Search failed. Please try again.');
      console.error('Search error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleConceptClick = async (cui: string) => {
    setIsLoading(true);
    try {
      const response = await api.get(`/api/umls/concept-details/${cui}`);
      setSelectedConcept(response.data);
      setActiveTab('details');
    } catch (error) {
      toast.error('Failed to load concept details');
      console.error('Concept details error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await api.post('/api/umls/lookup-codes-file', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // Handle the response which should contain download link
      if (response.data.download_url) {
        window.open(response.data.download_url, '_blank');
        toast.success('File processed successfully!');
      }
    } catch (error) {
      toast.error('File upload failed');
      console.error('Upload error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
    }
  };

  const downloadResults = async () => {
    if (!searchResults) return;
    
    try {
      const response = await api.post('/download-umls-results/search_results.json', {
        results: searchResults
      }, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'umls_search_results.json');
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      toast.success('Results downloaded successfully!');
    } catch (error) {
      toast.error('Download failed');
      console.error('Download error:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="card-header">
        <h1 className="text-3xl font-bold">UMLS Code Lookup</h1>
        <p className="opacity-90 mt-2">
          Search and explore the Unified Medical Language System (UMLS) for medical terminology and concepts
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="card p-1">
        <nav className="flex space-x-1">
          {[
            { id: 'search', label: 'Search', icon: MagnifyingGlassIcon },
            { id: 'batch', label: 'Batch Lookup', icon: DocumentArrowUpIcon },
            { id: 'details', label: 'Concept Details', icon: DocumentArrowDownIcon },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-md font-medium transition-all duration-200 ${
                activeTab === tab.id
                  ? 'bg-gradient-medical text-white shadow-lg'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <tab.icon className="w-5 h-5" />
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Search Tab */}
      {activeTab === 'search' && (
        <div className="space-y-6">
          {/* Search Form */}
          <div className="card p-6">
            <h2 className="text-xl font-semibold mb-4">Search UMLS Concepts</h2>
            <form onSubmit={handleSubmit(handleSearch)} className="space-y-4">
              <div className="grid md:grid-cols-3 gap-4">
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Search Term *
                  </label>
                  <input
                    type="text"
                    {...register('searchTerm', { required: 'Search term is required' })}
                    className="medical-input w-full"
                    placeholder="Enter medical term, concept, or code..."
                  />
                  {errors.searchTerm && (
                    <p className="text-red-500 text-sm mt-1">{String(errors.searchTerm?.message)}</p>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Search Type
                  </label>
                  <select {...register('searchType')} className="medical-input w-full">
                    <option value="exact">Exact Match</option>
                    <option value="contains">Contains</option>
                    <option value="starts_with">Starts With</option>
                    <option value="fuzzy">Fuzzy Search</option>
                  </select>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Max Results
                  </label>
                  <select {...register('maxResults')} className="medical-input w-full">
                    <option value="10">10 results</option>
                    <option value="20">20 results</option>
                    <option value="50">50 results</option>
                    <option value="100">100 results</option>
                  </select>
                </div>
                <div className="flex items-end">
                  <button
                    type="submit"
                    disabled={isLoading}
                    className="btn-primary flex items-center space-x-2 disabled:opacity-50"
                  >
                    {isLoading ? (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    ) : (
                      <MagnifyingGlassIcon className="w-5 h-5" />
                    )}
                    <span>{isLoading ? 'Searching...' : 'Search'}</span>
                  </button>
                </div>
              </div>
            </form>
          </div>

          {/* Search Results */}
          {searchResults && (
            <div className="card p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">
                  Search Results ({searchResults.total_results} found)
                </h3>
                <button
                  onClick={downloadResults}
                  className="btn-secondary flex items-center space-x-2"
                >
                  <DocumentArrowDownIcon className="w-5 h-5" />
                  <span>Download Results</span>
                </button>
              </div>

              {searchResults.normalized_term && (
                <div className="mb-4 p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-800">
                    <strong>Normalized term:</strong> {searchResults.normalized_term}
                  </p>
                </div>
              )}

              <div className="space-y-3 max-h-96 overflow-y-auto">
                {searchResults.concepts?.length > 0 ? (
                  searchResults.concepts.map((concept) => (
                  <div
                    key={concept.cui}
                    onClick={() => handleConceptClick(concept.cui)}
                    className="result-item cursor-pointer p-4 border rounded-lg hover:shadow-md transition-all duration-200 hover:translate-x-1 border-l-4 border-l-primary-500"
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <h4 className="font-semibold text-primary-600 mb-2">
                          {concept.preferred_name}
                        </h4>
                        <p className="text-sm text-gray-600 mb-2">
                          <strong>CUI:</strong> {concept.cui}
                        </p>
                        {concept.synonyms.length > 0 && (
                          <p className="text-sm text-gray-600 mb-2">
                            <strong>Synonyms:</strong> {concept.synonyms.slice(0, 3).join(', ')}
                            {concept.synonyms.length > 3 && ` (+${concept.synonyms.length - 3} more)`}
                          </p>
                        )}
                        <div className="flex flex-wrap gap-1 mt-2">
                          {concept.semantic_types.slice(0, 3).map((type, index) => (
                            <span
                              key={index}
                              className="px-2 py-1 bg-primary-100 text-primary-700 text-xs rounded-full"
                            >
                              {type}
                            </span>
                          ))}
                          {concept.semantic_types.length > 3 && (
                            <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
                              +{concept.semantic_types.length - 3} more
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <p>No concepts found for this search term.</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Batch Lookup Tab */}
      {activeTab === 'batch' && (
        <div className="card p-6">
          <h2 className="text-xl font-semibold mb-4">Batch Code Lookup</h2>
          <p className="text-gray-600 mb-6">
            Upload a file containing medical terms or codes for batch processing. 
            Supported formats: .txt, .csv, .json
          </p>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload File
              </label>
              <input
                type="file"
                accept=".txt,.csv,.json"
                onChange={handleFileChange}
                className="medical-input w-full"
              />
            </div>

            {uploadedFile && (
              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="text-sm">
                  <strong>Selected file:</strong> {uploadedFile.name}
                </p>
                <p className="text-sm text-gray-600">
                  Size: {(uploadedFile.size / 1024).toFixed(2)} KB
                </p>
              </div>
            )}

            <button
              onClick={() => uploadedFile && handleFileUpload(uploadedFile)}
              disabled={!uploadedFile || isLoading}
              className="btn-primary disabled:opacity-50"
            >
              {isLoading ? 'Processing...' : 'Process File'}
            </button>
          </div>

          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <h3 className="font-semibold text-blue-800 mb-2">File Format Guidelines:</h3>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• <strong>TXT:</strong> One term per line</li>
              <li>• <strong>CSV:</strong> Terms in first column, optional headers</li>
              <li>• <strong>JSON:</strong> Array of terms or objects with 'term' field</li>
            </ul>
          </div>
        </div>
      )}

      {/* Concept Details Tab */}
      {activeTab === 'details' && (
        <div className="card p-6">
          {selectedConcept ? (
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-primary-600 mb-2">
                  {selectedConcept.preferred_name}
                </h2>
                <p className="text-gray-600">
                  <strong>CUI:</strong> {selectedConcept.cui}
                </p>
              </div>

              {selectedConcept.definitions.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Definitions</h3>
                  <div className="space-y-2">
                    {selectedConcept.definitions.map((def, index) => (
                      <div key={index} className="p-3 bg-gray-50 rounded-lg">
                        <p className="text-sm">{def}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {selectedConcept.synonyms.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Synonyms</h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedConcept.synonyms.map((synonym, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-primary-100 text-primary-700 text-sm rounded-full"
                      >
                        {synonym}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <h3 className="text-lg font-semibold mb-3">Semantic Types</h3>
                <div className="flex flex-wrap gap-2">
                  {selectedConcept.semantic_types.map((type, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-secondary-100 text-secondary-700 text-sm rounded-full"
                    >
                      {type}
                    </span>
                  ))}
                </div>
              </div>

              {selectedConcept.sources.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">Sources</h3>
                  <div className="grid md:grid-cols-2 gap-2">
                    {selectedConcept.sources.map((source, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-gray-100 text-gray-700 text-sm rounded"
                      >
                        {source}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-12">
              <MagnifyingGlassIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-600 mb-2">
                No Concept Selected
              </h3>
              <p className="text-gray-500">
                Search for concepts and click on a result to view detailed information
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
