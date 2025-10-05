/**
 * Enhanced Medical Knowledge Search Component
 * Provides intelligent search across UMLS, Neo4j Knowledge Graph, and Medical Ontologies
 */

'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Search, Brain, Network, BookOpen, Zap, AlertCircle, CheckCircle, Info, ExternalLink, ArrowRight, Filter, Download, Share, Heart, Activity, Pill, TestTube } from 'lucide-react';
import Logo from '@/components/ui/Logo'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

import { ScrollArea } from '@/components/ui/scroll-area';

interface MedicalConcept {
  cui: string;
  preferred_name: string;
  synonyms: string[];
  definitions: string[];
  semantic_types: string[];
  source_vocabularies: string[];
  confidence: number;
  relationships?: ConceptRelationship[];
  hierarchy_level?: number;
  clinical_relevance_score?: number;
}

interface ConceptRelationship {
  source_cui: string;
  target_cui: string;
  relationship_type: string;
  strength: number;
  source: string;
  additional_info?: Record<string, any>;
}

interface SearchResult {
  query: string;
  concepts: MedicalConcept[];
  relationships: ConceptRelationship[];
  concept_hierarchy: Record<string, string[]>;
  similar_concepts: MedicalConcept[];
  clinical_context: {
    primary_semantic_types: Array<{ type: string; count: number }>;
    clinical_domains: string[];
    relationship_summary: Record<string, number>;
    clinical_significance: string;
  };
  search_metadata: {
    enhanced_query: string;
    search_type: string;
    sources_used: string[];
    query_processing_time: number;
  };
  execution_time: number;
  total_results: number;
}

interface AutocompleteSuggestion {
  text: string;
  cui: string;
  semantic_types: string[];
  confidence: number;
  is_synonym?: boolean;
}

const SEMANTIC_TYPE_ICONS = {
  'Disease or Syndrome': Heart,
  'Sign or Symptom': Activity,
  'Pharmacologic Substance': Pill,
  'Therapeutic or Preventive Procedure': (props: any) => <Logo size={18} {...props} />,
  'Laboratory Procedure': TestTube,
  'Body Part, Organ, or Organ Component': Brain,
  'Finding': Info,
  'Clinical Attribute': CheckCircle
};

const SEMANTIC_TYPE_COLORS = {
  'Disease or Syndrome': 'bg-red-100 text-red-800 border-red-200',
  'Sign or Symptom': 'bg-orange-100 text-orange-800 border-orange-200',
  'Pharmacologic Substance': 'bg-blue-100 text-blue-800 border-blue-200',
  'Therapeutic or Preventive Procedure': 'bg-green-100 text-green-800 border-green-200',
  'Laboratory Procedure': 'bg-purple-100 text-purple-800 border-purple-200',
  'Body Part, Organ, or Organ Component': 'bg-indigo-100 text-indigo-800 border-indigo-200',
  'Finding': 'bg-gray-100 text-gray-800 border-gray-200',
  'Clinical Attribute': 'bg-teal-100 text-teal-800 border-teal-200'
};

export default function EnhancedMedicalKnowledgeSearch() {
  // State management
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchType, setSearchType] = useState('comprehensive');
  const [includeRelationships, setIncludeRelationships] = useState(true);
  const [includeHierarchy, setIncludeHierarchy] = useState(true);
  const [selectedConcept, setSelectedConcept] = useState<MedicalConcept | null>(null);
  const [conceptExploration, setConceptExploration] = useState<any>(null);
  const [suggestions, setSuggestions] = useState<AutocompleteSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [serviceStatus, setServiceStatus] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('search');

  // Check service status on mount
  useEffect(() => {
    checkServiceStatus();
  }, []);

  const checkServiceStatus = async () => {
    try {
      const response = await fetch('/api/medical-knowledge/status');
      const data = await response.json();
      if (data.success) {
        setServiceStatus(data.status);
      }
    } catch (error) {
      console.error('Failed to check service status:', error);
    }
  };

  // Autocomplete functionality
  const handleInputChange = useCallback(async (value: string) => {
    setSearchQuery(value);
    
    if (value.length < 2) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    try {
      const response = await fetch(`/api/medical-knowledge/autocomplete?q=${encodeURIComponent(value)}&limit=8`);
      const data = await response.json();
      
      if (data.success && data.suggestions) {
        setSuggestions(data.suggestions);
        setShowSuggestions(true);
      }
    } catch (error) {
      console.error('Autocomplete failed:', error);
    }
  }, []);

  // Main search function
  const performSearch = async (query?: string) => {
    const searchTerm = query || searchQuery;
    if (!searchTerm.trim()) return;

    setIsSearching(true);
    setShowSuggestions(false);
    setSearchResults(null);

    try {
      const searchPayload = {
        query: searchTerm,
        search_type: searchType,
        max_results: 20,
        include_relationships: includeRelationships,
        include_hierarchy: includeHierarchy,
        clinical_context: {
          // Could add patient-specific context here
        }
      };

      const response = await fetch('/api/medical-knowledge/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(searchPayload)
      });

      const data = await response.json();
      
      if (data.success) {
        setSearchResults(data.search_result);
        setActiveTab('results');
      } else {
        throw new Error(data.error || 'Search failed');
      }
    } catch (error) {
      console.error('Search failed:', error);
      // Handle error state
    } finally {
      setIsSearching(false);
    }
  };

  // Concept exploration
  const exploreConcept = async (cui: string) => {
    if (!cui) return;

    try {
      setSelectedConcept(searchResults?.concepts.find(c => c.cui === cui) || null);
      
      const response = await fetch(`/api/medical-knowledge/concept/${cui}/explore?depth=2`);
      const data = await response.json();
      
      if (data.success) {
        setConceptExploration(data.concept_exploration);
        setActiveTab('exploration');
      }
    } catch (error) {
      console.error('Concept exploration failed:', error);
    }
  };

  // Handle suggestion selection
  const selectSuggestion = (suggestion: AutocompleteSuggestion) => {
    setSearchQuery(suggestion.text);
    setShowSuggestions(false);
    performSearch(suggestion.text);
  };

  // Handle Enter key
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      performSearch();
    }
  };

  // Get semantic type icon and color
  const getSemanticTypeDisplay = (semanticType: string) => {
    const IconComponent = SEMANTIC_TYPE_ICONS[semanticType as keyof typeof SEMANTIC_TYPE_ICONS] || Info;
    const colorClass = SEMANTIC_TYPE_COLORS[semanticType as keyof typeof SEMANTIC_TYPE_COLORS] || 'bg-gray-100 text-gray-800 border-gray-200';
    
    return {
      icon: IconComponent,
      colorClass
    };
  };
        
  // Render concept card
  const renderConceptCard = (concept: MedicalConcept, isMain: boolean = false) => (
    <Card key={concept.cui} className={`transition-all duration-200 hover:shadow-lg cursor-pointer ${isMain ? 'border-blue-200 bg-blue-50/30' : ''}`}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-lg font-semibold text-gray-900 mb-2">
              {concept.preferred_name}
            </CardTitle>
            <div className="flex items-center gap-2 mb-3">
              {concept.semantic_types.slice(0, 2).map((type, index) => {
                const { icon: IconComponent, colorClass } = getSemanticTypeDisplay(type);
                return (
                  <Badge key={index} variant="outline" className={`${colorClass} text-xs`}>
                    <IconComponent className="w-3 h-3 mr-1" />
                    {type}
                  </Badge>
                );
              })}
              {concept.semantic_types.length > 2 && (
                <Badge variant="outline" className="text-xs">
                  +{concept.semantic_types.length - 2} more
                </Badge>
              )}
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm font-medium text-gray-600">
              Confidence: {Math.round(concept.confidence * 100)}%
            </div>
            <div className="text-xs text-gray-500 mt-1">
              CUI: {concept.cui}
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {concept.definitions.length > 0 && (
          <div className="mb-4">
            <p className="text-sm text-gray-700 line-clamp-3">
              {concept.definitions[0]}
            </p>
          </div>
        )}
        
        {concept.synonyms.length > 0 && (
          <div className="mb-4">
            <p className="text-xs font-medium text-gray-600 mb-2">Also known as:</p>
            <div className="flex flex-wrap gap-1">
              {concept.synonyms.slice(0, 4).map((synonym, index) => (
                <Badge key={index} variant="secondary" className="text-xs">
                  {synonym}
                </Badge>
              ))}
              {concept.synonyms.length > 4 && (
                <Badge variant="secondary" className="text-xs">
                  +{concept.synonyms.length - 4} more
                </Badge>
              )}
            </div>
          </div>
        )}

        <div className="flex items-center justify-between mt-4">
          <div className="flex items-center gap-2">
            {concept.source_vocabularies.map((source, index) => (
              <Badge key={index} variant="outline" className="text-xs">
                {source}
              </Badge>
            ))}
          </div>
          
          <Button 
            size="sm" 
            variant="ghost" 
            onClick={() => exploreConcept(concept.cui)}
            className="text-blue-600 hover:text-blue-800"
          >
            <Network className="w-4 h-4 mr-1" />
            Explore
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center gap-3">
          <Brain className="text-blue-600" />
          Medical Knowledge Search
        </h1>
        <p className="text-gray-600">
          Intelligent search across UMLS, Knowledge Graphs, and Medical Ontologies
        </p>
      </div>

      {/* Service Status */}
      {serviceStatus && (
        <Card className="mb-6">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {serviceStatus.overall_status === 'fully_operational' ? (
                  <CheckCircle className="text-green-500" />
                ) : serviceStatus.overall_status === 'partially_operational' ? (
                  <AlertCircle className="text-yellow-500" />
                ) : (
                  <AlertCircle className="text-red-500" />
                )}
                <span className="font-medium">
                  Status: {serviceStatus.overall_status.replace('_', ' ').toUpperCase()}
                </span>
              </div>
              <div className="flex items-center gap-2">
                {serviceStatus.features_available.map((feature: string, index: number) => (
                  <Badge key={index} variant="secondary" className="text-xs">
                    {feature}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Search Interface */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="text-blue-600" />
            Search Medical Knowledge
          </CardTitle>
          <CardDescription>
            Search diseases, symptoms, medications, procedures, and medical concepts
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative mb-4">
            <Input
              type="text"
              placeholder="Enter medical term (e.g., 'diabetes', 'chest pain', 'metformin')..."
              value={searchQuery}
              onChange={(e) => handleInputChange(e.target.value)}
              onKeyPress={handleKeyPress}
              className="w-full pl-10 pr-20"
            />
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Button 
              onClick={() => performSearch()}
              disabled={!searchQuery.trim() || isSearching}
              className="absolute right-2 top-1/2 transform -translate-y-1/2"
              size="sm"
            >
              {isSearching ? (
                <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
              ) : (
                <Zap className="w-4 h-4" />
              )}
            </Button>

            {/* Autocomplete suggestions */}
            {showSuggestions && suggestions.length > 0 && (
              <div className="absolute top-full left-0 right-0 z-50 bg-white border rounded-md shadow-lg mt-1">
                <ScrollArea className="max-h-60">
                  {suggestions.map((suggestion, index) => (
                    <div
                      key={index}
                      className="px-4 py-3 hover:bg-gray-50 cursor-pointer border-b last:border-b-0"
                      onClick={() => selectSuggestion(suggestion)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium">{suggestion.text}</p>
                          <div className="flex items-center gap-2 mt-1">
                            {suggestion.semantic_types.slice(0, 2).map((type, idx) => {
                              const { colorClass } = getSemanticTypeDisplay(type);
                              return (
                                <Badge key={idx} variant="outline" className={`${colorClass} text-xs`}>
                                  {type}
                                </Badge>
                              );
                            })}
                            {suggestion.is_synonym && (
                              <Badge variant="outline" className="text-xs">
                                Synonym
                              </Badge>
                            )}
                          </div>
                        </div>
                        <div className="text-xs text-gray-500">
                          {Math.round(suggestion.confidence * 100)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </ScrollArea>
              </div>
            )}
          </div>

          {/* Search Options */}
          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-2">
              <label className="font-medium">Search Type:</label>
              <select 
                value={searchType} 
                onChange={(e) => setSearchType(e.target.value)}
                className="border rounded px-2 py-1"
              >
                <option value="comprehensive">Comprehensive</option>
                <option value="exact">Exact Match</option>
                <option value="semantic">Semantic</option>
                <option value="fuzzy">Fuzzy</option>
              </select>
            </div>
            
            <label className="flex items-center gap-2">
              <input 
                type="checkbox" 
                checked={includeRelationships} 
                onChange={(e) => setIncludeRelationships(e.target.checked)}
              />
              Include Relationships
            </label>
            
            <label className="flex items-center gap-2">
              <input 
                type="checkbox" 
                checked={includeHierarchy} 
                onChange={(e) => setIncludeHierarchy(e.target.checked)}
              />
              Include Hierarchy
            </label>
          </div>
        </CardContent>
      </Card>

      {/* Results Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="search">Search Results</TabsTrigger>
          <TabsTrigger value="results" disabled={!searchResults}>
            Detailed Analysis
          </TabsTrigger>
          <TabsTrigger value="exploration" disabled={!conceptExploration}>
            Concept Exploration
          </TabsTrigger>
        </TabsList>

        <TabsContent value="search" className="mt-6">
          <div className="text-center py-12">
            <Search className="mx-auto text-gray-400 w-12 h-12 mb-4" />
            <h3 className="text-lg font-semibold text-gray-600 mb-2">
              Search Medical Knowledge
            </h3>
            <p className="text-gray-500">
              Enter a medical term to explore concepts, relationships, and clinical context
            </p>
          </div>
        </TabsContent>

        <TabsContent value="results" className="mt-6">
          {searchResults && (
            <div className="space-y-6">
              {/* Search Summary */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>Search Results for "{searchResults.query}"</span>
                    <Badge variant="secondary">
                      {searchResults.total_results} results in {searchResults.execution_time.toFixed(2)}s
                    </Badge>
                  </CardTitle>
                  <CardDescription>
                    Sources: {searchResults.search_metadata.sources_used.join(', ')} • 
                    Search Type: {searchResults.search_metadata.search_type}
                  </CardDescription>
                </CardHeader>
                {searchResults.clinical_context.clinical_significance && (
                  <CardContent>
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <Info className="text-blue-600 w-4 h-4" />
                        <span className="font-medium text-blue-900">Clinical Significance</span>
                      </div>
                      <p className="text-blue-800 text-sm">
                        {searchResults.clinical_context.clinical_significance}
                      </p>
                    </div>
                  </CardContent>
                )}
              </Card>

              {/* Primary Results */}
              <div className="grid gap-4">
                <h3 className="text-lg font-semibold mb-2">Primary Concepts</h3>
                {searchResults.concepts.slice(0, 5).map((concept) => renderConceptCard(concept, true))}
              </div>

              {/* Similar Concepts */}
              {searchResults.similar_concepts.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold mb-4">Similar Concepts</h3>
                  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {searchResults.similar_concepts.map((concept) => renderConceptCard(concept))}
                  </div>
                </div>
              )}

              {/* Clinical Context */}
              {searchResults.clinical_context.primary_semantic_types.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Clinical Context Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <h4 className="font-medium mb-2">Primary Semantic Types</h4>
                        <div className="flex flex-wrap gap-2">
                          {searchResults.clinical_context.primary_semantic_types.map((item, index) => {
                            const { icon: IconComponent, colorClass } = getSemanticTypeDisplay(item.type);
                            return (
                              <Badge key={index} variant="outline" className={colorClass}>
                                <IconComponent className="w-3 h-3 mr-1" />
                                {item.type} ({item.count})
                              </Badge>
                            );
                          })}
                        </div>
                      </div>
                      
                      {Object.keys(searchResults.clinical_context.relationship_summary).length > 0 && (
                        <div>
                          <h4 className="font-medium mb-2">Relationship Summary</h4>
                          <div className="flex flex-wrap gap-2">
                            {Object.entries(searchResults.clinical_context.relationship_summary).map(([rel, count], index) => (
                              <Badge key={index} variant="secondary">
                                {rel.replace('_', ' ')}: {count}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </TabsContent>

        <TabsContent value="exploration" className="mt-6">
          {conceptExploration && selectedConcept && (
            <div className="space-y-6">
              {/* Concept Overview */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Network className="text-blue-600" />
                    Exploring: {selectedConcept.preferred_name}
                  </CardTitle>
                  <CardDescription>
                    CUI: {selectedConcept.cui} • Deep exploration of concept relationships and clinical context
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {selectedConcept.definitions.length > 0 && (
                    <p className="text-gray-700 mb-4">{selectedConcept.definitions[0]}</p>
                  )}
                  <div className="flex flex-wrap gap-2">
                    {selectedConcept.semantic_types.map((type, index) => {
                      const { icon: IconComponent, colorClass } = getSemanticTypeDisplay(type);
                      return (
                        <Badge key={index} variant="outline" className={colorClass}>
                          <IconComponent className="w-3 h-3 mr-1" />
                          {type}
                        </Badge>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>

              {/* Exploration sections would go here */}
              <div className="grid md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Clinical Pathways</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-500 text-sm">
                      Clinical pathways data would be displayed here
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Treatment Options</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-500 text-sm">
                      Treatment options would be displayed here
                    </p>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
