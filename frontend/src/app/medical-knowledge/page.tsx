/**
 * Medical Knowledge Search Page
 * Dedicated page for enhanced medical knowledge exploration
 */

import { Metadata } from 'next';
import EnhancedMedicalKnowledgeSearch from '@/components/knowledge/EnhancedMedicalKnowledgeSearch';

export const metadata: Metadata = {
  title: 'Medical Knowledge Search | CortexMD',
  description: 'Intelligent medical knowledge search across UMLS, Neo4j Knowledge Graphs, and Medical Ontologies',
  keywords: ['medical knowledge', 'UMLS', 'medical ontology', 'medical concepts', 'healthcare search', 'clinical decision support'],
};

export default function MedicalKnowledgeSearchPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      <EnhancedMedicalKnowledgeSearch />
    </div>
  );
}
