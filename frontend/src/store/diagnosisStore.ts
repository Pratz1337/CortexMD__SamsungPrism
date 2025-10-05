import { create } from 'zustand';
import { PatientInput, DiagnosisResult, ChatMessage, ProcessingStatus } from '@/types';

interface DiagnosisStore {
  // Patient Input State
  patientInput: PatientInput;
  setPatientInput: (input: Partial<PatientInput>) => void;
  resetPatientInput: () => void;

  // Current Session
  currentSessionId: string | null;
  setCurrentSessionId: (sessionId: string | null) => void;

  // Diagnosis Results
  diagnosisResults: DiagnosisResult | null;
  setDiagnosisResults: (results: DiagnosisResult | null) => void;

  // Processing Status
  processingStatus: ProcessingStatus | null;
  setProcessingStatus: (status: ProcessingStatus | null) => void;

  // Chat Messages
  chatMessages: ChatMessage[];
  addChatMessage: (message: ChatMessage) => void;
  clearChatMessages: () => void;

  // Loading States
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;

  // Error State
  error: string | null;
  setError: (error: string | null) => void;
}

const initialPatientInput: PatientInput = {
  symptoms: '',
  medical_history: '',
  age: 0,
  gender: 'other',
  current_medications: '',
  allergies: '',
  vital_signs: {},
  images: [],
  fhir_data: '',
};

export const useDiagnosisStore = create<DiagnosisStore>((set, get) => ({
  // Patient Input State
  patientInput: initialPatientInput,
  setPatientInput: (input) =>
    set((state) => ({
      patientInput: { ...state.patientInput, ...input },
    })),
  resetPatientInput: () => set({ patientInput: initialPatientInput }),

  // Current Session
  currentSessionId: null,
  setCurrentSessionId: (sessionId) => set({ currentSessionId: sessionId }),

  // Diagnosis Results
  diagnosisResults: null,
  setDiagnosisResults: (results) => set({ diagnosisResults: results }),

  // Processing Status
  processingStatus: null,
  setProcessingStatus: (status) => set({ processingStatus: status }),

  // Chat Messages
  chatMessages: [],
  addChatMessage: (message) =>
    set((state) => ({
      chatMessages: [...state.chatMessages, message],
    })),
  clearChatMessages: () => set({ chatMessages: [] }),

  // Loading States
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),

  // Error State
  error: null,
  setError: (error) => set({ error }),
}));
