import { GoogleGenerativeAI } from '@google/generative-ai';
import natural from 'natural';
import { franc } from 'franc';
import nlp from 'compromise';

class MultiExplanationService {
  constructor(apiKey) {
    this.genAI = new GoogleGenerativeAI(apiKey);
    this.primaryModel = this.genAI.getGenerativeModel({ model: 'gemini-pro' });
    this.secondaryModel = this.genAI.getGenerativeModel({ model: 'gemini-pro' });
    this.tokenizer = new natural.TreebankWordTokenizer();
    this.sentimentAnalyzer = new natural.SentimentAnalyzer('English', natural.PorterStemmer, 'afinn');
    this.tfidf = new natural.TfIdf();
  }

  async generateMultipleExplanations(patientData) {
    const explanations = await Promise.all([
      this.generatePrimaryExplanation(patientData),
      this.generateSecondaryExplanation(patientData, 'clinical'),
      this.generateSecondaryExplanation(patientData, 'epidemiological'),
      this.generateSecondaryExplanation(patientData, 'pathophysiological'),
      this.generateSecondaryExplanation(patientData, 'differential')
    ]);

    return this.processExplanations(explanations, patientData);
  }

  async generatePrimaryExplanation(data) {
    const prompt = `Generate a detailed medical explanation for:
    Patient: ${data.age} year old ${data.gender}
    Symptoms: ${data.symptoms}
    Duration: ${data.duration}
    Physical Exam: ${data.physicalExam}
    Focus on the most likely diagnosis and clinical reasoning.`;

    const result = await this.primaryModel.generateContent(prompt);
    return {
      text: result.response.text(),
      type: 'primary',
      focus: 'clinical_presentation'
    };
  }

  async generateSecondaryExplanation(data, focus) {
    const focusPrompts = {
      clinical: 'Focus on clinical manifestations and symptom patterns',
      epidemiological: 'Focus on age, gender, and demographic risk factors',
      pathophysiological: 'Focus on underlying pathological mechanisms and tissue characteristics',
      differential: 'Focus on differential diagnoses and distinguishing features'
    };

    const prompt = `${focusPrompts[focus]}
    Patient: ${data.age} year old ${data.gender}
    Symptoms: ${data.symptoms}
    Duration: ${data.duration}
    Physical Exam: ${data.physicalExam}`;

    const result = await this.secondaryModel.generateContent(prompt);
    return {
      text: result.response.text(),
      type: 'secondary',
      focus: focus
    };
  }

  processExplanations(explanations, patientData) {
    return explanations.map((exp, index) => {
      const verification = this.verifyWithFOL(exp.text, patientData);
      const confidence = this.calculateConfidence(exp, verification, patientData);
      
      return {
        id: index + 1,
        explanation: exp.text,
        type: exp.type,
        focus: exp.focus,
        verification: verification,
        confidence: confidence,
        verified: confidence.score > 0.8
      };
    });
  }

  verifyWithFOL(text, patientData) {
    const predicates = this.extractPredicates(text, patientData);
    const folRules = this.defineFOLRules();
    const results = {};

    for (const [predicate, value] of Object.entries(predicates)) {
      results[predicate] = this.evaluatePredicate(predicate, value, folRules, patientData);
    }

    return {
      predicates: predicates,
      evaluations: results,
      overallValidity: Object.values(results).filter(r => r).length / Object.keys(results).length
    };
  }

  extractPredicates(text, patientData) {
    const doc = nlp(text);
    const predicates = {};

    // Extract age-related predicates
    if (doc.match('#Value (year|years) old').found) {
      predicates.ageMatches = text.includes(patientData.age.toString());
    }

    // Extract gender predicates
    predicates.genderMatches = text.toLowerCase().includes(patientData.gender.toLowerCase());

    // Extract symptom predicates
    const symptoms = patientData.symptoms.toLowerCase().split(/[,;]/);
    predicates.symptomsCovered = symptoms.filter(s => 
      text.toLowerCase().includes(s.trim())
    ).length / symptoms.length;

    // Extract duration predicates
    predicates.durationMentioned = this.extractDurationMatch(text, patientData.duration);

    // Extract location predicates
    const locations = this.extractAnatomicalLocations(text);
    predicates.locationAccuracy = this.matchLocations(locations, patientData.physicalExam);

    // Extract diagnostic predicates
    predicates.diagnosisMentioned = this.extractDiagnosis(text);

    // Extract malignancy predicates
    predicates.malignancyAssessment = this.assessMalignancyMention(text, patientData);

    return predicates;
  }

  defineFOLRules() {
    return {
      ageMatches: (age, patientAge) => age === patientAge,
      genderMatches: (mentioned, actual) => mentioned === actual,
      symptomsCovered: (coverage) => coverage > 0.7,
      durationMentioned: (match) => match,
      locationAccuracy: (accuracy) => accuracy > 0.8,
      diagnosisMentioned: (diagnosis) => diagnosis !== null,
      malignancyAssessment: (assessment) => assessment.appropriate
    };
  }

  evaluatePredicate(predicate, value, rules, patientData) {
    switch (predicate) {
      case 'ageMatches':
        return value === true;
      case 'genderMatches':
        return value === true;
      case 'symptomsCovered':
        return value > 0.7;
      case 'durationMentioned':
        return value === true;
      case 'locationAccuracy':
        return value > 0.8;
      case 'diagnosisMentioned':
        return value !== null && value.length > 0;
      case 'malignancyAssessment':
        return value.appropriate === true;
      default:
        return false;
    }
  }

  extractDurationMatch(text, actualDuration) {
    const durationPatterns = [
      /(\d+)\s*(month|months)/i,
      /(\d+)\s*(week|weeks)/i,
      /(\d+)\s*(year|years)/i,
      /slowly\s*enlarging/i,
      /gradual\s*growth/i,
      /extended\s*period/i
    ];

    return durationPatterns.some(pattern => pattern.test(text));
  }

  extractAnatomicalLocations(text) {
    const anatomicalTerms = [
      'anterior thigh', 'posterior thigh', 'lateral thigh', 'medial thigh',
      'fascia', 'subcutaneous', 'deep', 'superficial', 'muscle', 'bone'
    ];

    return anatomicalTerms.filter(term => 
      text.toLowerCase().includes(term)
    );
  }

  matchLocations(extracted, physicalExam) {
    const examLocations = physicalExam.toLowerCase().split(/[\s,;]+/);
    const matches = extracted.filter(loc => 
      examLocations.some(examLoc => examLoc.includes(loc) || loc.includes(examLoc))
    );

    return extracted.length > 0 ? matches.length / extracted.length : 0;
  }

  extractDiagnosis(text) {
    const diagnosisPatterns = [
      /diagnosis\s*(?:of|is)\s*([^,.\n]+)/i,
      /(?:suggests?|indicates?|consistent with)\s*([^,.\n]+)/i,
      /([a-z]+sarcoma|[a-z]+oma|[a-z]+itis)/gi
    ];

    for (const pattern of diagnosisPatterns) {
      const match = text.match(pattern);
      if (match) return match[1].trim();
    }

    return null;
  }

  assessMalignancyMention(text, patientData) {
    const malignancyTerms = ['malignant', 'sarcoma', 'cancer', 'metastasis', 'aggressive'];
    const benignTerms = ['benign', 'lipoma', 'cyst', 'inflammation'];
    
    const hasMalignant = malignancyTerms.some(term => text.toLowerCase().includes(term));
    const hasBenign = benignTerms.some(term => text.toLowerCase().includes(term));

    // FOL rule: Deep mass in older patient should mention malignancy consideration
    const shouldMentionMalignancy = 
      patientData.age > 40 && 
      patientData.physicalExam.toLowerCase().includes('deep');

    return {
      mentioned: hasMalignant || hasBenign,
      appropriate: shouldMentionMalignancy ? hasMalignant : true
    };
  }

  calculateConfidence(explanation, verification, patientData) {
    let score = 0;
    const weights = {
      folValidity: 0.3,
      sentimentConsistency: 0.1,
      keywordDensity: 0.2,
      coherenceScore: 0.2,
      completeness: 0.2
    };

    // FOL validity score
    score += verification.overallValidity * weights.folValidity;

    // Sentiment consistency
    const sentiment = this.analyzeSentiment(explanation.text);
    const sentimentScore = this.evaluateSentimentConsistency(sentiment, patientData);
    score += sentimentScore * weights.sentimentConsistency;

    // Keyword density
    const keywordScore = this.calculateKeywordDensity(explanation.text, patientData);
    score += keywordScore * weights.keywordDensity;

    // Coherence score
    const coherenceScore = this.evaluateCoherence(explanation.text);
    score += coherenceScore * weights.coherenceScore;

    // Completeness score
    const completeness = this.evaluateCompleteness(explanation.text, patientData);
    score += completeness * weights.completeness;

    return {
      score: Math.min(score, 1),
      details: {
        folValidity: verification.overallValidity,
        sentimentConsistency: sentimentScore,
        keywordDensity: keywordScore,
        coherence: coherenceScore,
        completeness: completeness
      }
    };
  }

  analyzeSentiment(text) {
    const tokens = this.tokenizer.tokenize(text);
    return this.sentimentAnalyzer.getSentiment(tokens);
  }

  evaluateSentimentConsistency(sentiment, patientData) {
    // Medical explanations should be neutral to slightly negative for concerning conditions
    const expectedSentiment = patientData.physicalExam.includes('deep') ? -0.3 : 0;
    const difference = Math.abs(sentiment - expectedSentiment);
    return Math.max(0, 1 - difference);
  }

  calculateKeywordDensity(text, patientData) {
    const medicalKeywords = [
      'diagnosis', 'clinical', 'presentation', 'examination',
      'symptoms', 'patient', 'treatment', 'prognosis',
      'differential', 'etiology', 'pathology', 'manifestation'
    ];

    const tokens = this.tokenizer.tokenize(text.toLowerCase());
    const keywordCount = tokens.filter(token => 
      medicalKeywords.includes(token)
    ).length;

    return Math.min(keywordCount / tokens.length * 10, 1);
  }

  evaluateCoherence(text) {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    if (sentences.length < 2) return 0.5;

    // Check for logical connectors
    const connectors = ['therefore', 'however', 'furthermore', 'additionally', 
                       'consequently', 'moreover', 'specifically'];
    const connectorCount = connectors.filter(conn => 
      text.toLowerCase().includes(conn)
    ).length;

    // Check for medical reasoning patterns
    const reasoningPatterns = [
      /due to|because of|caused by/i,
      /consistent with|suggestive of|indicative of/i,
      /differential diagnosis|rule out/i
    ];
    const reasoningCount = reasoningPatterns.filter(pattern => 
      pattern.test(text)
    ).length;

    return Math.min((connectorCount + reasoningCount) / sentences.length, 1);
  }

  evaluateCompleteness(text, patientData) {
    const requiredElements = [
      { element: 'age', present: text.includes(patientData.age.toString()) },
      { element: 'gender', present: text.toLowerCase().includes(patientData.gender.toLowerCase()) },
      { element: 'symptoms', present: this.checkSymptomsCoverage(text, patientData.symptoms) },
      { element: 'duration', present: this.checkDurationMention(text, patientData.duration) },
      { element: 'location', present: this.checkLocationMention(text, patientData.physicalExam) },
      { element: 'diagnosis', present: this.extractDiagnosis(text) !== null }
    ];

    const presentCount = requiredElements.filter(req => req.present).length;
    return presentCount / requiredElements.length;
  }

  checkSymptomsCoverage(text, symptoms) {
    const symptomList = symptoms.toLowerCase().split(/[,;]/);
    const coveredSymptoms = symptomList.filter(symptom => 
      text.toLowerCase().includes(symptom.trim())
    );
    return coveredSymptoms.length > symptomList.length * 0.5;
  }

  checkDurationMention(text, duration) {
    return text.toLowerCase().includes('month') || 
           text.toLowerCase().includes('week') || 
           text.toLowerCase().includes('duration') ||
           text.toLowerCase().includes('period');
  }

  checkLocationMention(text, physicalExam) {
    const locations = physicalExam.toLowerCase().split(/[\s,;]+/);
    return locations.some(loc => text.toLowerCase().includes(loc));
  }
}

export default MultiExplanationService;
