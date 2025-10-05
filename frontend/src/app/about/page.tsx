import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Brain, Shield, Zap, Users, Award, ArrowRight, CheckCircle, Target, Lightbulb } from "lucide-react"

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
    <section className="relative bg-gradient-to-br from-blue-700 via-blue-600/80 to-purple-500 py-24 px-4 overflow-hidden">
      <div className="absolute inset-0 bg-[url('/abstract-medical-neural-network-pattern.jpg')] opacity-10"></div>
      <div className="container mx-auto text-center relative z-10">
        <Badge className="mb-6 bg-accent text-accent-foreground px-4 py-2 text-sm font-medium">
        🏆 Breakthrough in Explainable Medical AI
        </Badge>
        <h1 className="text-5xl md:text-7xl font-bold text-primary-foreground mb-6 text-balance">CortexMD</h1>
        <p className="text-xl md:text-2xl text-primary-foreground/90 mb-8 max-w-3xl mx-auto text-pretty">
        Revolutionizing medical diagnosis with transparent AI that doctors can trust. Where cutting-edge multimodal
        intelligence meets rigorous logical verification.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <Button size="lg" className="bg-accent hover:bg-accent/90 text-accent-foreground">
          Explore Technology <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
        <Button
          size="lg"
          variant="outline"
          className="border-primary-foreground text-primary-foreground hover:bg-primary-foreground hover:text-primary bg-transparent"
        >
          View Research
        </Button>
        </div>
      </div>
    </section>

      {/* Problem Statement */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6">The Critical Challenge</h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto text-pretty">
              AI models in healthcare function as black boxes. Medical practitioners hesitate to trust AI predictions
              without clear, verifiable reasoning.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="border-destructive/20 bg-destructive/5">
              <CardHeader>
                <Target className="h-12 w-12 text-destructive mb-4" />
                <CardTitle className="text-destructive">Life-and-Death Stakes</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Doctors can't trust AI diagnoses without knowing the reasoning, and patients deserve clear
                  explanations.
                </p>
              </CardContent>
            </Card>

            <Card className="border-destructive/20 bg-destructive/5">
              <CardHeader>
                <Shield className="h-12 w-12 text-destructive mb-4" />
                <CardTitle className="text-destructive">No Structured Validation</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Most AI diagnosis methods lack a step to break explanations into verifiable, logic-based statements.
                </p>
              </CardContent>
            </Card>

            <Card className="border-destructive/20 bg-destructive/5">
              <CardHeader>
                <Brain className="h-12 w-12 text-destructive mb-4" />
                <CardTitle className="text-destructive">Hallucination Risk</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  LLMs often generate convincing but false medical logic, which can mislead doctors.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Solution Overview */}
      <section className="py-20 px-4 bg-muted/30">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6">Our Revolutionary Solution</h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto text-pretty">
              CortexMD transforms opaque AI into a reliable clinical assistant through innovative First-Order Logic
              verification and multimodal intelligence.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              <div className="flex items-start gap-4">
                <div className="bg-primary text-primary-foreground rounded-full p-3 flex-shrink-0">
                  <Brain className="h-6 w-6" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">MedGemma Diagnosis Generation</h3>
                  <p className="text-muted-foreground">
                    Advanced multimodal AI processes medical text, FHIR records, and imaging for comprehensive
                    diagnostic context.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="bg-secondary text-secondary-foreground rounded-full p-3 flex-shrink-0">
                  <Zap className="h-6 w-6" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">Multiple Reasoning Paths</h3>
                  <p className="text-muted-foreground">
                    Secondary LLM generates diverse reasoning pathways, ensuring comprehensive analysis of each
                    diagnosis.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="bg-accent text-accent-foreground rounded-full p-3 flex-shrink-0">
                  <Shield className="h-6 w-6" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">FOL-Based Verification</h3>
                  <p className="text-muted-foreground">
                    First-Order Logic converts explanations into verifiable statements, checked against patient data and
                    medical ontologies.
                  </p>
                </div>
              </div>
            </div>

            <Card className="p-8 bg-card">
              <div className="text-center">
                <div className="bg-primary/10 rounded-full p-6 w-24 h-24 mx-auto mb-6 flex items-center justify-center">
                  <CheckCircle className="h-12 w-12 text-primary" />
                </div>
                <h3 className="text-2xl font-bold mb-4">Confidence Scoring</h3>
                <p className="text-muted-foreground mb-6">
                  Present diagnosis with verified explanations and confidence levels, enabling informed medical
                  decisions.
                </p>
                <div className="bg-accent/10 rounded-lg p-4">
                  <p className="text-sm font-medium text-accent">Trusted AI • Better Outcomes</p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </section>

      {/* Innovation Highlights */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6">Innovation Highlights</h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto text-pretty">
              Novel fusion of symbolic logic and neural models for real-time, reliable decision support.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="border-primary/20 bg-primary/5">
              <CardHeader>
                <Lightbulb className="h-12 w-12 text-primary mb-4" />
                <CardTitle>Interpretability & Explainability</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Beyond basic XAI - we provide causal, semantic explanations instead of just highlighting regions.
                </p>
                <Badge variant="secondary">Breakthrough Technology</Badge>
              </CardContent>
            </Card>

            <Card className="border-secondary/20 bg-secondary/5">
              <CardHeader>
                <Zap className="h-12 w-12 text-secondary mb-4" />
                <CardTitle>Real-Time Processing</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  FOL predicates checked independently and in parallel, dramatically speeding up verification.
                </p>
                <Badge variant="secondary">Performance Optimized</Badge>
              </CardContent>
            </Card>

            <Card className="border-accent/20 bg-accent/5">
              <CardHeader>
                <Brain className="h-12 w-12 text-accent mb-4" />
                <CardTitle>Novel Data Synthesis</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">
                  Creates structured, symbolic knowledge base through explanation decomposition into FOL predicates.
                </p>
                <Badge variant="secondary">Data Innovation</Badge>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-20 px-4 bg-muted/30">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6">Meet Team Windows 12 Devs</h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto text-pretty">
              Innovative minds from Ramaiah Institute of Technology pushing the boundaries of medical AI.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="text-center hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="w-24 h-24 bg-primary/10 rounded-full mx-auto mb-4 flex items-center justify-center">
                  <Users className="h-12 w-12 text-primary" />
                </div>
                <CardTitle>Prathmesh Sayal</CardTitle>
                <CardDescription>Lead AI Researcher</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Specializing in multimodal AI and explainable machine learning for healthcare applications.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="w-24 h-24 bg-secondary/10 rounded-full mx-auto mb-4 flex items-center justify-center">
                  <Brain className="h-12 w-12 text-secondary" />
                </div>
                <CardTitle>Kshiraja Nelapati</CardTitle>
                <CardDescription>Logic Systems Engineer</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Expert in First-Order Logic verification and symbolic reasoning systems for medical AI.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="w-24 h-24 bg-blue-100 rounded-full mx-auto mb-4 flex items-center justify-center">
                  <Zap className="h-12 w-12 text-blue-500" />
                </div>
                <CardTitle>Omkar</CardTitle>
                <CardDescription>Clinical Integration Specialist</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Focused on real-world clinical applications and healthcare system integration.
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="text-center mt-12">
            <Card className="inline-block p-6 bg-primary/5 border-primary/20">
              <div className="flex items-center gap-4">
                <Award className="h-8 w-8 text-primary" />
                <div className="text-left">
                  <p className="font-semibold">Ramaiah Institute of Technology</p>
                  <p className="text-sm text-muted-foreground">Estimated Project Cost: ₹4,70,000</p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </section>

      {/* Impact & Research */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6">Clinical & Societal Impact</h2>
          </div>

          <div className="grid lg:grid-cols-2 gap-12">
            <Card className="p-8 bg-gradient-to-br from-blue-50 to-purple-50 border border-blue-200/50">
              <CardHeader className="pb-4">
                <CardTitle className="text-2xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">For Healthcare Professionals</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-3">
                  <CheckCircle className="h-5 w-5 text-blue-500 flex-shrink-0" />
                  <p className="text-slate-700">Boosts clinician confidence in AI-assisted diagnosis</p>
                </div>
                <div className="flex items-center gap-3">
                  <CheckCircle className="h-5 w-5 text-purple-500 flex-shrink-0" />
                  <p className="text-slate-700">Reduces diagnostic errors through transparent reasoning</p>
                </div>
                <div className="flex items-center gap-3">
                  <CheckCircle className="h-5 w-5 text-blue-500 flex-shrink-0" />
                  <p className="text-slate-700">Enables safe adoption of advanced AI for better outcomes</p>
                </div>
              </CardContent>
            </Card>

            <Card className="p-8">
              <CardHeader className="pb-4">
                <CardTitle className="text-2xl">For Patients</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-3">
                  <CheckCircle className="h-5 w-5 text-accent flex-shrink-0" />
                  <p>Increases transparency in medical decision-making</p>
                </div>
                <div className="flex items-center gap-3">
                  <CheckCircle className="h-5 w-5 text-accent flex-shrink-0" />
                  <p>Builds trust through understandable explanations</p>
                </div>
                <div className="flex items-center gap-3">
                  <CheckCircle className="h-5 w-5 text-accent flex-shrink-0" />
                  <p>Empowers with clear, comprehensible health insights</p>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="text-center mt-12">
            <Card className="inline-block p-6 bg-accent/5 border-accent/20">
              <p className="text-sm text-muted-foreground mb-2">Research Validation</p>
              <p className="font-semibold">
                Our approach demonstrates superior performance compared to traditional diagnostic methods
              </p>
              <Button variant="link" className="text-accent p-0 h-auto mt-2">
                View Research Paper →
              </Button>
            </Card>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-20 px-4 bg-gradient-to-r from-primary to-secondary">
        <div className="container mx-auto text-center">
          <h2 className="text-4xl md:text-5xl font-bold text-primary-foreground mb-6 text-balance">
            Ready to Transform Medical AI?
          </h2>
          <p className="text-xl text-primary-foreground/90 mb-8 max-w-2xl mx-auto text-pretty">
            Join us in revolutionizing healthcare with transparent, trustworthy AI that puts doctors and patients first.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" className="bg-accent hover:bg-accent/90 text-accent-foreground">
              Explore Platform
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="border-primary-foreground text-primary-foreground hover:bg-primary-foreground hover:text-primary bg-transparent"
            >
              Contact Team
            </Button>
          </div>
        </div>
      </section>
    </div>
  )
}
