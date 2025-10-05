"use client";

import React, { useEffect, useRef, useState } from "react";
import { motion, useMotionTemplate, useMotionValue, animate } from "framer-motion";
import { 
  Activity, 
  Brain, 
  Database, 
  FileText, 
  Shield, 
  Zap, 
  ArrowRight, 
  CheckCircle, 
  Users, 
  Clock, 
  Target,
  Eye,
  Network,
  Cpu
} from "lucide-react";
import { HealthcareLoadingScreen } from "@/components/ui/HealthcareLoadingScreen"

// Utility function for className merging
function cn(...classes: (string | undefined | null | false)[]): string {
  return classes.filter(Boolean).join(' ');
}
  import Logo from '@/components/ui/Logo'

// Button component
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'outline' | 'ghost';
  size?: 'default' | 'sm' | 'lg';
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', ...props }, ref) => {
    const baseClasses = "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50";
    
    const variants = {
      default: "bg-primary text-primary-foreground hover:bg-primary/90",
      outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
      ghost: "hover:bg-accent hover:text-accent-foreground"
    };
    
    const sizes = {
      default: "h-10 px-4 py-2",
      sm: "h-9 rounded-md px-3",
      lg: "h-11 rounded-md px-8"
    };

    return (
      <button
        className={cn(baseClasses, variants[variant], sizes[size], className)}
        ref={ref}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";

// Card components
const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn("rounded-lg border bg-card text-card-foreground shadow-sm", className)}
      {...props}
    />
  )
);
Card.displayName = "Card";

const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
  )
);
CardHeader.displayName = "CardHeader";

const CardTitle = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn("text-2xl font-semibold leading-none tracking-tight", className)}
      {...props}
    />
  )
);
CardTitle.displayName = "CardTitle";

const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
  )
);
CardContent.displayName = "CardContent";

// Floating Paths Background Component
function FloatingPaths({ position }: { position: number }) {
  const paths = Array.from({ length: 24 }, (_, i) => ({
    id: i,
    d: `M-${280 - i * 4 * position} -${150 + i * 5}C-${280 - i * 4 * position} -${150 + i * 5} -${250 - i * 4 * position} ${180 - i * 5} ${120 - i * 4 * position} ${280 - i * 5}C${490 - i * 4 * position} ${380 - i * 5} ${550 - i * 4 * position} ${650 - i * 5} ${550 - i * 4 * position} ${650 - i * 5}`,
    color: `rgba(59,130,246,${0.05 + i * 0.02})`,
    width: 0.3 + i * 0.02,
  }));

  return (
    <div className="absolute inset-0 pointer-events-none">
      <svg
        className="w-full h-full text-blue-500"
        viewBox="0 0 696 316"
        fill="none"
      >
        <title>Medical Background Paths</title>
        {paths.map((path) => (
          <motion.path
            key={path.id}
            d={path.d}
            stroke="currentColor"
            strokeWidth={path.width}
            strokeOpacity={0.05 + path.id * 0.02}
            initial={{ pathLength: 0.2, opacity: 0.4 }}
            animate={{
              pathLength: 1,
              opacity: [0.2, 0.4, 0.2],
              pathOffset: [0, 1, 0],
            }}
            transition={{
              duration: 25 + Math.random() * 15,
              repeat: Infinity,
              ease: "linear",
            }}
          />
        ))}
      </svg>
    </div>
  );
}

// Glow Card Component
interface GlowCardProps {
  children: React.ReactNode;
  className?: string;
  glowColor?: 'blue' | 'purple' | 'green' | 'red' | 'orange';
}

const glowColorMap = {
  blue: { base: 220, spread: 200 },
  purple: { base: 280, spread: 300 },
  green: { base: 120, spread: 200 },
  red: { base: 0, spread: 200 },
  orange: { base: 30, spread: 200 }
};

function GlowCard({ children, className = '', glowColor = 'blue' }: GlowCardProps) {
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const syncPointer = (e: PointerEvent) => {
      const { clientX: x, clientY: y } = e;
      
      if (cardRef.current) {
        cardRef.current.style.setProperty('--x', x.toFixed(2));
        cardRef.current.style.setProperty('--xp', (x / window.innerWidth).toFixed(2));
        cardRef.current.style.setProperty('--y', y.toFixed(2));
        cardRef.current.style.setProperty('--yp', (y / window.innerHeight).toFixed(2));
      }
    };

    document.addEventListener('pointermove', syncPointer);
    return () => document.removeEventListener('pointermove', syncPointer);
  }, []);

  const { base, spread } = glowColorMap[glowColor];

  const getInlineStyles = () => ({
    '--base': base,
    '--spread': spread,
    '--radius': '14',
    '--border': '2',
    '--backdrop': 'hsl(0 0% 60% / 0.08)',
    '--backup-border': 'var(--backdrop)',
    '--size': '150',
    '--outer': '1',
    '--border-size': 'calc(var(--border, 2) * 1px)',
    '--spotlight-size': 'calc(var(--size, 150) * 1px)',
    '--hue': 'calc(var(--base) + (var(--xp, 0) * var(--spread, 0)))',
    backgroundImage: `radial-gradient(
      var(--spotlight-size) var(--spotlight-size) at
      calc(var(--x, 0) * 1px)
      calc(var(--y, 0) * 1px),
      hsl(var(--hue, 210) calc(var(--saturation, 100) * 1%) calc(var(--lightness, 70) * 1%) / var(--bg-spot-opacity, 0.05)), transparent
    )`,
    backgroundColor: 'var(--backdrop, transparent)',
    backgroundSize: 'calc(100% + (2 * var(--border-size))) calc(100% + (2 * var(--border-size)))',
    backgroundPosition: '50% 50%',
    backgroundAttachment: 'fixed',
    border: 'var(--border-size) solid var(--backup-border)',
    position: 'relative' as const,
    touchAction: 'none' as const,
  });

  const beforeAfterStyles = `
    [data-glow]::before,
    [data-glow]::after {
      pointer-events: none;
      content: "";
      position: absolute;
      inset: calc(var(--border-size) * -1);
      border: var(--border-size) solid transparent;
      border-radius: calc(var(--radius) * 1px);
      background-attachment: fixed;
      background-size: calc(100% + (2 * var(--border-size))) calc(100% + (2 * var(--border-size)));
      background-repeat: no-repeat;
      background-position: 50% 50%;
      mask: linear-gradient(transparent, transparent), linear-gradient(white, white);
      mask-clip: padding-box, border-box;
      mask-composite: intersect;
    }
    
    [data-glow]::before {
      background-image: radial-gradient(
        calc(var(--spotlight-size) * 0.75) calc(var(--spotlight-size) * 0.75) at
        calc(var(--x, 0) * 1px)
        calc(var(--y, 0) * 1px),
        hsl(var(--hue, 210) calc(var(--saturation, 100) * 1%) calc(var(--lightness, 50) * 1%) / var(--border-spot-opacity, 0.3)), transparent 100%
      );
      filter: brightness(1.5);
    }
    
    [data-glow]::after {
      background-image: radial-gradient(
        calc(var(--spotlight-size) * 0.5) calc(var(--spotlight-size) * 0.5) at
        calc(var(--x, 0) * 1px)
        calc(var(--y, 0) * 1px),
        hsl(0 100% 100% / var(--border-light-opacity, 0.2)), transparent 100%
      );
    }
  `;

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: beforeAfterStyles }} />
      <div
        ref={cardRef}
        data-glow
        style={getInlineStyles()}
        className={cn(
          "rounded-2xl relative backdrop-blur-sm p-6",
          className
        )}
      >
        {children}
      </div>
    </>
  );
}

// Rainbow Button Component
interface RainbowButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {}

function RainbowButton({ children, className, ...props }: RainbowButtonProps) {
  return (
    <>
      <style>{`
        :root {
          --color-1: 220 100% 63%;
          --color-2: 270 100% 63%;
          --color-3: 210 100% 63%;
          --color-4: 195 100% 63%;
          --color-5: 90 100% 63%;
        }
        @keyframes rainbow {
          0% { background-position: 0%; }
          100% { background-position: 200%; }
        }
        .animate-rainbow {
          animation: rainbow 2s infinite linear;
        }
      `}</style>
      <button
        className={cn(
          "group relative inline-flex h-11 animate-rainbow cursor-pointer items-center justify-center rounded-xl border-0 bg-[length:200%] px-8 py-2 font-medium text-primary-foreground transition-colors [background-clip:padding-box,border-box,border-box] [background-origin:border-box] [border:calc(0.08*1rem)_solid_transparent] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50",
          "before:absolute before:bottom-[-20%] before:left-1/2 before:z-0 before:h-1/5 before:w-3/5 before:-translate-x-1/2 before:animate-rainbow before:bg-[linear-gradient(90deg,hsl(var(--color-1)),hsl(var(--color-5)),hsl(var(--color-3)),hsl(var(--color-4)),hsl(var(--color-2)))] before:bg-[length:200%] before:[filter:blur(calc(0.8*1rem))]",
          "bg-[linear-gradient(#121213,#121213),linear-gradient(#121213_50%,rgba(18,18,19,0.6)_80%,rgba(18,18,19,0)),linear-gradient(90deg,hsl(var(--color-1)),hsl(var(--color-5)),hsl(var(--color-3)),hsl(var(--color-4)),hsl(var(--color-2)))]",
          "dark:bg-[linear-gradient(#fff,#fff),linear-gradient(#fff_50%,rgba(255,255,255,0.6)_80%,rgba(0,0,0,0)),linear-gradient(90deg,hsl(var(--color-1)),hsl(var(--color-5)),hsl(var(--color-3)),hsl(var(--color-4)),hsl(var(--color-2)))]",
          className,
        )}
        {...props}
      >
        {children}
      </button>
    </>
  );
}

// Main CortexMD Landing Page Component
function CortexMDLanding({ handleLaunchApp }: { handleLaunchApp: () => void }) {
  const color = useMotionValue("#3b82f6");

  useEffect(() => {
    animate(color, ["#3b82f6", "#1e40af", "#2563eb", "#3b82f6"], {
      ease: "easeInOut",
      duration: 8,
      repeat: Infinity,
      repeatType: "mirror",
    });
  }, [color]);

  const backgroundImage = useMotionTemplate`radial-gradient(125% 125% at 50% 0%, #ffffff 50%, ${color})`;

  const features = [
    {
      icon: Brain,
      title: "Multimodal AI Fusion",
      description: "Seamlessly integrates medical imaging, FHIR data, and clinical notes for comprehensive analysis"
    },
    {
      icon: Eye,
      title: "Explainable Reasoning",
      description: "Every diagnosis comes with verifiable reasoning chains and evidence-based explanations"
    },
    {
      icon: Database,
      title: "FHIR Integration",
      description: "Native support for structured clinical data standards and interoperability"
    },
    {
      icon: FileText,
      title: "AR Clinical Notes",
      description: "Advanced scanning and interpretation of unstructured medical documentation"
    },
    {
      icon: Shield,
      title: "Reduced Hallucinations",
      description: "FOL verification ensures accuracy and minimizes AI-generated errors"
    },
    {
      icon: Zap,
      title: "Faster Diagnosis",
      description: "Accelerate clinical decision-making with AI-powered insights"
    }
  ];

  const benefits = [
    { icon: Target, title: "98% Diagnostic Accuracy", description: "Clinically validated performance" },
    { icon: Clock, title: "75% Faster Diagnosis", description: "Reduce time to treatment" },
    { icon: Users, title: "500+ Hospitals", description: "Trusted by healthcare leaders" }
  ];

  const techStack = [
    { name: "MedGemma", description: "Medical language model" },
    { name: "FOL Verification", description: "First-order logic validation" },
    { name: "UMLS Ontologies", description: "Medical knowledge graphs" },
    { name: "Neo4j", description: "Graph database technology" }
  ];

  return (
    <motion.div 
      style={{ backgroundImage }}
      className="min-h-screen bg-gradient-to-br from-blue-50 to-white relative overflow-hidden"
    >
      {/* Background Elements */}
      <div className="absolute inset-0">
        <FloatingPaths position={1} />
        <FloatingPaths position={-1} />
      </div>

      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-lg border-b border-blue-100">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-800 rounded-xl flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold text-gray-900">CortexMD</span>
          </div>
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-gray-600 hover:text-blue-600 transition-colors">
              Features
            </a>
            <a href="#technology" className="text-gray-600 hover:text-blue-600 transition-colors">
              Technology
            </a>
            <a href="#contact" className="text-gray-600 hover:text-blue-600 transition-colors">
              Contact
            </a>
            <Button 
              onClick={handleLaunchApp}
              className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white"
            >
              Launch Platform
            </Button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 pt-32 pb-16 px-4">
        <div className="container mx-auto text-center max-w-6xl">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-blue-100 text-blue-800 text-sm font-medium mb-8">
              <Logo size={20} />
              AI-Powered Clinical Diagnostics
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold mb-6 text-gray-900 leading-tight">
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-blue-800">
                CortexMD
              </span>
              <br />
              <span className="text-4xl md:text-5xl font-normal text-gray-700">
                Multimodal AI Clinical Assistant
              </span>
            </h1>
            
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto leading-relaxed">
              Fusing imaging, structured clinical data (FHIR), and unstructured medical text to produce 
              diagnoses with verifiable, explainable reasoning that clinicians can trust.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <RainbowButton 
                onClick={handleLaunchApp}
                className="text-white"
              >
                Start Free Trial
                <ArrowRight className="ml-2 w-4 h-4" />
              </RainbowButton>
              <Button variant="outline" size="lg" className="border-blue-200 text-blue-700 hover:bg-blue-50">
                Schedule Demo
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="relative z-10 py-16 px-4">
        <div className="container mx-auto max-w-6xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Advanced AI Capabilities
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Cutting-edge technology designed specifically for clinical excellence
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <GlowCard className="h-full bg-white/80 backdrop-blur-sm border border-blue-100">
                  <div className="p-6">
                    <feature.icon className="w-12 h-12 text-blue-600 mb-4" />
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600">
                      {feature.description}
                    </p>
                  </div>
                </GlowCard>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="relative z-10 py-16 px-4 bg-gradient-to-r from-blue-600 to-blue-700">
        <div className="container mx-auto max-w-6xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Proven Clinical Impact
            </h2>
            <p className="text-lg text-blue-100 max-w-2xl mx-auto">
              Real results from healthcare institutions worldwide
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {benefits.map((benefit, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="text-center"
              >
                <benefit.icon className="w-16 h-16 text-blue-200 mx-auto mb-4" />
                <h3 className="text-2xl font-bold text-white mb-2">
                  {benefit.title}
                </h3>
                <p className="text-blue-100">
                  {benefit.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Stack */}
      <section className="relative z-10 py-16 px-4">
        <div className="container mx-auto max-w-6xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Powered by Advanced Technology
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Built on the latest breakthroughs in medical AI and knowledge representation
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {techStack.map((tech, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <Card className="h-full bg-gradient-to-br from-gray-50 to-white border-gray-200 hover:shadow-lg transition-shadow">
                  <CardHeader className="pb-2">
                    <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-3">
                      <Network className="w-6 h-6 text-blue-600" />
                    </div>
                    <CardTitle className="text-lg">{tech.name}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-600 text-sm">{tech.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 py-20 px-4">
        <div className="container mx-auto max-w-4xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
              Ready to Transform Clinical Diagnostics?
            </h2>
            <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto">
              Join leading healthcare institutions using CortexMD to improve patient outcomes 
              and accelerate clinical decision-making.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <RainbowButton onClick={handleLaunchApp} className="text-white text-lg px-8 py-3">
                Get Started Today
                <ArrowRight className="ml-2 w-5 h-5" />
              </RainbowButton>
              <Button variant="outline" size="lg" className="border-blue-200 text-blue-700 hover:bg-blue-50">
                Contact Sales
              </Button>
            </div>

            <div className="mt-8 flex items-center justify-center gap-6 text-sm text-gray-500">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                HIPAA Compliant
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                FDA Cleared
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                24/7 Support
              </div>
            </div>
          </motion.div>
        </div>
      </section>
    </motion.div>
  );
}

export default function LandingPage() {
  const [showApp, setShowApp] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const handleLaunchApp = () => {
    setIsLoading(true)
    // Simulate loading time for database connection
    setTimeout(() => {
      setShowApp(true)
      setIsLoading(false)
    }, 2000)
  }

  if (isLoading) {
    return <HealthcareLoadingScreen variant="heartbeat" message="Connecting to patient database..." />
  }

  if (showApp) {
    return <CortexMDApp />
  }

  return <CortexMDLanding handleLaunchApp={handleLaunchApp} />
}

function CortexMDApp() {
  const [activeView, setActiveView] = useState<"patients" | "new-patient">("patients")
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(null)

  // Import components dynamically to avoid initial load
  const PatientSelector = require("@/components/patient/PatientSelector").PatientSelector
  const PatientDashboard = require("@/components/patient/PatientDashboard").PatientDashboard
  const NewPatientForm = require("@/components/patient/NewPatientForm").NewPatientForm
  const SystemHealthCard = require("@/components/ui/SystemHealthCard").SystemHealthCard

  const renderContent = () => {
    if (selectedPatientId) {
      return <PatientDashboard patientId={selectedPatientId} onBack={() => setSelectedPatientId(null)} />
    }

    switch (activeView) {
      case "patients":
        return <PatientSelector onSelectPatient={setSelectedPatientId} />
      case "new-patient":
        return (
          <NewPatientForm
            onPatientCreated={(patientId:any) => {
              setActiveView("patients")
              setSelectedPatientId(patientId)
            }}
          />
        )
      default:
        return <PatientSelector onSelectPatient={setSelectedPatientId} />
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <div className="container mx-auto px-6 py-8">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-8 p-6 bg-white/70 backdrop-blur-xl rounded-2xl border border-white/20 shadow-xl shadow-blue-500/10">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/25">
                <Brain className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  CortexMD Platform
                </h1>
                <p className="text-sm text-primary font-medium">Advanced Medical Intelligence System</p>
              </div>
            </div>
            {!selectedPatientId && (
              <div className="flex gap-3">
                <Button
                  onClick={() => setActiveView("patients")}
                  variant={activeView === "patients" ? "default" : "outline"}
                  className={`px-6 py-3 font-semibold transition-all duration-300 ${
                    activeView === "patients"
                      ? "bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg shadow-blue-500/25"
                      : "bg-white/50 hover:bg-white/80 text-primary border-gray-200 hover:border-blue-300 hover:shadow-md"
                  }`}
                >
                  <Users className="w-4 h-4 mr-2" />
                  Patients
                </Button>
                <Button
                  onClick={() => setActiveView("new-patient")}
                  variant={activeView === "new-patient" ? "default" : "outline"}
                  className={`px-6 py-3 font-semibold transition-all duration-300 ${
                    activeView === "new-patient"
                      ? "bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg shadow-blue-500/25"
                      : "bg-white/50 hover:bg-white/80 text-primary border-gray-200 hover:border-blue-300 hover:shadow-md"
                  }`}
                >
                  New Patient
                </Button>
              </div>
            )}
          </div>

          {!selectedPatientId && (
            <div className="mb-6">
              <SystemHealthCard />
            </div>
          )}
        </div>

        <div className="bg-white/70 backdrop-blur-xl rounded-2xl border border-white/20 shadow-2xl shadow-blue-500/10 overflow-hidden">
          <div className="p-8">{renderContent()}</div>
        </div>
      </div>
    </div>
  )
}
